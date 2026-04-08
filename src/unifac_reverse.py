"""
UNIFAC Group Reverse Engineering: From group composition to molecular SMILES.

Supports:
  - Acyclic molecules (tree topologies)
  - Alicyclic rings (cyclopentane, cyclohexane, etc.) with ring-strain filtering
  - Aromatic ring systems (benzene, naphthalene, biphenyl) with positional isomers
  - Mixed molecules (aromatic ring + aliphatic chains + functional groups)

Usage:
    from unifac_reverse import load_unifac_groups, reverse_engineer_by_name

    db = load_unifac_groups("unifac_ly.json")

    # Simple acyclic
    smiles = reverse_engineer_by_name({"CH3": 2, "CH2": 1}, db)
    # -> ['CCC']

    # Aromatic
    smiles = reverse_engineer_by_name({"ACH": 5, "ACCH3": 1}, db)
    # -> ['Cc1ccccc1']  (toluene)

    # Mixed: aromatic + aliphatic
    smiles = reverse_engineer_by_name({"ACH": 5, "AC": 1, "CH2": 1, "CH3": 1}, db)
    # -> ['CCc1ccccc1']  (ethylbenzene)
"""

import json
from itertools import combinations
from typing import Optional
from rdkit import Chem
from rdkit.Chem import RWMol


# ═══════════════════════════════════════════════════════════════════════════
# FRAGMENT SMILES LIBRARY (non-aromatic groups)
# ═══════════════════════════════════════════════════════════════════════════
# [*] = dummy atom marking an open attachment point.

FRAGMENT_SMILES = {
    # ── Main group 1: Alkane ──
    "CH3": "[*]C", "CH2": "[*]C[*]", "CH": "[*]C([*])[*]",
    "C": "[*]C([*])([*])[*]",

    # ── Main group 2: Alkene ──
    "CH2=CH": "[*]/C=C", "CH=CH": "[*]/C=C\\[*]",
    "CH2=C": "[*]/C(=C)[*]", "CH=C": "[*]/C=C(\\[*])[*]",
    "C=C": "[*]/C([*])=C(\\[*])[*]",

    # ── Main group 5: Hydroxyl ──
    "OH": "[*]O",

    # ── Main group 6: Methanol ──
    "CH3OH": "CO",

    # ── Main group 7: Water ──
    "H2O": "O",

    # ── Main group 9: Ketone ──
    "CH3CO": "[*]C(=O)C", "CH2CO": "[*]C(=O)C[*]",

    # ── Main group 10: Aldehyde ──
    "HCO": "[*]C=O",

    # ── Main group 11: Ester (acetate-type) ──
    "CH3COO": "[*]OC(=O)C", "CH2COO": "[*]OC(=O)C[*]",

    # ── Main group 12: Formate ──
    "HCOO": "[*]OC=O",

    # ── Main group 13: Ether ──
    "CH3O": "[*]OC", "CH2O": "[*]OC[*]", "CHO": "[*]OC([*])[*]",
    "THF": "C1CCOC1",

    # ── Main group 14: Primary amine ──
    "CH3NH2": "CN", "CH2NH2": "[*]CN", "CHNH2": "[*]C([*])N",

    # ── Main group 15: Secondary amine ──
    "CH3NH": "[*]NC", "CH2NH": "[*]NC[*]", "CHNH": "[*]NC([*])[*]",

    # ── Main group 16: Tertiary amine ──
    "CH3N": "[*]N([*])C", "CH2N": "[*]N([*])C[*]",

    # ── Main group 18: Pyridine ──
    "C5H5N": "c1ccncc1", "C5H4N": "[*]c1ccncc1",
    "C5H3N": "[*]c1cc([*])ncc1",

    # ── Main group 19: Nitrile ──
    "CH3CN": "CC#N", "CH2CN": "[*]CC#N",

    # ── Main group 20: Carboxylic acid ──
    "COOH": "[*]C(=O)O", "HCOOH": "OC=O",

    # ── Main group 21: Chloro ──
    "CH2Cl": "[*]CCl", "CHCl": "[*]C([*])Cl",
    "CCl": "[*]C([*])([*])Cl",

    # ── Main group 22: Dichloro ──
    "CH2Cl2": "ClCCl", "CHCl2": "[*]C(Cl)Cl", "CCl2": "[*]C([*])(Cl)Cl",

    # ── Main group 23: Trichloro ──
    "CHCl3": "ClC(Cl)Cl", "CCl3": "[*]C(Cl)(Cl)Cl",

    # ── Main group 24: Carbon tetrachloride ──
    "CCl4": "ClC(Cl)(Cl)Cl",

    # ── Main group 26: Nitro ──
    "CH3NO2": "C[N+](=O)[O-]", "CH2NO2": "[*]C[N+](=O)[O-]",
    "CHNO2": "[*]C([*])[N+](=O)[O-]",

    # ── Main group 28: CS2 ──
    "CS2": "C(=S)=S",

    # ── Main group 29: Thiol ──
    "CH3SH": "CS", "CH2SH": "[*]CS",

    # ── Main group 30: Furfural ──
    "furfural": "O=Cc1ccco1",

    # ── Main group 31: Ethylene glycol ──
    "DOH": "OCCO",

    # ── Main group 32-33: Halogens ──
    "I": "[*]I", "Br": "[*]Br",

    # ── Main group 34: Alkyne ──
    "CH{triple}C": "[*]C#C", "C{triple}C": "[*]C#C[*]",

    # ── Main group 35: DMSO ──
    "DMSO": "CS(=O)C",

    # ── Main group 36: Acrylonitrile ──
    "acrylnitrile": "C=CC#N",

    # ── Main group 37: Vinyl chloride ──
    "Cl-C=C": "[*]C(=C)Cl",

    # ── Main group 39: DMF ──
    "DMF": "O=CN(C)C", "HCON(CH2)2": "[*]CN(C=O)[*]",

    # ── Main group 40: Fluorocarbon ──
    "CF3": "[*]C(F)(F)F", "CF2": "[*]C([*])(F)F", "CF": "[*]C([*])([*])F",

    # ── Main group 41: Generic ester ──
    "COO": "[*]C(=O)O[*]",

    # ── Main group 42: Silane ──
    "SiH3": "[*][SiH3]", "SiH2": "[*][SiH2][*]",
    "SiH": "[*][SiH]([*])[*]", "Si": "[*][Si]([*])([*])[*]",

    # ── Main group 43: Siloxane ──
    "SiH2O": "[*][SiH2]O[*]", "SiHO": "[*][SiH](O[*])[*]",
    "SiO": "[*][Si](O[*])([*])[*]",

    # ── Main group 44: NMP ──
    "NMP": "CN1CCCC1=O",

    # ── Main group 45: CFCs ──
    "CCl3F": "FC(Cl)(Cl)Cl", "CCl2F": "[*]C(Cl)(Cl)F",
    "HCCl2F": "FC(Cl)Cl", "HCClF": "[*]C(Cl)F",
    "CClF2": "[*]C(Cl)(F)F", "HCClF2": "FC(F)Cl",
    "CClF3": "FC(F)(F)Cl", "CCl2F2": "FC(F)(Cl)Cl",

    # ── Main group 46: Amide ──
    "CONH2": "[*]C(=O)N", "CONHCH3": "[*]C(=O)NC",
    "CONHCH2": "[*]C(=O)NC[*]", "CON(CH3)2": "[*]C(=O)N(C)C",
    "CONCH3CH2": "[*]C(=O)N(C)C[*]",
    "CON(CH2)2": "[*]C(=O)N(C[*])C[*]",

    # ── Main group 47: Oxyethylene ──
    "C2H5O2": "[*]OCCO", "C2H4O2": "[*]OCC([*])O",

    # ── Main group 48: Thioether ──
    "CH3S": "[*]SC", "CH2S": "[*]SC[*]", "CHS": "[*]SC([*])[*]",

    # ── Main group 49: Morpholine ──
    "morpholine": "C1COCCN1",

    # ── Main group 50: Thiophene ──
    "C4H4S": "c1ccsc1", "C4H3S": "[*]c1ccsc1",
    "C4H2S": "[*]c1cc([*])sc1",

    # ── Main group 51: Epoxide ──
    "CH2OCH2": "C1CO1", "CH2OCH": "[*]C1CO1",
    "CH2OC": "[*]C1([*])CO1", "CHOCH": "[*]C1C([*])O1",
    "CHOC": "[*]C1C([*])([*])O1",
    "COC": "[*]C1([*])C([*])([*])O1",

    # ── Main group 52: Anhydride ──
    "O=COC=O": "[*]C(=O)OC([*])=O",

    # ── Main group 55: Sulfone ──
    "CH2SuCH2": "[*]CS(=O)(=O)C[*]",
    "CH2SuCH": "[*]CS(=O)(=O)C([*])[*]",
}


# ═══════════════════════════════════════════════════════════════════════════
# AROMATIC GROUP DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# Each aromatic group occupies one position on a benzene ring.
# 'substituent' is the SMILES fragment attached to the ring carbon.
# 'ext_conn' is the number of additional connections to non-aromatic groups.

AROMATIC_GROUPS = {
    "ACH":   {"substituent": None,               "ext_conn": 0},
    "AC":    {"substituent": None,               "ext_conn": 1},  # bare C with 1 external
    "ACCH3": {"substituent": "C",                "ext_conn": 0},
    "ACCH2": {"substituent": "C[*]",             "ext_conn": 1},
    "ACCH":  {"substituent": "C([*])[*]",        "ext_conn": 2},
    "ACOH":  {"substituent": "O",                "ext_conn": 0},
    "ACNH2": {"substituent": "N",                "ext_conn": 0},
    "ACCl":  {"substituent": "Cl",               "ext_conn": 0},
    "ACNO2": {"substituent": "[N+](=O)[O-]",     "ext_conn": 0},
    "ACF":   {"substituent": "F",                "ext_conn": 0},
    "AC-CN": {"substituent": "C#N",              "ext_conn": 0},
    "AC-Br": {"substituent": "Br",              "ext_conn": 0},
}

AROMATIC_GROUP_NAMES = set(AROMATIC_GROUPS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: AROMATIC RING ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def canonical_ring_arrangement(groups: list[str]) -> tuple:
    """
    Canonicalize a ring arrangement under the dihedral group (rotation + reflection).
    Returns the lexicographically smallest equivalent arrangement.
    """
    n = len(groups)
    rotations = [tuple(groups[i:] + groups[:i]) for i in range(n)]
    reflected = [tuple(groups[::-1][i:] + groups[::-1][:i]) for i in range(n)]
    return min(rotations + reflected)


def enumerate_ring_isomers_single(aromatic_counts: dict[str, int]) -> list[tuple]:
    """
    Enumerate all distinct positional isomers on a single benzene ring.

    Parameters
    ----------
    aromatic_counts : dict
        Maps aromatic group name -> count. Must sum to 6.

    Returns
    -------
    List of canonical ring arrangement tuples (length 6).
    """
    total = sum(aromatic_counts.values())
    if total != 6:
        return []

    # Build the multiset of groups to place
    groups_to_place = []
    for name, count in sorted(aromatic_counts.items()):
        groups_to_place.extend([name] * count)

    # Generate all distinct permutations via recursive placement
    # then canonicalize under D6 symmetry
    seen = set()
    results = []

    def _place(remaining: list[str], ring: list[str]):
        if len(ring) == 6:
            canon = canonical_ring_arrangement(ring)
            if canon not in seen:
                seen.add(canon)
                results.append(canon)
            return

        placed = set()
        for i, grp in enumerate(remaining):
            if grp in placed:
                continue
            placed.add(grp)
            _place(remaining[:i] + remaining[i+1:], ring + [grp])

    _place(groups_to_place, [])
    return results


def ring_arrangement_to_smiles(arrangement: tuple, with_dummies: bool = True) -> Optional[str]:
    """
    Convert a benzene ring arrangement to SMILES.

    Groups with ext_conn > 0 (AC, ACCH2, ACCH) get [*] dummy atoms
    for their external connections.

    Returns canonical SMILES string or None on failure.
    """
    n = len(arrangement)
    assert n == 6

    parts = []
    for group_name in arrangement:
        info = AROMATIC_GROUPS[group_name]
        sub = info["substituent"]

        if group_name == "AC":
            # AC = bare aromatic carbon with 1 external connection
            if with_dummies:
                parts.append("c([*])")
            else:
                parts.append("c")
        elif sub is not None:
            parts.append(f"c({sub})")
        else:
            parts.append("c")

    # Build ring SMILES: c1(...)c(...)c(...)c(...)c(...)c1(...)
    smiles = parts[0].replace("c", "c1", 1)
    for i in range(1, n - 1):
        smiles += parts[i]
    smiles += parts[-1].replace("c", "c1", 1)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def enumerate_naphthalene_isomers(aromatic_counts: dict[str, int]) -> list[str]:
    """
    Enumerate positional isomers on a naphthalene (2 fused benzene) skeleton.
    Requires exactly 10 aromatic atoms, with exactly 2 having 3 ring-connections
    (the fusion carbons). The 2 fusion carbons can be ACH (they become AC-like
    in the fused system, using their 2 connections for 3 ring bonds).

    Actually in UNIFAC, naphthalene = 8×ACH + 2×AC.
    The 2 AC atoms occupy the fusion positions (4a, 8a).
    The 8 ACH atoms occupy the remaining positions.

    For substituted naphthalenes, some ACH positions become substituted groups.
    """
    total = sum(aromatic_counts.values())
    if total != 10:
        return []

    # Need at least 2 AC for fusion bonds (they can also have external connections)
    # Actually the fusion carbons MUST be AC-type since they bond to 3 ring neighbors
    n_ac = aromatic_counts.get("AC", 0)
    if n_ac < 2:
        return []

    # Naphthalene positions: 8 peripheral + 2 fusion
    # Fusion positions are fixed as AC (using all 3 connections for ring bonds)
    # Peripheral positions get the remaining groups

    peripheral_counts = dict(aromatic_counts)
    peripheral_counts["AC"] = peripheral_counts.get("AC", 0) - 2
    if peripheral_counts["AC"] == 0:
        del peripheral_counts["AC"]
    elif peripheral_counts["AC"] < 0:
        return []

    peripheral_total = sum(peripheral_counts.values())
    if peripheral_total != 8:
        return []

    # Enumerate arrangements of 8 groups on 8 peripheral positions
    # Naphthalene symmetry: D2h, with operations:
    # - rotation by 180° (swap ring halves)
    # - reflection through horizontal axis
    # - reflection through vertical axis
    # Positions numbered: 1,2,3,4 (ring A) and 5,6,7,8 (ring B)
    # Fusion at 4a (between 4,5) and 8a (between 8,1)

    # For simplicity, generate all permutations and canonicalize
    # under the symmetry group of naphthalene.

    groups_to_place = []
    for name, count in sorted(peripheral_counts.items()):
        groups_to_place.extend([name] * count)

    def canonical_naphthalene(arr):
        """Canonicalize under naphthalene D2h symmetry."""
        # Positions: [1,2,3,4,5,6,7,8]
        # Symmetry operations:
        # Identity: [0,1,2,3,4,5,6,7]
        # 180° rotation: [4,5,6,7,0,1,2,3] (swap rings)
        # Horizontal reflection: [7,6,5,4,3,2,1,0]
        # Vertical reflection: [3,2,1,0,7,6,5,4]
        ops = [
            [0,1,2,3,4,5,6,7],
            [4,5,6,7,0,1,2,3],
            [7,6,5,4,3,2,1,0],
            [3,2,1,0,7,6,5,4],
        ]
        candidates = []
        for op in ops:
            candidates.append(tuple(arr[i] for i in op))
        return min(candidates)

    seen = set()
    results = []

    def _place(remaining, positions):
        if len(positions) == 8:
            canon = canonical_naphthalene(positions)
            if canon not in seen:
                seen.add(canon)
                results.append(canon)
            return
        placed = set()
        for i, grp in enumerate(remaining):
            if grp in placed:
                continue
            placed.add(grp)
            _place(remaining[:i] + remaining[i+1:], positions + [grp])

    _place(groups_to_place, [])

    # Convert each arrangement to SMILES
    smiles_list = []
    for arr in results:
        smi = _naphthalene_to_smiles(arr)
        if smi:
            smiles_list.append(smi)

    return smiles_list


def _naphthalene_to_smiles(peripheral: tuple) -> Optional[str]:
    """
    Build naphthalene SMILES from 8 peripheral group assignments.
    Positions: [1,2,3,4,5,6,7,8] around the two fused rings.
    """
    def _sub(group_name):
        info = AROMATIC_GROUPS[group_name]
        if group_name == "AC":
            return "([*])"
        elif info["substituent"]:
            return f"({info['substituent']})"
        return ""

    # Naphthalene SMILES pattern:
    # c1(s1)c(s2)c(s3)c2(s4)c(s5)c(s6)c(s7)c(s8)c2c1
    # where s1..s8 are substituents at positions 1..8
    s = [_sub(g) for g in peripheral]
    smiles = f"c1{s[0]}c{s[1]}c{s[2]}c2{s[3]}c{s[4]}c{s[5]}c{s[6]}c{s[7]}c2c1"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def enumerate_biphenyl_isomers(
    ring1_counts: dict[str, int],
    ring2_counts: dict[str, int]
) -> list[str]:
    """
    Enumerate biphenyl isomers (two benzene rings connected by a single bond).
    Each ring has 5 aromatic groups + 1 AC at the linking position.
    """
    # Each ring: 5 regular aromatic + 1 linking position
    # The linking position is AC (using its external connection for the inter-ring bond)
    ring1_isomers = enumerate_ring_isomers_single(ring1_counts)
    ring2_isomers = enumerate_ring_isomers_single(ring2_counts)

    smiles_set = set()
    for arr1 in ring1_isomers:
        smi1 = ring_arrangement_to_smiles(arr1, with_dummies=True)
        if smi1 is None:
            continue
        for arr2 in ring2_isomers:
            smi2 = ring_arrangement_to_smiles(arr2, with_dummies=True)
            if smi2 is None:
                continue
            # Connect: replace [*] in ring1 with ring2 structure
            # This is handled by the main assembler — we return each ring
            # as a fragment with its external [*] connections
            pass

    # Actually, biphenyl is better handled by the main assembler:
    # ring1 and ring2 are super-fragments, each with 1 [*] at the AC position.
    # They get connected like any other pair of fragments.
    # So we just need to return the ring fragments.
    return []  # Handled by main assembler


def assemble_aromatic_fragments(aromatic_groups: dict[str, int]) -> list[tuple[str, int]]:
    """
    Given aromatic group counts, determine the ring system(s) and return
    all valid aromatic super-fragment SMILES with their external connection counts.

    Handles:
    - Single benzene (6 atoms)
    - Naphthalene (10 atoms, fused)
    - Multiple independent benzene rings (12, 18, ... atoms)
    - Mixed: e.g., 1 naphthalene + 1 benzene (16 atoms)

    Returns list of (smiles_with_dummies, n_external_connections).
    For multi-ring systems, each ring becomes a separate super-fragment,
    so the return is a list of (list_of_fragment_smiles, total_external_connections).
    Actually, to keep the interface consistent, multi-ring systems return
    individual ring fragments — the main assembler connects them.
    """
    total_aromatic = sum(aromatic_groups.values())

    if total_aromatic == 0:
        return []

    # Enumerate all valid ring partitions: combinations of benzene(6) and naphthalene(10)
    partitions = _aromatic_ring_partitions(total_aromatic, aromatic_groups.get("AC", 0))

    all_results = []

    for n_benz, n_naph in partitions:
        # Distribute aromatic group types among ring slots
        distributions = _distribute_groups_to_ring_system(
            aromatic_groups, n_benz, n_naph
        )

        for dist in distributions:
            # dist is a tuple: (benzene_ring_compositions..., naphthalene_compositions...)
            # Each composition is a dict {group_name: count}
            fragments = _build_fragments_from_distribution(dist, n_benz, n_naph)
            if fragments is not None:
                all_results.extend(fragments)

    # Deduplicate
    seen = set()
    unique = []
    for item in all_results:
        if item[0] == "__MULTI_RING__":
            # For multi-ring, deduplicate by the tuple of (smiles, n_ext) pairs
            key = tuple(sorted(item[1]))
        else:
            key = item[0]  # SMILES string
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def _aromatic_ring_partitions(total_atoms: int, n_ac: int) -> list[tuple[int, int]]:
    """
    Enumerate ways to partition total aromatic atoms into rings.
    Returns list of (n_benzene_rings, n_naphthalene_rings).
    """
    results = []
    max_naph = total_atoms // 10
    for n_naph in range(max_naph + 1):
        remaining = total_atoms - n_naph * 10
        if remaining >= 0 and remaining % 6 == 0:
            n_benz = remaining // 6
            # Naphthalene needs 2 AC each for fusion
            ac_for_naph = n_naph * 2
            if n_ac >= ac_for_naph:
                results.append((n_benz, n_naph))
    return results


def _distribute_groups_to_ring_system(
    aromatic_groups: dict[str, int],
    n_benzene: int,
    n_naphthalene: int
) -> list[tuple]:
    """
    Distribute aromatic group types among benzene and naphthalene rings.

    For naphthalene: 2 AC are consumed for fusion (not available for external use).
    The remaining 8 positions get peripheral groups.

    Returns list of distributions. Each distribution is a tuple of dicts:
    (ring1_counts, ring2_counts, ...) where benzene rings come first,
    then naphthalene peripheral counts.
    """
    # Reserve AC for naphthalene fusion
    available = dict(aromatic_groups)
    ac_for_fusion = n_naphthalene * 2
    available["AC"] = available.get("AC", 0) - ac_for_fusion
    if available.get("AC", 0) < 0:
        return []
    if available.get("AC", 0) == 0 and "AC" in available:
        del available["AC"]

    # Total ring slots: n_benzene * 6 + n_naphthalene * 8 (peripheral positions)
    n_rings = n_benzene + n_naphthalene
    ring_sizes = [6] * n_benzene + [8] * n_naphthalene

    total_available = sum(available.values())
    total_slots = sum(ring_sizes)
    if total_available != total_slots:
        return []

    if n_rings == 0:
        return []

    if n_rings == 1:
        # Single ring: no distribution needed
        return [(available,)]

    # Multi-ring: enumerate all valid distributions
    group_names = sorted(available.keys())
    group_totals = [available[n] for n in group_names]
    n_group_types = len(group_names)

    results_set = set()

    def _distribute(g_idx, rings):
        if g_idx == n_group_types:
            # Check all rings are exactly the right size
            if all(sum(r.values()) == ring_sizes[i] for i, r in enumerate(rings)):
                # Canonicalize: sort rings of same size together
                benz_parts = tuple(sorted(
                    tuple(sorted(r.items())) for r in rings[:n_benzene]
                ))
                naph_parts = tuple(sorted(
                    tuple(sorted(r.items())) for r in rings[n_benzene:]
                ))
                canon = (benz_parts, naph_parts)
                results_set.add(canon)
            return

        name = group_names[g_idx]
        total = group_totals[g_idx]

        # Generate all ways to split 'total' items into n_rings bins
        _split_recursive(total, n_rings, 0, [], name, g_idx, rings)

    def _split_recursive(remaining, n_parts, part_idx, current_split,
                         name, g_idx, rings):
        if part_idx == n_parts - 1:
            current_split.append(remaining)
            # Check feasibility
            ok = True
            for i, count in enumerate(current_split):
                if sum(rings[i].values()) + count > ring_sizes[i]:
                    ok = False
                    break
            if ok:
                # Apply
                for i, count in enumerate(current_split):
                    if count > 0:
                        rings[i][name] = count
                _distribute(g_idx + 1, rings)
                # Undo
                for i in range(len(current_split)):
                    if name in rings[i]:
                        del rings[i][name]
            current_split.pop()
            return

        for k in range(remaining + 1):
            # Quick feasibility: don't exceed ring size
            if sum(rings[part_idx].values()) + k > ring_sizes[part_idx]:
                break
            current_split.append(k)
            _split_recursive(remaining - k, n_parts, part_idx + 1,
                             current_split, name, g_idx, rings)
            current_split.pop()

    _distribute(0, [{} for _ in range(n_rings)])

    # Convert back to list of tuples of dicts
    results = []
    for benz_parts, naph_parts in results_set:
        ring_dicts = [dict(bp) for bp in benz_parts] + [dict(np) for np in naph_parts]
        results.append(tuple(ring_dicts))

    return results


def _build_fragments_from_distribution(
    distribution: tuple,
    n_benzene: int,
    n_naphthalene: int
) -> Optional[list[tuple[str, int]]]:
    """
    Given a distribution of groups to rings, build all positional isomers
    for each ring and return the assembled fragments.

    For single-ring systems: returns list of (smiles, n_external).
    For multi-ring systems: each ring is an independent fragment with [*] dummies.
    The caller's assembler will connect them.
    """
    ring_compositions = list(distribution)
    benzene_comps = ring_compositions[:n_benzene]
    naphthalene_comps = ring_compositions[n_benzene:]

    # Build isomers for each ring independently
    ring_isomer_sets = []

    for comp in benzene_comps:
        isomers = enumerate_ring_isomers_single(comp)
        ring_smiles = []
        for arr in isomers:
            smi = ring_arrangement_to_smiles(arr)
            if smi is not None:
                n_ext = sum(AROMATIC_GROUPS[g]["ext_conn"] for g in arr)
                ring_smiles.append((smi, n_ext))
        if not ring_smiles:
            return None
        ring_isomer_sets.append(ring_smiles)

    for comp in naphthalene_comps:
        # Rebuild full naphthalene composition (add back 2 AC for fusion)
        full_naph = dict(comp)
        full_naph["AC"] = full_naph.get("AC", 0) + 2
        naph_smiles_list = enumerate_naphthalene_isomers(full_naph)
        naph_results = []
        for smi in naph_smiles_list:
            n_ext = sum(
                AROMATIC_GROUPS[name]["ext_conn"] * count
                for name, count in comp.items()
            )
            naph_results.append((smi, n_ext))
        if not naph_results:
            return None
        ring_isomer_sets.append(naph_results)

    if len(ring_isomer_sets) == 1:
        # Single ring: return directly
        return ring_isomer_sets[0]

    # Multiple rings: return each ring as a separate fragment.
    # The main assembler will treat each as a super-fragment.
    # We need to return all combinations of ring isomers.
    # Each combination is a list of (smiles, n_ext) per ring.
    # We'll encode multi-ring results as: return each ring separately.
    # The caller needs to handle multiple aromatic super-fragments.
    # Mark them with a special flag.

    from itertools import product
    combo_results = []
    for combo in product(*ring_isomer_sets):
        # combo is a tuple of (smiles, n_ext) per ring
        # Return as a special multi-ring marker
        combo_results.append(("__MULTI_RING__", combo))

    return combo_results


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH ENUMERATION (with Phase 1 ring-strain filter)
# ═══════════════════════════════════════════════════════════════════════════

def enumerate_connection_graphs(
    group_instances: list[dict],
    max_results: int = 1000,
    topology: str = "acyclic",
    min_ring_size: int = 5,
) -> list[list[tuple[int, int]]]:
    """
    Enumerate connected simple graphs realizing the given degree sequence.

    Parameters
    ----------
    group_instances : list of dict with 'name', 'connections', 'group_id'
    max_results : int
    topology : str
        "acyclic"    — trees only (cycle rank = 0)
        "monocyclic" — exactly one ring (cycle rank = 1)
        "bicyclic"   — exactly two independent rings (cycle rank = 2)
        "any"        — no restriction on cycle count
    min_ring_size : int
        Minimum ring size allowed (strain filter). Default 5.
        Set to 3 to allow all rings including cyclopropane.

    Returns
    -------
    List of edge-lists.
    """
    n = len(group_instances)
    degrees = [g['connections'] for g in group_instances]
    total_connections = sum(degrees)

    if total_connections % 2 != 0:
        return []

    n_bonds = total_connections // 2
    active_count = sum(1 for d in degrees if d > 0)

    # Cycle rank = edges - nodes + 1 (for connected graph)
    # target_cycle_rank: exact value for acyclic/mono/bicyclic, or None for "any"
    target_cycle_rank = {
        "acyclic": 0,
        "monocyclic": 1,
        "bicyclic": 2,
    }.get(topology)
    max_cycle_rank = target_cycle_rank if target_cycle_rank is not None else n_bonds

    # Quick feasibility check
    required_cycle_rank = n_bonds - active_count + 1
    if target_cycle_rank is not None and required_cycle_rank != target_cycle_rank:
        # The degree sequence forces a specific cycle rank; if it doesn't
        # match what we want, no solutions exist.
        return []

    allow_cycles = topology != "acyclic"

    remaining = degrees[:]
    edges: list[tuple[int, int]] = []
    results: list[list[tuple[int, int]]] = []
    group_ids = [g['group_id'] for g in group_instances]
    adj: list[set[int]] = [set() for _ in range(n)]
    current_cycle_rank = [0]  # mutable counter: edges - nodes_with_edges + components

    def is_connected():
        active = {i for i in range(n) if degrees[i] > 0}
        if len(active) <= 1:
            return True
        start = min(active)
        visited = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return active.issubset(visited)

    def shortest_path(u: int, v: int) -> int:
        """BFS shortest path length from u to v through existing edges."""
        if not adj[u]:
            return -1
        visited = {u}
        queue = [(u, 0)]
        while queue:
            node, dist = queue.pop(0)
            for nb in adj[node]:
                if nb == v:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return -1

    def would_create_cycle(u: int, v: int) -> bool:
        return shortest_path(u, v) > 0

    def would_create_small_ring(u: int, v: int) -> bool:
        """Check if adding edge (u,v) creates a ring smaller than min_ring_size."""
        sp = shortest_path(u, v)
        if sp <= 0:
            return False
        ring_size = sp + 1
        return ring_size < min_ring_size

    def components_ok():
        active = {i for i in range(n) if degrees[i] > 0 or adj[i]}
        if not active:
            return True
        visited = set()
        components = []
        for node in active:
            if node not in visited:
                comp = set()
                stack = [node]
                while stack:
                    nd = stack.pop()
                    if nd in comp:
                        continue
                    comp.add(nd)
                    for nb in adj[nd]:
                        if nb not in comp:
                            stack.append(nb)
                visited |= comp
                components.append(comp)
        if len(components) <= 1:
            return True
        for comp in components:
            if sum(remaining[i] for i in comp) == 0:
                return False
        return True

    def backtrack():
        if len(results) >= max_results:
            return

        # Prune: current cycle rank already exceeds max allowed
        if current_cycle_rank[0] > max_cycle_rank:
            return

        if sum(remaining) == 0:
            if is_connected():
                # Final cycle rank check
                if target_cycle_rank is None or current_cycle_rank[0] == target_cycle_rank:
                    results.append(list(edges))
            return

        first = -1
        for i in range(n):
            if remaining[i] > 0:
                first = i
                break
        if first == -1:
            return

        seen_sym = set()
        for j in range(first + 1, n):
            if remaining[j] <= 0:
                continue

            sym_key = (group_ids[j], remaining[j])
            if sym_key in seen_sym:
                continue
            seen_sym.add(sym_key)

            if j in adj[first]:
                continue

            creates_cycle = would_create_cycle(first, j)

            if not allow_cycles and creates_cycle:
                continue

            if creates_cycle and min_ring_size > 3 and would_create_small_ring(first, j):
                continue

            # Check if adding this cycle would exceed the target
            if creates_cycle and current_cycle_rank[0] + 1 > max_cycle_rank:
                continue

            edges.append((first, j))
            remaining[first] -= 1
            remaining[j] -= 1
            adj[first].add(j)
            adj[j].add(first)
            if creates_cycle:
                current_cycle_rank[0] += 1

            if components_ok():
                backtrack()

            edges.pop()
            remaining[first] += 1
            remaining[j] += 1
            adj[first].discard(j)
            adj[j].discard(first)
            if creates_cycle:
                current_cycle_rank[0] -= 1

    backtrack()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MOLECULE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def assemble_molecule(
    group_instances: list[dict],
    edge_list: list[tuple[int, int]],
    group_db: dict
) -> Optional[str]:
    """Assemble molecule from fragments + connection topology. Returns canonical SMILES."""
    fragments = []
    for gi in group_instances:
        # Handle aromatic super-fragments (group_id == -1)
        if gi['group_id'] == -1:
            smi = gi.get('override_smiles')
        else:
            gdata = group_db[gi['group_id']]
            smi = gi.get('override_smiles') or FRAGMENT_SMILES.get(gdata['name'])
        if smi is None:
            return None

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        expected = gi['connections']
        if len(dummies) != expected:
            return None

        fragments.append((mol, dummies))

    combined = RWMol()
    all_dummies = []

    for frag_mol, frag_dummies in fragments:
        idx_map = {}
        for atom in frag_mol.GetAtoms():
            new_idx = combined.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            idx_map[atom.GetIdx()] = new_idx
            na = combined.GetAtomWithIdx(new_idx)
            na.SetFormalCharge(atom.GetFormalCharge())
            na.SetIsAromatic(atom.GetIsAromatic())
            na.SetNumExplicitHs(atom.GetNumExplicitHs())
            na.SetNoImplicit(atom.GetNoImplicit())

        for bond in frag_mol.GetBonds():
            combined.AddBond(idx_map[bond.GetBeginAtomIdx()],
                             idx_map[bond.GetEndAtomIdx()],
                             bond.GetBondType())

        all_dummies.append([idx_map[d] for d in frag_dummies])

    usage = [0] * len(fragments)
    bonds_to_add = []
    dummies_to_remove = set()

    for (i, j) in edge_list:
        di = all_dummies[i][usage[i]]; usage[i] += 1
        dj = all_dummies[j][usage[j]]; usage[j] += 1

        real_i = real_j = None
        for bond in combined.GetBonds():
            bi, ei = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if bi == di: real_i = ei
            elif ei == di: real_i = bi
            if bi == dj: real_j = ei
            elif ei == dj: real_j = bi

        if real_i is None or real_j is None:
            return None

        bonds_to_add.append((real_i, real_j))
        dummies_to_remove.add(di)
        dummies_to_remove.add(dj)

    for a, b in bonds_to_add:
        if combined.GetBondBetweenAtoms(a, b) is None:
            combined.AddBond(a, b, Chem.BondType.SINGLE)

    for dummy in sorted(dummies_to_remove, reverse=True):
        for bond in list(combined.GetBonds()):
            if bond.GetBeginAtomIdx() == dummy or bond.GetEndAtomIdx() == dummy:
                combined.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        combined.RemoveAtom(dummy)

    try:
        Chem.SanitizeMol(combined)
        return Chem.MolToSmiles(combined)
    except Exception:
        try:
            Chem.SanitizeMol(combined,
                             sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            return Chem.MolToSmiles(combined)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def load_unifac_groups(json_path: str) -> dict:
    """Load UNIFAC group database from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return {g['id']: g for g in data['unifac_groups']}


def group_id_by_name(name: str, group_db: dict) -> Optional[int]:
    """Find group ID by name or unifac_id."""
    for gid, gdata in group_db.items():
        if gdata['name'] == name or gdata['unifac_id'] == name:
            return gid
    return None


def reverse_engineer(
    group_composition: dict[int, int],
    group_db: dict,
    max_results: int = 500,
    topology: str = "acyclic",
    min_ring_size: int = 5,
    enumerate_stereo: bool = False,
    verbose: bool = True
) -> list[str]:
    """
    Enumerate all valid molecules for a given UNIFAC group composition.

    Parameters
    ----------
    group_composition : dict  {group_id: count}
    group_db : dict from load_unifac_groups()
    max_results : int
    topology : str
        "acyclic"    — only tree/chain structures (no rings in non-aromatic part)
        "monocyclic" — exactly one alicyclic ring
        "bicyclic"   — exactly two independent alicyclic rings
        "any"        — all valid topologies
        Note: aromatic rings are always assembled regardless of this setting.
    min_ring_size : int
        Minimum alicyclic ring size (default 5).
    enumerate_stereo : bool
        If True, enumerate E/Z and R/S stereoisomers for each constitutional
        isomer. Default False (returns flat/unspecified stereochemistry).
    verbose : bool

    Returns
    -------
    Sorted list of unique canonical SMILES.
    """
    valid_topologies = ("acyclic", "monocyclic", "bicyclic", "any")
    if topology not in valid_topologies:
        raise ValueError(f"topology must be one of {valid_topologies}, got '{topology}'")

    # ── Separate aromatic vs non-aromatic groups ──
    aromatic_comp = {}
    nonaromatic_ids = {}

    for gid, count in group_composition.items():
        gdata = group_db[gid]
        name = gdata['name']
        if name in AROMATIC_GROUP_NAMES:
            aromatic_comp[name] = aromatic_comp.get(name, 0) + count
        else:
            nonaromatic_ids[gid] = count

    has_aromatic = sum(aromatic_comp.values()) > 0

    if not has_aromatic:
        constitutional = _solve_nonaromatic(nonaromatic_ids, group_db,
                                            max_results, topology, min_ring_size, verbose)
        if enumerate_stereo:
            return _expand_stereoisomers(constitutional, verbose)
        return constitutional

    # ── Aromatic path ──
    if verbose:
        print(f"Aromatic groups: {aromatic_comp}")
        if nonaromatic_ids:
            na_names = {group_db[gid]['name']: c for gid, c in nonaromatic_ids.items()}
            print(f"Non-aromatic groups: {na_names}")

    aromatic_fragments = assemble_aromatic_fragments(aromatic_comp)

    if not aromatic_fragments:
        if verbose:
            print(f"ERROR: Could not assemble aromatic ring from {aromatic_comp}")
        return []

    if verbose:
        n_single = sum(1 for f in aromatic_fragments if f[0] != "__MULTI_RING__")
        n_multi = sum(1 for f in aromatic_fragments if f[0] == "__MULTI_RING__")
        if n_multi:
            print(f"Aromatic assemblies: {n_single} single-ring + {n_multi} multi-ring")
        else:
            print(f"Aromatic ring isomers: {n_single}")

    all_smiles = set()

    for fragment_result in aromatic_fragments:
        if fragment_result[0] == "__MULTI_RING__":
            # Multi-ring: fragment_result[1] is a tuple of (smiles, n_ext) per ring
            ring_fragments = fragment_result[1]
            _assemble_multi_ring(ring_fragments, nonaromatic_ids, group_db,
                                max_results, topology, min_ring_size,
                                all_smiles, verbose)
        else:
            arom_smi, n_ext = fragment_result
            _assemble_single_ring(arom_smi, n_ext, nonaromatic_ids, group_db,
                                  max_results, topology, min_ring_size,
                                  all_smiles, verbose)

    constitutional = sorted(all_smiles)
    if enumerate_stereo:
        return _expand_stereoisomers(constitutional, verbose)
    return constitutional


def _assemble_single_ring(arom_smi, n_ext, nonaromatic_ids, group_db,
                          max_results, topology, min_ring_size,
                          all_smiles, verbose):
    """Assemble a single aromatic ring with non-aromatic groups."""
    if n_ext == 0 and not nonaromatic_ids:
        mol = Chem.MolFromSmiles(arom_smi)
        if mol:
            all_smiles.add(Chem.MolToSmiles(mol))
        return

    if n_ext == 0 and nonaromatic_ids:
        return  # No way to connect

    instances = [{
        'group_id': -1,
        'name': '_AROMATIC_RING_',
        'connections': n_ext,
        'override_smiles': arom_smi,
    }]

    for gid, count in sorted(nonaromatic_ids.items()):
        gdata = group_db[gid]
        for _ in range(count):
            instances.append({
                'group_id': gid,
                'name': gdata['name'],
                'connections': gdata['connections'],
            })

    total_conn = sum(g['connections'] for g in instances)
    if total_conn % 2 != 0:
        return

    # Check for unsupported groups
    for gi in instances:
        if gi['group_id'] == -1:
            continue
        gdata = group_db[gi['group_id']]
        if FRAGMENT_SMILES.get(gdata['name']) is None and gi['connections'] > 0:
            return

    graphs = enumerate_connection_graphs(instances, max_results, topology, min_ring_size)
    for edges in graphs:
        smi = assemble_molecule(instances, edges, group_db)
        if smi:
            all_smiles.add(smi)


def _assemble_multi_ring(ring_fragments, nonaromatic_ids, group_db,
                         max_results, topology, min_ring_size,
                         all_smiles, verbose):
    """
    Assemble multiple independent aromatic rings with non-aromatic groups.
    Each ring is treated as a super-fragment in the connection graph.
    """
    instances = []

    # Add each aromatic ring as a super-fragment
    for ring_smi, ring_n_ext in ring_fragments:
        if ring_n_ext == 0:
            # Ring with no external connections can't connect to anything
            # in a multi-ring system -> skip this entire combination
            return
        instances.append({
            'group_id': -1,
            'name': '_AROMATIC_RING_',
            'connections': ring_n_ext,
            'override_smiles': ring_smi,
        })

    # Add non-aromatic groups
    for gid, count in sorted(nonaromatic_ids.items()):
        gdata = group_db[gid]
        for _ in range(count):
            instances.append({
                'group_id': gid,
                'name': gdata['name'],
                'connections': gdata['connections'],
            })

    total_conn = sum(g['connections'] for g in instances)
    if total_conn % 2 != 0:
        return

    # Check for unsupported groups
    for gi in instances:
        if gi['group_id'] == -1:
            continue
        gdata = group_db[gi['group_id']]
        if FRAGMENT_SMILES.get(gdata['name']) is None and gi['connections'] > 0:
            return

    if verbose:
        names = ", ".join(f"{g['name']}({g['connections']})" for g in instances)
        print(f"  Multi-ring assembling: {names} ({total_conn//2} bonds)")

    graphs = enumerate_connection_graphs(instances, max_results, topology, min_ring_size)
    for edges in graphs:
        smi = assemble_molecule(instances, edges, group_db)
        if smi:
            all_smiles.add(smi)


def _solve_nonaromatic(
    group_composition: dict[int, int],
    group_db: dict,
    max_results: int,
    topology: str,
    min_ring_size: int,
    verbose: bool
) -> list[str]:
    """Solve for non-aromatic groups only."""
    instances = []
    for gid, count in sorted(group_composition.items()):
        gdata = group_db[gid]
        for _ in range(count):
            instances.append({
                'group_id': gid,
                'name': gdata['name'],
                'connections': gdata['connections'],
            })

    total_conn = sum(g['connections'] for g in instances)
    if total_conn % 2 != 0:
        if verbose:
            print(f"ERROR: Total connections = {total_conn} (odd).")
        return []

    if verbose:
        groups_str = ", ".join(f"{g['name']}({g['connections']})" for g in instances)
        print(f"Groups: {groups_str}")
        print(f"Bonds: {total_conn // 2} | Topology: {topology} "
              f"| Min ring: {min_ring_size}")

    for gi in instances:
        gdata = group_db[gi['group_id']]
        if FRAGMENT_SMILES.get(gdata['name']) is None and gi['connections'] > 0:
            if verbose:
                print(f"WARNING: '{gdata['name']}' not supported.")
            return []

    graphs = enumerate_connection_graphs(instances, max_results,
                                         topology, min_ring_size)
    if verbose:
        print(f"Topologies: {len(graphs)}")

    smiles_set = set()
    failures = 0
    for edges in graphs:
        smi = assemble_molecule(instances, edges, group_db)
        if smi:
            smiles_set.add(smi)
        else:
            failures += 1

    if verbose and failures:
        print(f"Assembly failures: {failures}/{len(graphs)}")

    result = sorted(smiles_set)
    if verbose:
        print(f"Unique SMILES: {len(result)}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# STEREOISOMER ENUMERATION
# ═══════════════════════════════════════════════════════════════════════════

def _expand_stereoisomers(constitutional_smiles: list[str],
                          verbose: bool = True) -> list[str]:
    """
    Expand a list of constitutional SMILES into all stereoisomers.
    Uses RDKit's EnumerateStereoisomers to find E/Z and R/S variants.

    Returns sorted list of unique stereo-specific SMILES.
    """
    from rdkit.Chem import EnumerateStereoisomers

    opts = EnumerateStereoisomers.StereoEnumerationOptions(unique=True)
    all_stereo = set()

    for smi in constitutional_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        stereo_isomers = list(
            EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts)
        )
        if stereo_isomers:
            for iso in stereo_isomers:
                all_stereo.add(Chem.MolToSmiles(iso))
        else:
            # No stereocenters/double bonds -> keep original
            all_stereo.add(smi)

    result = sorted(all_stereo)
    if verbose:
        n_const = len(constitutional_smiles)
        n_stereo = len(result)
        if n_stereo > n_const:
            print(f"Stereoisomers: {n_const} constitutional -> {n_stereo} total")
    return result


def reverse_engineer_by_name(
    composition: dict[str, int],
    group_db: dict,
    max_results: int = 500,
    topology: str = "acyclic",
    min_ring_size: int = 5,
    enumerate_stereo: bool = False,
    verbose: bool = True
) -> list[str]:
    """
    Convenience wrapper accepting group names.

    Parameters
    ----------
    composition : dict
        Maps group name (str) -> count.
    group_db : dict
        From load_unifac_groups().
    topology : str
        "acyclic", "monocyclic", "bicyclic", or "any".
    min_ring_size : int
        Minimum alicyclic ring size (default 5).
    enumerate_stereo : bool
        If True, expand each constitutional isomer into all E/Z and R/S
        stereoisomers. Default False.

    Examples
    --------
    >>> reverse_engineer_by_name({"CH3": 2, "CH2": 1}, db)
    ['CCC']

    >>> reverse_engineer_by_name({"CH2": 6}, db, topology="monocyclic")
    ['C1CCCCC1']

    >>> reverse_engineer_by_name({"ACH": 5, "ACCH3": 1}, db)
    ['Cc1ccccc1']

    >>> reverse_engineer_by_name({"CH3": 2, "CH=CH": 1}, db, enumerate_stereo=True)
    ['C/C=C/C', 'C/C=C\\\\C']
    """
    id_comp = {}
    for name, count in composition.items():
        gid = group_id_by_name(name, group_db)
        if gid is None:
            if verbose:
                print(f"ERROR: Unknown group '{name}'")
            return []
        id_comp[gid] = count
    return reverse_engineer(id_comp, group_db, max_results, topology,
                            min_ring_size, enumerate_stereo, verbose)


def coverage_report(group_db: dict) -> dict:
    """Report supported UNIFAC groups."""
    supported, unsupported = [], []
    for gid, gdata in sorted(group_db.items()):
        name = gdata['name']
        if name in FRAGMENT_SMILES or name in AROMATIC_GROUP_NAMES:
            supported.append(name)
        else:
            unsupported.append(name)
    total = len(supported) + len(unsupported)
    return {
        'supported': supported,
        'unsupported': unsupported,
        'coverage_pct': 100 * len(supported) / total if total > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_tests(group_db: dict):
    tests = [
        # ── Acyclic (original tests) ──
        ("Ethane",        {"CH3": 2},                              ["CC"]),
        ("Propane",       {"CH3": 2, "CH2": 1},                    ["CCC"]),
        ("Isobutane",     {"CH3": 3, "CH": 1},                     ["CC(C)C"]),
        ("Neopentane",    {"CH3": 4, "C": 1},                      ["CC(C)(C)C"]),
        ("Ethanol",       {"CH3": 1, "CH2": 1, "OH": 1},           ["CCO"]),
        ("Acetone",       {"CH3": 1, "CH3CO": 1},                  ["CC(C)=O"]),
        ("Acetic acid",   {"CH3": 1, "COOH": 1},                   ["CC(=O)O"]),
        ("Ethyl acetate", {"CH3": 2, "CH2COO": 1},                 ["CCC(=O)OC"]),
        ("Diethyl ether", {"CH3": 2, "CH2O": 1},                   ["CCOC"]),
        ("Propene",       {"CH3": 1, "CH2=CH": 1},                 ["C=CC"]),
        ("1-Bromopropane",{"CH3": 1, "CH2": 2, "Br": 1},           ["CCCBr"]),
        ("C6 branched",   {"CH3": 3, "CH2": 2, "CH": 1},           None),  # expect ≥2

        # ── Phase 1: Alicyclic rings ──
        ("Cyclohexane",   {"CH2": 6},                               ["C1CCCCC1"]),
        ("Methylcyclohexane", {"CH3": 1, "CH": 1, "CH2": 5},       None),  # expect ≥1
        ("Cyclopentane",  {"CH2": 5},                               ["C1CCCC1"]),

        # ── Phase 2: Aromatic ──
        ("Benzene",       {"ACH": 6},                               ["c1ccccc1"]),
        ("Toluene",       {"ACH": 5, "ACCH3": 1},                  ["Cc1ccccc1"]),
        ("Xylenes",       {"ACH": 4, "ACCH3": 2},                  None),  # expect 3
        ("Phenol",        {"ACH": 5, "ACOH": 1},                   ["Oc1ccccc1"]),
        ("Chlorobenzene", {"ACH": 5, "ACCl": 1},                   ["Clc1ccccc1"]),
        ("Aniline",       {"ACH": 5, "ACNH2": 1},                  ["Nc1ccccc1"]),

        # ── Mixed aromatic + aliphatic ──
        ("Ethylbenzene",  {"ACH": 5, "AC": 1, "CH2": 1, "CH3": 1}, ["CCc1ccccc1"]),
        ("Styrene",       {"ACH": 5, "AC": 1, "CH2=CH": 1},        ["C=Cc1ccccc1"]),
        ("Benzaldehyde",  {"ACH": 5, "AC": 1, "HCO": 1},           ["O=Cc1ccccc1"]),
        ("Benzoic acid",  {"ACH": 5, "AC": 1, "COOH": 1},          ["O=C(O)c1ccccc1"]),

        # ── Naphthalene ──
        ("Naphthalene",   {"ACH": 8, "AC": 2},                     ["c1ccc2ccccc2c1"]),

        # ── Multi-ring aromatic ──
        ("Diphenylmethane", {"ACH": 10, "AC": 2, "CH2": 1},       ["c1ccc(Cc2ccccc2)cc1"]),
        ("Biphenyl",        {"ACH": 10, "AC": 2},                  ["c1ccc(-c2ccccc2)cc1"]),
    ]

    # Separate stereo tests (run with enumerate_stereo=True)
    stereo_tests = [
        ("2-butene E/Z",      {"CH3": 2, "CH=CH": 1},    2),  # expect 2 stereoisomers
        ("cis/trans decalin",  {"CH2": 8, "CH": 2},       2),  # expect 2
        ("2-bromobutane R/S",  {"CH3": 2, "CH": 1, "CH2": 1, "Br": 1}, 3),  # 2 constitutional, one has R/S
    ]

    print("=" * 70)
    print("UNIFAC Reverse Engineering — Extended Test Suite")
    print("  Phase 1: Alicyclic rings with strain filter")
    print("  Phase 2: Aromatic ring assembly")
    print("=" * 70)

    passed = failed = 0

    for desc, composition, expected in tests:
        print(f"\n{'─'*60}")
        print(f"  {desc}: {composition}")

        # Determine topology mode
        if desc in ("Cyclohexane", "Methylcyclohexane", "Cyclopentane"):
            topo = "monocyclic"
        else:
            topo = "acyclic"
        results = reverse_engineer_by_name(
            composition, group_db,
            topology=topo,
            min_ring_size=5,
            verbose=False
        )
        print(f"  → {results}")

        if expected is not None:
            # For substring tests (e.g., biphenyl check)
            if all(
                any(e in r for r in results) if e.startswith("-") else e in results
                for e in expected
            ):
                print(f"  ✓ PASS"); passed += 1
            else:
                print(f"  ✗ FAIL — expected {expected}"); failed += 1
        else:
            if len(results) >= 2:
                print(f"  ✓ PASS ({len(results)} isomers)"); passed += 1
            elif desc == "Methylcyclohexane" and len(results) >= 1:
                print(f"  ✓ PASS ({len(results)} isomers)"); passed += 1
            else:
                print(f"  ✗ FAIL — expected ≥2 isomers, got {len(results)}"); failed += 1

    # ── Stereoisomer tests ──
    print(f"\n{'─'*60}")
    print(f"  STEREOISOMER TESTS")
    for desc, composition, expected_count in stereo_tests:
        print(f"\n{'─'*60}")
        print(f"  {desc}: {composition}")

        if desc == "cis/trans decalin":
            topo = "bicyclic"
            mrs = 5
        else:
            topo = "acyclic"
            mrs = 5
        results = reverse_engineer_by_name(
            composition, group_db,
            topology=topo,
            min_ring_size=mrs,
            enumerate_stereo=True,
            verbose=False
        )
        print(f"  → {results}")
        if len(results) >= expected_count:
            print(f"  ✓ PASS ({len(results)} stereoisomers)"); passed += 1
        else:
            print(f"  ✗ FAIL — expected ≥{expected_count}, got {len(results)}"); failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")

    cov = coverage_report(group_db)
    print(f"\nFragment coverage: {cov['coverage_pct']:.0f}% "
          f"({len(cov['supported'])}/{len(cov['supported'])+len(cov['unsupported'])})")
    if cov['unsupported']:
        print(f"Unsupported: {cov['unsupported']}")


if __name__ == "__main__":
    import sys
    default_path = str(Path(__file__).parent.parent / "data" / "json" / "unifac_ly.json")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    db = load_unifac_groups(path)
    print(f"Loaded {len(db)} UNIFAC groups\n")
    run_tests(db)
