import json
import copy
from rdkit import Chem
from rdkit.Chem import rdmolops

# ============================================================
# Load SMARTS database
# ============================================================

SMARTS_JSON_PATH = "data/json/smarts.json"

with open(SMARTS_JSON_PATH) as f:
    SMARTS_DB = json.load(f)["data"]

print(SMARTS_DB)
def mols_from_smarts_entry(entry):
    """
    A SMARTS entry may be a single SMARTS or a list of SMARTS.
    We return a list of RDKit Mol objects.
    """
    if isinstance(entry, list):
        return [Chem.MolFromSmarts(s) for s in entry]
    return [Chem.MolFromSmarts(entry)]


# ============================================================
# Attachment atom detection
# ============================================================

def attachment_atoms(mol):
    """
    Return atom indices that can accept at least one additional
    single bond based on valence.
    """
    attachable = []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            continue

        try:
            max_v = atom.GetImplicitValence() + atom.GetExplicitValence()
        except Exception:
            continue

        if atom.GetTotalDegree() < max_v:
            attachable.append(atom.GetIdx())

    return attachable


# ============================================================
# Molecule attachment with ring constraints
# ============================================================

def attach_mols(molA, idxA, molB, idxB,
                allow_cycles=True,
                allow_bicycles=True):
    """
    Combine molA and molB with a single bond between idxA and idxB.
    Enforce ring and bicyclic constraints.
    """

    combo = Chem.CombineMols(molA, molB)
    em = Chem.EditableMol(combo)
    offset = molA.GetNumAtoms()

    em.AddBond(idxA, idxB + offset, Chem.BondType.SINGLE)
    mol = em.GetMol()

    # Incremental sanitization
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_PROPERTIES)
    except Exception:
        return None

    ring_count = Chem.GetSSSR(mol)

    if not allow_cycles and ring_count > 0:
        return None

    if allow_cycles and not allow_bicycles and ring_count > 1:
        return None

    return mol


# ============================================================
# Assembly engine
# ============================================================

def assemble_fragments(fragments,
                       max_solutions=10,
                       allow_cycles=True,
                       allow_bicycles=True):
    """
    Enumeratively assemble fragments into valid molecules.
    """

    solutions = set()
    results = []

    def dfs(current_mol, remaining_frags):
        if len(results) >= max_solutions:
            return

        if not remaining_frags:
            try:
                Chem.SanitizeMol(current_mol)
                smi = Chem.MolToSmiles(current_mol, canonical=True)
                if smi not in solutions:
                    solutions.add(smi)
                    results.append(smi)
            except Exception:
                pass
            return

        attachA = attachment_atoms(current_mol)
        if not attachA:
            return

        for i, frag in enumerate(remaining_frags):
            attachB = attachment_atoms(frag)
            if not attachB:
                continue

            for a in attachA:
                for b in attachB:
                    new_mol = attach_mols(
                        current_mol, a, frag, b,
                        allow_cycles=allow_cycles,
                        allow_bicycles=allow_bicycles
                    )
                    if new_mol is None:
                        continue

                    dfs(
                        new_mol,
                        remaining_frags[:i] + remaining_frags[i+1:]
                    )

    # Try each fragment as a starting point
    for i, start in enumerate(fragments):
        dfs(start, fragments[:i] + fragments[i+1:])

    return results


# ============================================================
# User-facing API
# ============================================================

def build_molecules(group_counts,
                    max_solutions=10,
                    allow_cycles=True,
                    allow_bicycles=True):
    """
    group_counts: dict { group_name: exact_count }
    """

    fragments = []

    for group, count in group_counts.items():
        if group not in SMARTS_DB:
            raise KeyError(f"Unknown SMARTS group: {group}")

        smarts_mols = mols_from_smarts_entry(SMARTS_DB[group])

        # Use the first SMARTS variant by default
        for _ in range(count):
            fragments.append(copy.deepcopy(smarts_mols[0]))

    return assemble_fragments(
        fragments,
        max_solutions=max_solutions,
        allow_cycles=allow_cycles,
        allow_bicycles=allow_bicycles
    )


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    group_counts = {
        "CH3": 1,
        "CH2": 3,
        "OH": 1
    }

    smiles_list = build_molecules(
        group_counts,
        max_solutions=10,
        allow_cycles=False,
        allow_bicycles=False
    )

    for smi in smiles_list:
        print(smi)
