"""
THF-Water: binary VLE (T-xy) diagrams for A-B, A-E and B-E pairs.

A-B (THF-water) is independent of the entrainer, so it is computed once.
For each candidate entrainer (E), computes VLE along the two entrainer-specific
binary edges of the ternary system:
    A-E: THF-entrainer   (x_B = 0)
    B-E: water-entrainer (x_A = 0)

Total plots: 1 (A-B) + 2 per entrainer x 3 entrainers = 7.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.molecules import StructuralValidator
from src.pure import PropertyCalculator, AntoineCalculator
from src.data import load_unifac_data
from src.mixture.unifac.pyomo_vle import pyo_vle


ENTRAINERS = {
    "ethylene_glycol": {"DOH": 1},
    "DMSO": {"DMSO": 1},
    "Glycerol": {"CH2": 2, "CH": 1, "OH": 3},
}

N_DATA = 50


def solve_binary_edge(mixture, t, active_idx, T0, rk_data, qk_data, interaction_data, antoine_params):
    """Solve VLE along a binary edge of the ternary/binary system.

    active_idx: tuple of the two component indices that vary (any remaining
    components are held at 0).
    """
    n_comp = len(mixture)
    x = np.zeros((len(t), n_comp))
    x[:, active_idx[0]] = t
    x[:, active_idx[1]] = 1 - t

    results, model = pyo_vle(x, T0, mixture, rk_data, qk_data, interaction_data,
                              antoine_params, verbose=False)

    T = np.empty_like(T0)
    y = np.empty_like(x)
    for i in model.T.keys():
        T[i] = model.T[i].value
    for i in model.y.keys():
        y[i] = model.y[i].value

    return results, x, T, y


def plot_edge(ax, x, T, y, idx, label_a, title):
    ax.plot(x[:, idx], T, 'r-', label='Liquid')
    ax.plot(y[:, idx], T, 'b-', label='Vapor')
    ax.set_xlabel(f'{label_a} fraction')
    ax.set_ylabel('Temperature (K)')
    ax.set_title(title)
    ax.legend()
    ax.grid()


def main():
    rk_data, qk_data, interaction_data = load_unifac_data()
    t = np.linspace(0.0, 1.0, N_DATA)

    fig, axes = plt.subplots(1 + len(ENTRAINERS), 2, figsize=(12, 5 * (1 + len(ENTRAINERS))))

    # --- A-B: THF-water (entrainer-independent, computed once) ---
    binary_mixture = {'THF': {'THF': 1}, 'water': {'H2O': 1}}
    validator = StructuralValidator()
    validator.validate(binary_mixture)

    prop_list = ['tc', 'pc', 'tb', 'ac', 'hv']
    pure_calc = PropertyCalculator()
    binary_props = pure_calc.calculate(binary_mixture, prop_list)
    binary_props['THF']['tc'] = 540.2
    binary_props['THF']['pc'] = 51.90
    binary_props['THF']['tb'] = 339.1
    binary_props['water']['tc'] = 647.1
    binary_props['water']['pc'] = 220.6
    binary_props['water']['tb'] = 373.2

    antoine_calc = AntoineCalculator()
    binary_antoine = {}
    for compound, props in binary_props.items():
        res = antoine_calc.calculate_antoine_parameters(
            binary_mixture, critical_properties=props, method='enthalpy_vap'
        )
        binary_antoine[compound] = {'A': res['A'], 'B': res['B'], 'C': res['C']}

    T0_AB = np.linspace(binary_props['water']['tb'], binary_props['THF']['tb'], N_DATA)
    results_AB, x_AB, T_AB, y_AB = solve_binary_edge(
        binary_mixture, t, (0, 1), T0_AB, rk_data, qk_data, interaction_data, binary_antoine
    )
    print(f"A-B (THF-water): {results_AB.solver.status}/{results_AB.solver.termination_condition}")

    # Turn off the unused second column in the first row, plot A-B in the first cell
    plot_edge(axes[0, 0], x_AB, T_AB, y_AB, 0, 'THF', 'A-B (THF-water)')
    axes[0, 1].axis('off')

    # --- A-E and B-E for each entrainer ---
    for row, (name, groups) in enumerate(ENTRAINERS.items(), start=1):
        mixture = {'THF': {'THF': 1}, 'water': {'H2O': 1}, name: groups}
        comps = list(mixture.keys())  # [THF, water, entrainer] -> indices 0, 1, 2

        validator = StructuralValidator()
        validator.validate(mixture)

        pure_props = pure_calc.calculate(mixture, prop_list)
        pure_props['THF']['tc'] = 540.2
        pure_props['THF']['pc'] = 51.90
        pure_props['THF']['tb'] = 339.1
        pure_props['water']['tc'] = 647.1
        pure_props['water']['pc'] = 220.6
        pure_props['water']['tb'] = 373.2

        antoine_params = {}
        for compound, props in pure_props.items():
            res = antoine_calc.calculate_antoine_parameters(
                mixture, critical_properties=props, method='enthalpy_vap'
            )
            antoine_params[compound] = {'A': res['A'], 'B': res['B'], 'C': res['C']}

        tb = {c: pure_props[c]['tb'] for c in mixture}

        # A-E: x_B = 0
        T0_AE = np.linspace(tb[comps[2]], tb[comps[0]], N_DATA)
        res_AE, x_AE, T_AE, y_AE = solve_binary_edge(
            mixture, t, (0, 2), T0_AE, rk_data, qk_data, interaction_data, antoine_params
        )
        print(f"{name} A-E (THF-{name}): {res_AE.solver.status}/{res_AE.solver.termination_condition}")
        plot_edge(axes[row, 0], x_AE, T_AE, y_AE, 0, 'THF', f'{name}: A-E (THF-{name})')

        # B-E: x_A = 0
        T0_BE = np.linspace(tb[comps[2]], tb[comps[1]], N_DATA)
        res_BE, x_BE, T_BE, y_BE = solve_binary_edge(
            mixture, t, (1, 2), T0_BE, rk_data, qk_data, interaction_data, antoine_params
        )
        print(f"{name} B-E (water-{name}): {res_BE.solver.status}/{res_BE.solver.termination_condition}")
        plot_edge(axes[row, 1], x_BE, T_BE, y_BE, 1, 'water', f'{name}: B-E (water-{name})')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thf_water_binary_vle.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    main()
