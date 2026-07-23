"""
Ethanol-Water: binary VLE (T-xy) diagram using UNIFAC.

Computes the T-xy diagram for the ethanol-water binary across the full
composition range. Experimental data for ethanol (Tc, Pc, Tb) can be
supplied via the EXPERIMENTAL_PROPS override below once available.
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


MIXTURE = {
    'ethanol': {'CH3': 1, 'CH2': 1, 'OH': 1},
    'water': {'H2O': 1},
}

# Experimental overrides (fill in once available). Leave a key out to use the
# Constantinou-Gani group-contribution estimate instead.
EXPERIMENTAL_PROPS = {
    'ethanol': {
        'tc': 514.7,  # K
        'pc': 63.4,  # bar
        'tb': 351.5,  # K
    },
    'water': {
        'tc': 647.1,   # K
        'pc': 220.6,   # bar
        'tb': 373.2,   # K
    },
}

N_DATA = 100


def main():
    rk_data, qk_data, interaction_data = load_unifac_data()

    validator = StructuralValidator()
    validator.validate(MIXTURE)

    prop_list = ['tc', 'pc', 'tb', 'ac', 'hv']
    pure_calc = PropertyCalculator()
    pure_props = pure_calc.calculate(MIXTURE, prop_list)

    for compound, overrides in EXPERIMENTAL_PROPS.items():
        for prop, value in overrides.items():
            if value is not None:
                pure_props[compound][prop] = value

    print("Pure component properties:")
    for compound, props in pure_props.items():
        print(f"  {compound}: {props}")

    antoine_calc = AntoineCalculator()
    antoine_params = {}
    for compound, props in pure_props.items():
        res = antoine_calc.calculate_antoine_parameters(
            MIXTURE, critical_properties=props, method='enthalpy_vap'
        )
        antoine_params[compound] = {'A': res['A'], 'B': res['B'], 'C': res['C']}
        print(f"  {compound} Antoine: {antoine_params[compound]}")

    t = np.linspace(0.001, 0.999, N_DATA)
    x = np.column_stack([t, 1 - t])
    T0 = np.linspace(pure_props['water']['tb'], pure_props['ethanol']['tb'], N_DATA)

    results, model = pyo_vle(x, T0, MIXTURE, rk_data, qk_data, interaction_data,
                              antoine_params, verbose=False)
    print(f"VLE solve: {results.solver.status}/{results.solver.termination_condition}")

    gamma = np.empty_like(x)
    T = np.empty_like(T0)
    y = np.empty_like(x)
    psat = np.empty_like(x)
    for i in model.gamma.keys():
        gamma[i] = model.gamma[i].value
    for i in model.T.keys():
        T[i] = model.T[i].value
    for i in model.y.keys():
        y[i] = model.y[i].value
    for i in model.psat.keys():
        psat[i] = model.psat[i].value

    K = gamma * psat / 1.01325
    alpha = K[:, 0] / K[:, 1]  # ethanol/water relative volatility

    print(f"Min T: {T.min():.2f} K at x_ethanol = {x[T.argmin(), 0]:.3f}")
    print(f"Tb ethanol: {pure_props['ethanol']['tb']:.2f} K, Tb water: {pure_props['water']['tb']:.2f} K")
    sign_changes = np.where(np.diff(np.sign(alpha - 1)))[0]
    if len(sign_changes):
        print(f"alpha=1 crossing(s) at x_ethanol ~ {x[sign_changes, 0]}")
    else:
        print("No alpha=1 crossing found - no azeotrope predicted by UNIFAC")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(x[:, 0], T, 'r-', label='Liquid')
    axes[0].plot(y[:, 0], T, 'b-', label='Vapor')
    axes[0].set_xlabel('Ethanol fraction')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title('T-xy Diagram: Ethanol-Water (UNIFAC)')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(x[:, 0], alpha, 'g-')
    axes[1].axhline(1.0, color='k', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('Ethanol fraction')
    axes[1].set_ylabel('alpha (ethanol/water)')
    axes[1].set_title('Relative Volatility: Ethanol-Water')
    axes[1].grid()

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ethanol_water_vle.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    main()
