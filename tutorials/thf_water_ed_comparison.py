"""
THF-Water: E/D vs composition for each candidate entrainer.

Same VLE + E/D approach as tutorials/08_iss.ipynb, applied to the THF-water
binary with each of the three predefined entrainers (ethylene glycol, DMSO,
glycerol) evaluated separately along the A-E edge (water fraction = 0).
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

N_DATA = 100


def main():
    rk_data, qk_data, interaction_data = load_unifac_data()

    x1 = np.linspace(0.0, 1.0, N_DATA)
    x2 = np.zeros(N_DATA)
    x3 = 1 - x1
    x_thf_water = np.column_stack([x1, x2, x3])

    results_by_entrainer = {}

    for name, groups in ENTRAINERS.items():
        mixture = {'THF': {'THF': 1}, 'water': {'H2O': 1}, name: groups}

        validator = StructuralValidator()
        validator.validate(mixture)

        prop_list = ['tc', 'pc', 'tb', 'ac', 'hv']
        pure_calc = PropertyCalculator()
        pure_props = pure_calc.calculate(mixture, prop_list)

        # Override THF/water with experimental data (same as screen_thf_water.yaml)
        pure_props['THF']['tc'] = 540.2
        pure_props['THF']['pc'] = 51.90
        pure_props['THF']['tb'] = 339.1
        pure_props['water']['tc'] = 647.1
        pure_props['water']['pc'] = 220.6
        pure_props['water']['tb'] = 373.2

        antoine_calc = AntoineCalculator()
        antoine_params = {}
        for compound, props in pure_props.items():
            res = antoine_calc.calculate_antoine_parameters(
                mixture, critical_properties=props, method='enthalpy_vap'
            )
            antoine_params[compound] = {'A': res['A'], 'B': res['B'], 'C': res['C']}

        tb1 = pure_props['THF']['tb']
        tb3 = pure_props[name]['tb']
        T0 = np.linspace(tb3, tb1, N_DATA)

        solver_results, model = pyo_vle(
            x_thf_water, T0, mixture, rk_data, qk_data, interaction_data,
            antoine_params, verbose=False
        )
        print(f"{name}: {solver_results.solver.status}/{solver_results.solver.termination_condition}")

        gamma = np.empty_like(x_thf_water)
        T = np.empty_like(T0)
        psat = np.empty_like(x_thf_water)

        for i in model.gamma.keys():
            gamma[i] = model.gamma[i].value
        for i in model.T.keys():
            T[i] = model.T[i].value
        for i in model.psat.keys():
            psat[i] = model.psat[i].value

        K = gamma * psat / 1.01325
        ED = 1 - ((1 - K[:, 1]) / (x_thf_water[:, 0] * (K[:, 0] - K[:, 1])))

        results_by_entrainer[name] = {'x': x_thf_water[:, 0], 'ED': ED}

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in results_by_entrainer.items():
        ax.plot(data['x'], data['ED'], label=name)

    ax.set_xlabel('THF liquid fraction (x_THF)')
    ax.set_ylabel('E/D')
    ax.set_title('E/D vs THF fraction for each entrainer (THF-Water system)')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
