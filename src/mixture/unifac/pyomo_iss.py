"""
Pyomo UNIFAC implementation for optimization and molecular design.

Following the established coding style with commented-out variables and 
the specific variable naming convention (tau, s, eta, G, etc.).
"""

import pyomo.environ as pyo
import numpy as np
from typing import Dict, List, Optional, Tuple
from .interface import unifac


def _convert_mixture_format(mixture):
    """
    Convert mixture from compound format to group format.
    
    Args:
        mixture: Compound format {compound_name: {group: count}}
    
    Returns:
        Group format {group: [counts for each component]}
    """
    # Extract compounds (the nested dicts)
    compounds = list(mixture.values())
    
    # Get all unique groups across all compounds
    all_groups = set()
    for compound in compounds:
        all_groups.update(compound.keys())
    all_groups = sorted(all_groups)
    
    # Build group format: {group: [count_in_comp1, count_in_comp2, ...]}
    group_format = {}
    for group in all_groups:
        group_format[group] = [compound.get(group, 0) for compound in compounds]
    
    return group_format


def _convert_antoine_format(antoine_params):
    """
    Convert Antoine parameters from nested dict to array format.
    
    Args:
        antoine_params: Dict format {compound_name: {'A': val, 'B': val, 'C': val}}
    
    Returns:
        Array format (n_components, 3) with [A, B, C] for each component
    """
    compounds = list(antoine_params.keys())
    n_comp = len(compounds)
    
    # Build array: [[A1, B1, C1], [A2, B2, C2], ...]
    antoine_array = np.zeros((n_comp, 3))
    for i, compound in enumerate(compounds):
        antoine_array[i, 0] = antoine_params[compound]['A']
        antoine_array[i, 1] = antoine_params[compound]['B'] 
        antoine_array[i, 2] = antoine_params[compound]['C']
    
    return antoine_array


def _generate_initialization_values(x, T, mixture, rk_data, qk_data, interaction_data, antoine_params):
    """
    Generate initialization values using the UNIFAC interface for better convergence.
    
    Args:
        x: liquid composition array (n_data, n_components)
        T: temperature array (n_data,) in Kelvin
        mixture: compound format {compound_name: {group: count}}
        rk_data, qk_data, interaction_data: UNIFAC parameter dictionaries
        antoine_params: Antoine parameters dict
    
    Returns:
        Dict with initialization values for all variables
    """
    try:
        # Calculate UNIFAC values using the interface
        results = unifac(mixture, x, T, rk_data, qk_data, interaction_data)
        
        # Convert Antoine parameters
        antoine_array = _convert_antoine_format(antoine_params)
        n_data, n_comp = x.shape
        
        # Calculate vapor pressures using Antoine equation
        psat_init = np.zeros((n_data, n_comp))
        for j in range(n_data):
            for i in range(n_comp):
                A, B, C = antoine_array[i]
                psat_init[j, i] = 10**(A - B / (C + T[j]))
        
        # Calculate initial vapor compositions using VLE equilibrium
        P = 1.01325  # Default pressure in bar
        y_init = np.zeros((n_data, n_comp))
        for j in range(n_data):
            for i in range(n_comp):
                y_init[j, i] = psat_init[j, i] * x[j, i] * results['gamma'][j, i] / P
            # Normalize y to sum to 1
            y_init[j, :] = y_init[j, :] / np.sum(y_init[j, :])
        
        # Create initialization dictionary
        init_values = {
            'r': results['r'],
            'q': results['q'], 
            'J': results['J'],
            'L': results['L'],
            'e': results['e'],
            'theta': results['theta'],
            'psi': results['psi'],
            'beta': results['beta'],
            'omega': results['omega'],
            'log_gamma_c': results['log_gamma_c'],
            'log_gamma_r': results['log_gamma_r'],
            'gamma': results['gamma'],
            'psat': psat_init,
            'y': y_init
        }
        
        return init_values
        
    except Exception as e:
        print(f"Warning: Failed to generate initialization values: {e}")
        print("Using default initialization values instead.")
        return None


def pyo_iss(x, T, mixture, rk_data, qk_data, interaction_data, antoine_params, x_iso,verbose=False):
    """
    Pyomo model for vapour-liquid equilibrium.
    
    :param x: liquid composition array (n_data, n_components)
    :param T: temperature array (n_data,) in Kelvin
    :param mixture: compound format {compound_name: {group: count}}
    :param rk_data: UNIFAC Rk parameters dict
    :param qk_data: UNIFAC Qk parameters dict  
    :param interaction_data: UNIFAC interaction parameters dict
    :param antoine_params: Antoine parameters dict {compound_name: {'A': val, 'B': val, 'C': val}}
    :param verbose: If True, print solver output
    :return: (solution, model)
    """
    # Generate initialization values using original mixture format
    init_values = _generate_initialization_values(x, T, mixture, rk_data, qk_data, interaction_data, antoine_params)
    
    # Auto-convert formats for pyomo model
    mixture = _convert_mixture_format(mixture)
    antoine_array = _convert_antoine_format(antoine_params)

    # we need the 
    def x_bounds_rule(model, dp, i):
      if dp == 0 and i == 0:  # Only for x[0,0]
          return (0.1, x_iso)  # (lower, upper)
      else:
          return (0, 1)  # Default bounds for other variables
    
    # create the model
    model = pyo.ConcreteModel('vle')
    
    # Formats already converted above
    
    # define the parameters of the model
    x = np.atleast_2d(x)
    T = np.atleast_1d(T)
    n_comp = x.shape[1]
    n_data = x.shape[0]
    groups = list(mixture.keys())
    all_groups = list(rk_data.keys())
    
    # define sets
    model.k = pyo.Set(initialize=groups, name='Groups present')
    model.i = pyo.RangeSet(0, n_comp - 1, name='no. of Compounds')
    model.dp = pyo.RangeSet(0, n_data - 1, name='no. of data points')
    model.grp = pyo.Set(initialize=all_groups, name='all Groups')
    
    # define the constants/parameters
    model.T = pyo.Var(model.dp, initialize=T, 
                       within=pyo.PositiveReals, name='azeotrope Temperature in K')
    
    # Flatten interaction parameters for Pyomo
    a_dict = {}
    for k1 in interaction_data:
        for k2 in interaction_data[k1]:
            a_dict[(k1, k2)] = interaction_data[k1][k2]
    model.a = pyo.Param(model.grp, model.grp, initialize=a_dict, mutable=False,
                        name='unifac interaction a_mk', domain=pyo.Reals)
    
    model.Rk = pyo.Param(model.grp, initialize=rk_data, mutable=False, name='unifac Rk parameters')
    model.Qk = pyo.Param(model.grp, initialize=qk_data, mutable=False, name='unifac Qk parameters')
    
    model.x = pyo.Var(model.dp, model.i, 
                       initialize={(j, i): float(x[j, i]) for j in range(n_data) for i in range(n_comp)},
                       name='Liquid composition x', bounds=x_bounds_rule, domain=pyo.UnitInterval)

    # Group composition matrix
    v_dict = {}
    for i in range(n_comp):
        for k in groups:
            v_dict[(i, k)] = mixture[k][i]
    model.v = pyo.Param(model.i, model.k, initialize=v_dict, mutable=False, name='compounds')
    
    # Antoine parameters
    model.A = pyo.Param(model.i, initialize={i: float(antoine_array[i, 0]) for i in range(n_comp)}, 
                       mutable=False, name='Antoine A', domain=pyo.Reals)
    model.B = pyo.Param(model.i, initialize={i: float(antoine_array[i, 1]) for i in range(n_comp)}, 
                       mutable=False, name='Antoine B', domain=pyo.Reals)
    model.C = pyo.Param(model.i, initialize={i: float(antoine_array[i, 2]) for i in range(n_comp)}, 
                       mutable=False, name='Antoine C', domain=pyo.Reals)

    # System pressure parameter (default 1.01325 bar = 1 atm)
    model.P = pyo.Param(initialize=1.01325, mutable=False, name='Total pressure in bar', domain=pyo.PositiveReals)
    
    # Helper function to create initialization dictionaries
    def create_init_dict_1d(values, fallback):
        """Create 1D initialization dict"""
        if init_values is not None and values is not None:
            return {i: float(values[i]) for i in range(len(values))}
        return fallback
    
    def create_init_dict_2d(values, fallback, keys1=None, keys2=None):
        """Create 2D initialization dict with optional custom keys"""
        if init_values is not None and values is not None:
            values = np.atleast_2d(values)  # Ensure 2D for single data point case
            if keys1 is not None and keys2 is not None:
                return {(k1, k2): float(values[i, j]) for i, k1 in enumerate(keys1) for j, k2 in enumerate(keys2)}
            else:
                return {(j, i): float(values[j, i]) for j in range(values.shape[0]) for i in range(values.shape[1])}
        return fallback
    
    def create_init_dict_3d(values, fallback, keys1=None, keys2=None, keys3=None):
        """Create 3D initialization dict with optional custom keys"""
        if init_values is not None and values is not None:
            # Ensure at least 3D (handle single data point case)
            if values.ndim == 2:
                values = values[np.newaxis, :, :]  # Add data point dimension
            if keys1 is not None and keys2 is not None and keys3 is not None:
                return {(k1, k2, k3): float(values[i, j, k]) 
                       for i, k1 in enumerate(keys1) 
                       for j, k2 in enumerate(keys2) 
                       for k, k3 in enumerate(keys3)}
            else:
                return {(j, i, k): float(values[j, i, k]) for j in range(values.shape[0]) 
                        for i in range(values.shape[1]) for k in range(values.shape[2])}
        return fallback

    # define the model variables - matching equations.py nomenclature exactly
    model.r = pyo.Var(model.i, 
                     initialize=create_init_dict_1d(init_values['r'] if init_values else None, 1.2), 
                     within=pyo.PositiveReals, name='unifac r_i')
    model.q = pyo.Var(model.i, 
                     initialize=create_init_dict_1d(init_values['q'] if init_values else None, 4.5), 
                     within=pyo.PositiveReals, name='unifac q_i')
    model.J = pyo.Var(model.dp, model.i, 
                     initialize=create_init_dict_2d(init_values['J'] if init_values else None, 1.13), 
                     within=pyo.PositiveReals, name='unifac J_i')
    model.L = pyo.Var(model.dp, model.i, 
                     initialize=create_init_dict_2d(init_values['L'] if init_values else None, 1.15), 
                     within=pyo.PositiveReals, name='unifac L_i')
    model.e = pyo.Var(model.i, model.k, 
                     initialize=create_init_dict_2d(init_values['e'] if init_values else None, 0.5, 
                                                   keys1=range(n_comp), keys2=groups), 
                     within=pyo.NonNegativeReals, name='unifac e_ik')
    model.theta = pyo.Var(model.dp, model.k, 
                         initialize=create_init_dict_2d(init_values['theta'] if init_values else None, 1.2,
                                                       keys1=range(n_data), keys2=groups), 
                         within=pyo.NonNegativeReals, name='unifac theta_k')
    
    model.psi = pyo.Var(model.dp, model.k, model.k, 
                       initialize=create_init_dict_3d(init_values['psi'] if init_values else None, 0.79,
                                                     keys1=range(n_data), keys2=groups, keys3=groups), 
                       within=pyo.PositiveReals, name='unifac psi_mk')
    model.beta = pyo.Var(model.dp, model.i, model.k, 
                        initialize=create_init_dict_3d(init_values['beta'] if init_values else None, 1.7,
                                                      keys1=range(n_data), keys2=range(n_comp), keys3=groups),
                          within=pyo.PositiveReals, name='unifac beta_ik')
    model.omega = pyo.Var(model.dp, model.k, 
                         initialize=create_init_dict_2d(init_values['omega'] if init_values else None, 1.1,
                                                       keys1=range(n_data), keys2=groups), 
                         within=pyo.PositiveReals, name='unifac omega_k')
    model.log_gamma_c = pyo.Var(model.dp, model.i, 
                               initialize=create_init_dict_2d(init_values['log_gamma_c'] if init_values else None, 0.1), 
                               within=pyo.Reals, name='unifac log_gamma_c')
    model.log_gamma_r = pyo.Var(model.dp, model.i, 
                               initialize=create_init_dict_2d(init_values['log_gamma_r'] if init_values else None, 0.1), 
                               within=pyo.Reals, name='unifac log_gamma_r')
    model.gamma = pyo.Var(model.dp, model.i, 
                         initialize=create_init_dict_2d(init_values['gamma'] if init_values else None, 1.5), 
                         within=pyo.Reals, name='unifac total gamma')
    
    # VLE-specific variables
    model.psat = pyo.Var(model.dp, model.i, 
                        initialize=create_init_dict_2d(init_values['psat'] if init_values else None, 101325), 
                        within=pyo.PositiveReals, name='vapor pressure in Pa')
    model.y = pyo.Var(model.dp, model.i, 
                     initialize=create_init_dict_2d(init_values['y'] if init_values else None, 0.5), 
                     within=pyo.UnitInterval, name='vapor composition y')
    
    def eq_r(m, i):
        """
        Molecular volume parameter (Equation A.3):
        r_i = Σ ν_{i,k} R_k
        """
        return m.r[i] == sum(m.Rk[k] * m.v[i, k] for k in m.k)
    
    model.eq_r = pyo.Constraint(model.i, rule=eq_r, name='Constraint for calculating r_i of unifac')
    
    def eq_q(m, i):
        """
        Molecular surface area parameter (Equation A.4):
        q_i = Σ ν_{i,k} Q_k
        """
        return m.q[i] == sum(m.Qk[k] * m.v[i, k] for k in m.k)
    
    model.eq_q = pyo.Constraint(model.i, rule=eq_q, name='Constraint for calculating q_i of unifac')
    
    def eq_J(m, j, i):
        """
        Volume fraction parameter (Equation A.9):
        J_{i,t} = r_i / Σ x_τ r_τ
        """
        return m.J[j, i] == m.r[i] / sum(m.r[ii] * m.x[j, ii] for ii in m.i)
    
    model.eq_J = pyo.Constraint(model.dp, model.i, rule=eq_J, name='Constraint for calculating J_i of unifac')
    
    def eq_L(m, j, i):
        """
        Surface area fraction parameter:
        L_i = q_i / Σ x_j q_j
        """
        return m.L[j, i] == m.q[i] / sum(m.q[ii] * m.x[j, ii] for ii in m.i)
    
    model.eq_L = pyo.Constraint(model.dp, model.i, rule=eq_L, name='Constraint for calculating L_i of unifac')
    
    def eq_e(m, i, k):
        """
        Group surface fraction in component:
        e_{i,k} = (v_{i,k} * Q_k) / q_i
        """
        return m.e[i, k] == m.v[i, k] * m.Qk[k] / m.q[i]
    
    model.eq_e = pyo.Constraint(model.i, model.k, rule=eq_e, name='Constraint for calculating e_ik of unifac')
    
    def eq_theta(m, dp, k):
        """
        Group surface area fraction in mixture:
        theta_k = Σ x_i v_{i,k} Q_k / Σ x_i q_i
        """
        return m.theta[dp, k] == (sum(m.x[dp, i] * m.v[i, k] * m.Qk[k] for i in m.i) / 
                                 sum(m.x[dp, i] * m.q[i] for i in m.i))
    
    model.eq_theta = pyo.Constraint(model.dp, model.k, rule=eq_theta,
                                    name='Constraint for calculating theta_k of unifac')
    
    def eq_psi(m, dp, k, l):
        """
        Group interaction parameter:
        psi_{k,l} = exp(-a_{k,l} / T)
        """
        return m.psi[dp, k, l] == pyo.exp(-m.a[k, l] / m.T[dp])
    
    model.eq_psi = pyo.Constraint(model.dp, model.k, model.k, rule=eq_psi,
                                  name='Constraint for calculating psi_kl of unifac')
    
    def eq_beta(m, dp, i, k):
        """
        Beta parameter:
        beta_{i,k} = Σ v_{i,l} Q_l psi_{l,k} / q_i
        """
        return m.beta[dp, i, k] == (sum(m.v[i, l] * m.Qk[l] * m.psi[dp, l, k] for l in m.k) / m.q[i])
    
    model.eq_beta = pyo.Constraint(model.dp, model.i, model.k, rule=eq_beta,
                                   name='Constraint for calculating beta_ik of unifac')
    
    def eq_omega(m, dp, k):
        """
        Omega parameter:
        omega_k = Σ theta_l psi_{l,k}
        """
        return m.omega[dp, k] == sum(m.theta[dp, l] * m.psi[dp, l, k] for l in m.k)
    
    model.eq_omega = pyo.Constraint(model.dp, model.k, rule=eq_omega,
                                    name='Constraint for calculating omega_k of unifac')
    
    def eq_log_gamma_c(m, dp, i):
        """
        Combinatorial activity coefficient:
        log_gamma_c = 1 - J + ln(J) - 5*q*(1 - J/L + ln(J/L))
        """
        return m.log_gamma_c[dp, i] == (1 - m.J[dp, i] + pyo.log(m.J[dp, i]) - 
                                       5 * m.q[i] * (1 - m.J[dp, i] / m.L[dp, i] + 
                                                    pyo.log(m.J[dp, i] / m.L[dp, i])))
    
    model.eq_log_gamma_c = pyo.Constraint(model.dp, model.i, rule=eq_log_gamma_c,
                                         name='Constraint for calculating log_gamma_c of unifac')
    
    def eq_log_gamma_r(m, dp, i):
        """
        Residual activity coefficient:
        log_gamma_r = q * (1 - Σ(theta * beta / omega - e * ln(beta / omega)))
        """
        return m.log_gamma_r[dp, i] == m.q[i] * (1 - sum(m.theta[dp, k] * m.beta[dp, i, k] / m.omega[dp, k] - 
                                                         m.e[i, k] * pyo.log(m.beta[dp, i, k] / m.omega[dp, k]) 
                                                         for k in m.k))
    
    model.eq_log_gamma_r = pyo.Constraint(model.dp, model.i, rule=eq_log_gamma_r,
                                         name='Constraint for calculating log_gamma_r of unifac')
    
    def eq_gamma(m, dp, i):
        """
        Total activity coefficient:
        gamma = exp(log_gamma_c + log_gamma_r)
        """
        return m.gamma[dp, i] == pyo.exp(m.log_gamma_c[dp, i] + m.log_gamma_r[dp, i])
    
    model.eq_gamma = pyo.Constraint(model.dp, model.i, rule=eq_gamma, name='Constraint for calculating gamma')
    
    # VLE-specific constraints
    def eq_psat(m, dp, i):
        """Antoine equation for vapor pressure"""
        return m.psat[dp, i] == 10**(m.A[i] - m.B[i] / (m.C[i] + m.T[dp]))
    model.eq_psat = pyo.Constraint(model.dp, model.i, rule=eq_psat, name='Antoine equation for vapor pressure')
    
    def eq_y(m, dp, i):
        """Vapor-liquid equilibrium (Raoult's law with activity coefficients)"""
        return m.y[dp, i] == m.psat[dp, i] * m.x[dp, i] * m.gamma[dp, i] / m.P
    model.eq_y = pyo.Constraint(model.dp, model.i, rule=eq_y, name='VLE equilibrium equation')
    
    def eq_sum_y(m, dp):
        """Vapor composition normalization"""
        return sum(m.y[dp, i] for i in m.i) == 1.0
    model.eq_sum_y = pyo.Constraint(model.dp, rule=eq_sum_y, name='Vapor composition normalization')

    def eq_xb(m):
        """ absence of B in the rectifying section"""
        return m.x[0, 1] == 0
    
    model.eq_xb = pyo.Constraint(rule=eq_xb, name='absence of B in the rectifying section')
    
    def eq_ED(m):
        """ entrainer-to-distilate ratio equaton"""
        return 1 - ((1 - (m.gamma[0, 1] * m.psat[0, 1]))/ (m.x[0, 0] * ((m.gamma[0, 0] * m.psat[0, 0]) - (m.gamma[0, 1] * m.psat[0, 1]))))

    model.obj = pyo.Objective(rule=eq_ED, sense=pyo.minimize, name='Objective: Minimize entrainer-to-distilate ratio')

    def eq_sum_x(m, dp):
        """liquid composition normalization"""
        return sum(m.x[dp, i] for i in m.i) == 1.0
    model.eq_sum_x = pyo.Constraint(model.dp, rule=eq_sum_x, name='liquid composition normalization')
    


    
    # define the solver
    opt = pyo.SolverFactory('ipopt')
    # define solver options
    opt.options["halt_on_ampl_error"] = "yes"
    opt.options["max_iter"] = 10000
    opt.options["tol"] = 1e-8
    # solve the optimization problem
    solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True)
    
    return solution, model


if __name__ == "__main__":
    print("=== Pyomo VLE Implementation ===")
    print("Use pyo_vle() function for VLE calculations in optimization frameworks.")