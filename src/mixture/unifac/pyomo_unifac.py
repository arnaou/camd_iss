"""
Pyomo UNIFAC implementation for optimization and molecular design.

Following the established coding style with commented-out variables and 
the specific variable naming convention (tau, s, eta, G, etc.).
"""

import pyomo.environ as pyo
import numpy as np
from typing import Dict, List, Optional, Tuple


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


def pyo_unifac(x, T, mixture, rk_data, qk_data, interaction_data):
    """
    Pyomo UNIFAC model for activity coefficient calculations in optimization.
    
    :param x: liquid composition array (n_data, n_components)
    :param T: temperature array (n_data,) in Kelvin
    :param mixture: group format {group: [counts for each component]}
    :param rk_data: UNIFAC Rk parameters dict
    :param qk_data: UNIFAC Qk parameters dict  
    :param interaction_data: UNIFAC interaction parameters dict
    :return: (solution, model)
    """
    # create the model
    model = pyo.ConcreteModel()
    
    # Auto-convert mixture format 
    mixture = _convert_mixture_format(mixture)
    
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
    model.T = pyo.Param(model.dp, initialize={j: float(T[j]) for j in range(n_data)}, 
                       within=pyo.PositiveReals, name='Temperature in K')
    
    # Flatten interaction parameters for Pyomo
    a_dict = {}
    for k1 in interaction_data:
        for k2 in interaction_data[k1]:
            a_dict[(k1, k2)] = interaction_data[k1][k2]
    model.a = pyo.Param(model.grp, model.grp, initialize=a_dict, mutable=False,
                        name='unifac interaction a_mk', domain=pyo.Reals)
    
    model.Rk = pyo.Param(model.grp, initialize=rk_data, mutable=False, name='unifac Rk parameters')
    model.Qk = pyo.Param(model.grp, initialize=qk_data, mutable=False, name='unifac Qk parameters')
    
    model.x = pyo.Param(model.dp, model.i, 
                       initialize={(j, i): float(x[j, i]) for j in range(n_data) for i in range(n_comp)},
                       mutable=False, name='Liquid composition x', domain=pyo.Reals)
    
    # Group composition matrix
    v_dict = {}
    for i in range(n_comp):
        for k in groups:
            v_dict[(i, k)] = mixture[k][i]
    model.v = pyo.Param(model.i, model.k, initialize=v_dict, mutable=False, name='compounds')
    
    # define the model variables
    model.r = pyo.Var(model.i, initialize=1.2, within=pyo.PositiveReals, name='unifac r_i')
    model.q = pyo.Var(model.i, initialize=4.5, within=pyo.PositiveReals, name='unifac q_i')
    model.J = pyo.Var(model.dp, model.i, initialize=1.13, within=pyo.PositiveReals, name='unifac J_i')
    model.L = pyo.Var(model.dp, model.i, initialize=1.15, within=pyo.PositiveReals, name='unifac L_i')
    model.G = pyo.Var(model.k, model.i, initialize=1.5, within=pyo.PositiveReals, name='unifac G_ki')
    model.theta = pyo.Var(model.dp, model.k, initialize=1.2, within=pyo.PositiveReals, name='unifac theta_k')
    
    model.tau = pyo.Var(model.dp, model.k, model.k, initialize=0.79, within=pyo.PositiveReals, name='unifac tau_mk')
    model.s = pyo.Var(model.dp, model.k, model.i, initialize=1.7, within=pyo.PositiveReals, name='unifac s_ki')
    model.eta = pyo.Var(model.dp, model.k, initialize=1.1, within=pyo.PositiveReals, name='unifac eta_k')
    model.gamma_c = pyo.Var(model.dp, model.i, initialize=1.5, within=pyo.Reals, name='unifac combinatorial gamma')
    model.gamma_r = pyo.Var(model.dp, model.i, initialize=1.5, within=pyo.Reals, name='unifac residual gamma')
    model.gamma = pyo.Var(model.dp, model.i, initialize=1.5, within=pyo.Reals, name='unifac total gamma')
    
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
        Surface area fraction parameter (Equation A.10):
        L_{i,t} = q_i / Σ x_τ q_τ
        """
        return m.L[j, i] == m.q[i] / sum(m.q[ii] * m.x[j, ii] for ii in m.i)
    
    model.eq_L = pyo.Constraint(model.dp, model.i, rule=eq_L, name='Constraint for calculating L_i of unifac')
    
    def eq_G(m, k, i):
        """
        Group surface area contribution:
        G_{k,i} = ν_{i,k} Q_k
        """
        return m.G[k, i] == m.v[i, k] * m.Qk[k]
    
    model.eq_G = pyo.Constraint(model.k, model.i, rule=eq_G, name='Constraint for calculating G_k,i of unifac')
    
    def eq_theta(m, dp, k):
        """
        Group surface area fraction (Equation A.14):
        θ_{k,t} = Σ x_i ν_{i,k} Q_k / Σ x_i q_i
        """
        return m.theta[dp, k] == sum(m.G[k, i] * m.x[dp, i] for i in m.i) / sum(m.x[dp, i] * m.q[i] for i in m.i)
    
    model.eq_theta = pyo.Constraint(model.dp, model.k, rule=eq_theta,
                                    name='Constraint for calculating theta_k of unifac')
    
    def eq_tau(m, dp, k, l):
        """
        Group interaction parameter (Equation A.16):
        τ_{k,l,t} = exp(-a_{k,l} / T_t)
        """
        return m.tau[dp, k, l] == pyo.exp(-m.a[k, l] / m.T[dp])
    
    model.eq_tau = pyo.Constraint(model.dp, model.k, model.k, rule=eq_tau,
                                  name='Constraint for calculating tau_k,m of unifac')
    
    def eq_s(m, dp, k, i):
        """
        Beta parameter numerator (Equation A.13 modified):
        s_{k,i,t} = Σ ν_{i,l} Q_l τ_{l,k,t}
        """
        return m.s[dp, k, i] == sum(m.G[l, i] * m.tau[dp, l, k] for l in m.k)
    
    model.eq_s = pyo.Constraint(model.dp, model.k, model.i, rule=eq_s,
                                name='Constraint for calculating s_k,i of unifac')
    
    def eq_eta(m, dp, k):
        """
        Omega parameter (Equation A.15 modified):
        η_{k,t} = Σ x_i s_{k,i,t}
        """
        return m.eta[dp, k] == sum(m.s[dp, k, i] * m.x[dp, i] for i in m.i)
    
    model.eq_eta = pyo.Constraint(model.dp, model.k, rule=eq_eta,
                                  name='Constraint for calculating eta_k of unifac')
    
    def eq_gamma_c(m, dp, i):
        """
        Combinatorial activity coefficient (Equation A.2):
        ln γ^C_{i,t} = 1 - J_{i,t} + ln J_{i,t} - 5q_i(1 - J_{i,t}/L_{i,t} + ln J_{i,t}/L_{i,t})
        """
        return m.gamma_c[dp, i] == 1 - m.J[dp, i] + pyo.log(m.J[dp, i]) - 5 * m.q[i] * (
                1 - m.J[dp, i] / m.L[dp, i] + pyo.log(m.J[dp, i] / m.L[dp, i]))
    
    model.eq_gamma_c = pyo.Constraint(model.dp, model.i, rule=eq_gamma_c,
                                      name='Constraint for calculating gammaC_i of unifac')
    
    def eq_gamma_r(m, dp, i):
        """
        Residual activity coefficient (Equation A.11 modified):
        ln γ^R_{i,t} = q_i[1 - ln(Σ θ_{k,t} s_{k,i,t} / η_{k,t}) - Σ(θ_{k,t} s_{k,i,t} / η_{k,t} - G_{k,i} ln(s_{k,i,t} / η_{k,t}))]
        """
        return m.gamma_r[dp, i] == m.q[i] * (1 - pyo.log(sum(m.theta[dp, k] * m.s[dp, k, i] / m.eta[dp, k] for k in m.k))) - sum(
            m.theta[dp, k] * m.s[dp, k, i] / m.eta[dp, k] - m.G[k, i] * pyo.log(m.s[dp, k, i] / m.eta[dp, k]) for k in m.k)
    
    model.eq_gamma_r = pyo.Constraint(model.dp, model.i, rule=eq_gamma_r,
                                      name='Constraint for calculating gammaR_i of unifac')
    
    def eq_gamma(m, dp, i):
        """
        Total activity coefficient (Equation A.1):
        γ_{i,t} = exp(ln γ^C_{i,t} + ln γ^R_{i,t})
        """
        return m.gamma[dp, i] == pyo.exp(m.gamma_r[dp, i] + m.gamma_c[dp, i])
    
    model.eq_gamma = pyo.Constraint(model.dp, model.i, rule=eq_gamma, name='Constraint for calculating gamma')
    
    # define the solver
    opt = pyo.SolverFactory('ipopt')
    # define solver options
    opt.options["halt_on_ampl_error"] = "yes"
    opt.options["max_iter"] = 10000
    opt.options["tol"] = 1e-8
    # solve the optimization problem
    solution = opt.solve(model, tee=True, symbolic_solver_labels=True)
    
    return solution, model


if __name__ == "__main__":
    print("=== Pyomo UNIFAC Implementation ===")
    print("Use pyo_unifac() function for UNIFAC calculations in optimization frameworks.")