"""
Pyomo UNIFAC implementation for optimization and molecular design.

Following the established coding style with commented-out variables and 
the specific variable naming convention (tau, s, eta, G, etc.).
"""

import pyomo.environ as pyo
import numpy as np
from typing import Dict, List, Optional, Tuple
from .utils import convert_mixture_format as _convert_mixture_format
from .utils import convert_antoine_format as _convert_antoine_format


def pyo_azeo(x, T, mixture, rk_data, qk_data, interaction_data, antoine_params, verbose=False):
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
    # Auto-convert formats
    mixture = _convert_mixture_format(mixture)
    antoine_array = _convert_antoine_format(antoine_params)
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
    model.T = pyo.Var(model.dp, initialize={j: float(T[j]) for j in range(n_data)}, 
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
                        name='Liquid azeotropic composition x', domain=pyo.UnitInterval)
    
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
    
    # define the model variables - matching equations.py nomenclature exactly
    model.r = pyo.Var(model.i, initialize=1.2, within=pyo.PositiveReals, name='unifac r_i')
    model.q = pyo.Var(model.i, initialize=4.5, within=pyo.PositiveReals, name='unifac q_i')
    model.J = pyo.Var(model.dp, model.i, initialize=1.13, within=pyo.PositiveReals, name='unifac J_i')
    model.L = pyo.Var(model.dp, model.i, initialize=1.15, within=pyo.PositiveReals, name='unifac L_i')
    model.e = pyo.Var(model.i, model.k, initialize=0.5, within=pyo.NonNegativeReals, name='unifac e_ik')
    model.theta = pyo.Var(model.dp, model.k, initialize=1.2, within=pyo.PositiveReals, name='unifac theta_k')
    
    model.psi = pyo.Var(model.dp, model.k, model.k, initialize=0.79, within=pyo.PositiveReals, name='unifac psi_mk')
    model.beta = pyo.Var(model.dp, model.i, model.k, initialize=1.7, within=pyo.PositiveReals, name='unifac beta_ik')
    model.omega = pyo.Var(model.dp, model.k, initialize=1.1, within=pyo.PositiveReals, name='unifac omega_k')
    model.log_gamma_c = pyo.Var(model.dp, model.i, initialize=0.1, within=pyo.Reals, name='unifac log_gamma_c')
    model.log_gamma_r = pyo.Var(model.dp, model.i, initialize=0.1, within=pyo.Reals, name='unifac log_gamma_r')
    model.gamma = pyo.Var(model.dp, model.i, initialize=1.5, within=pyo.Reals, name='unifac total gamma')
    
    # VLE-specific variables
    model.psat = pyo.Var(model.dp, model.i, initialize=101325, within=pyo.PositiveReals, 
                        name='vapor pressure in Pa')
    model.y = pyo.Var(model.dp, model.i, initialize=0.5, within=pyo.UnitInterval, 
                     name='vapor composition y')
    
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

    def eq_sum_x(m, dp):
        """liquid composition normalization"""
        return sum(m.x[dp, i] for i in m.i) == 1.0
    model.eq_sum_x = pyo.Constraint(model.dp, rule=eq_sum_x, name='liquid composition normalization')

    def eq_azeotrope(m, dp):
        """Azeotrope condition: 1 = gamma_A/psat_A * gamma_B/psat_B"""
        return 1 == (m.gamma[dp, 0] / m.gamma[dp, 1]) * (m.psat[dp, 0] / m.psat[dp, 1])#(m.y[dp,0]/m.x[dp,0])/(m.y[dp,1]/m.x[dp,1])# (m.gamma[dp, 0] / m.gamma[dp, 1]) * (m.psat[dp, 0] / m.psat[dp, 1])
    model.eq_azeotrope = pyo.Constraint(model.dp, rule=eq_azeotrope, name='Azeotrope condition')

    # define the solver
    opt = pyo.SolverFactory('ipopt')
    # define solver options
    opt.options["halt_on_ampl_error"] = "yes"
    opt.options["max_iter"] = 10000
    opt.options["tol"] = 1e-8
    opt.options["acceptable_tol"] = 1e-6
    opt.options["mu_init"] = 1e-2
    opt.options["bound_push"] = 1e-8
    # solve the optimization problem
    solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True)
    
    return solution, model


if __name__ == "__main__":
    print("=== Pyomo VLE Implementation ===")
    print("Use pyo_vle() function for VLE calculations in optimization frameworks.")