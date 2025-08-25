# Import from the src package
import numpy as np
from src import unifac_wrapper, load_unifac_data
from src.property_calculator import calculate_properties
from src.antoine_calculator import calculate_antoine_parameters

# Load data once
rk_data, qk_data, interaction_data =load_unifac_data()

# Define your mixture
mixture = {
    'A': {'CH3': 1, 'CH3CO': 1}, # acetone
    'B': {'CH3OH': 1}, # methanol
    }

# calculate Tb of the two compounds
props_A = calculate_properties(mixture['A'])
props_B = calculate_properties(mixture['B'])
# determine the Antoine parameters
antoine_params_A = calculate_antoine_parameters(props_A)
antoine_params_B = calculate_antoine_parameters(props_B)

# define the number of datapoints
n_data = 10
# define the liquid compositions x
x = np.stack((np.linspace(0, 1, n_data), 1 - np.linspace(0, 1, n_data)), axis=-1)
# create initial guess of the temperature
T = np.linspace(props_A['tb'], props_B['tb'], n_data)

# Create flattened interaction data after loading JSON
# interaction_params = {}
# for group_i, group_dict in interaction_data.items():
#     for group_j, value in group_dict.items():
#         interaction_params[(group_i, group_j)] = value

# # as a dictionary comprehension:
# e_dict = {(i, group): e[i, k]
#         for i in range(e.shape[0])
#         for k, group in enumerate(groups)}

# theta_dict = {(i, group): theta[i,k]
#             for i in range(len(x))
#             for k, group in enumerate(groups)}

# psi_dict = {(t, group_i, group_j): psi[i, j]
#             for t in range(1)
#             for i, group_i in enumerate(groups)
#             for j, group_j in enumerate(groups)}

# omega_dict ={(t, group): omega[t, k]
#               for t in range(1)
#               for i, group in enumerate(groups)}
# #%%
# # create the model
# model = pyo.ConcreteModel()
# n_comp = x.shape[1]
# n_data = x.shape[0]

# # define the sets
# model.k = pyo.Set(initialize=groups, name='groups')
# model.i = pyo.RangeSet(0, n_comp - 1, name=' no. of Compounds')
# model.dp = pyo.RangeSet(0, n_data - 1, name='no. of data points')
# model.grp = pyo.Set(initialize=interaction_data.keys(), name='all groups')
# # define data
# model.Rk = pyo.Param(model.grp, initialize=rk_data, mutable=False, name='unifac Rk parameters')
# model.Qk = pyo.Param(model.grp, initialize=qk_data, mutable=False, name='unifac Qk parameters')
# model.a = pyo.Param(model.grp, model.grp, initialize=interaction_params, mutable=False,
#                     name='unifac interaction a_mk', domain=pyo.Reals)
# model.T = pyo.Param(model.dp, initialize=T, mutable=False, name='Temperature in K')
# model.x = pyo.Param(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(x)}, mutable=False,
#                     name='Liquid composition x', domain=pyo.Reals)
# # define constants: these variable do not change w.r.t. x and T
# model.r = pyo.Param(model.i, initialize={index: v for index, v in np.ndenumerate(r)}, mutable=False, name='unifac r_i')
# model.q = pyo.Param(model.i, initialize={index: v for index, v in np.ndenumerate(q)}, mutable=False, name='unifac q_i')
# model.e = pyo.Param(model.i, model.k, initialize=e_dict, mutable=False, name='unifac e_ik')

# # define variable, change with x but not T
# model.J = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(J)}, within=pyo.PositiveReals, name='unifac J_i')
# model.L = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(L)}, within=pyo.PositiveReals, name='unifac L_i')
# model.gamma_c = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(gamma_c)}, within=pyo.Reals, name='unifac gamma combinatorial')
# model.theta = pyo.Var(model.dp, model.k, initialize=theta_dict, within=pyo.Reals, name='unifac theta_k')

# # define variables that are dependent on x and T
# model.psi = pyo.Var(model.dp, model.k, model.k, initialize=psi_dict, within=pyo.PositiveReals, name='unifac psi_k')
# model.omega = pyo.Var(model.dp, model.k, initialize=omega_dict, within=pyo.PositiveReals, name='unifac omega_k')
# model.gamma_r = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(np.matrix(gamma_r))}, within=pyo.Reals, name='unifac gamma residual')
# model.gamma = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(gamma)}, within=pyo.PositiveReals, name='unifac gamma total')

# # define the equations
# def eq_J():
#     return 

# def eq_L():
#     return m.L[dp, i] == 

# def eq_gamma_c():
#     return

# def eq_theta():
#     return

# def eq_psi():
#     return

# def eq_omega():
#     return

# def eq_gamma_r():
#     return

# def eq_gamma(m, dp, i):
#     return m.gamma[dp, i] == pyo.exp(m.gamma_r[dp, i] + m.gamma_c[dp, i])

# %%
