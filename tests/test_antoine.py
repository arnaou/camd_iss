# Project: camd_iss
#
# Title: testing the procedure for determining the antoine parameters
#
# Author: Adem Rosenkvist Nielsen Aouichaoui
#
# Email: arnaou@kt.dtu.dk
#######################################################################################################################

# import paths
from tests import _PATH_DATA

# import python native modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pytest



# import costume modules
from utils.gc import load_gc, prop_calc, inspect_contribs, solve_antoine, psat_validation
from utils.pyom import ensure_file_exists_in_cwd, parse_pyomo_log

# load the data
df_gc, _ = load_gc(_PATH_DATA + '/data_unifac.xlsx', 'dataframe')
# define the components: water, DMSO, methanol, ethylenglycol, heptane, diethylamine
df_comps = pd.DataFrame({
    'H2O':   [1, 0, 0, 0, 0, 0],
    'DMSO':  [0, 1, 0, 0, 0, 0],
    'CH3OH': [0, 0, 1, 0, 0, 0],
    'DOH':   [0, 0, 0, 1, 0, 0],
    'CH3':   [0, 0, 0, 0, 2, 2],
    'CH2':   [0, 0, 0, 0, 5, 1],
    'CH2NH': [0, 0, 0, 0, 0, 1]})

df_comps.index = ['H2O', 'DMSO', 'MeOH', 'DOH', 'HEP', 'DEA']  # heptane: CCCCCCC, diethylamine: CCNCC, water: H2O, ethylene glycol: DOH

# define property list
prop_list = ['Tc', 'Pc', 'Tb', 'Ac']
# inspect missing contributions
grp_list = df_comps.columns
print(inspect_contribs(df_gc.loc[grp_list,prop_list]))

# calculate the properties
df_props = prop_calc(df_comps, prop_list, df_gc)
# convert bar to MPa
df_props.loc[:, 'Pc'] = df_props['Pc'] / 10  # convert to MPa


# #%% primary solution: generating a good initial guess
#
#
# # print('------------- Finding the saturated pressure at at Tc*0.7 ----------')
# # Pw_pyo = []
# # Pw_clc = []
# # A0_pyo = []
# # B0_pyo = []
# # C0_pyo = []
# # A0_clc = []
# # B0_clc = []
# # C0_clc = []
# # A_pyo = []
# # B_pyo = []
# # C_pyo = []
# A_fun = []
# B_fun = []
# C_fun = []
# are_P0 = []
# are_Pc = []
# are_w = []
# for i in df_space.index:
#     # """
#     # Step 1: find the saturated pressure at the acentric factor e.i. T=Tc*0.7
#     # """
#     # extract the data
#     P0 = 0.101325
#     Pc = df_props.loc[i, 'Pc']
#     Tc = df_props.loc[i, 'Tc']
#     Tb = df_props.loc[i, 'Tb']
#     w = df_props.loc[i, 'Ac']
#     # solve for antoine parameters
#     output = solve_antoine(Tb, Tc, Pc, P0, w, True)
#     A_fun.append(output['A'])
#     B_fun.append(output['B'])
#     C_fun.append(output['C'])
#     # validate antoine parameters (A, B, C, T, Tb, Tc, Pc, w, name)
#     output = psat_validation(output['A'], output['B'], output['C'], T=None, Tb=Tb, Tc=Tc, Pc=Pc, w=w, name=i)
#
#     output['fig'].show()
#
#     are_P0.append(output['are_P0'])
#     are_Pc.append(output['are_Pc'])
#     are_w.append(output['are_w'])
#
#     # print(f'ARE of P0: {are_P0}')
#     # print(f'ARE of Pc: {are_Pc}')
#     # print(f'ARE of w: {are_w}')
#
#
# df_are = pd.DataFrame({'comps': df_props.index,'are_P0': are_P0, 'are_Pc': are_Pc, 'are_w': are_w})
# print(df_are)
#     # # calculate the saturated vapor pressure for Tr = 0.7
#     # Pw = np.power(10, -(w + 1)) * Pc












# #%% Pyomo implementation
#     # # define the model type
#     # model = pyo.ConcreteModel('Saturated pressure corresponding to the acentric factor')
#     # # define the parameters
#     # model.pc = pyo.Param(initialize=Pc, mutable=False, name='the critical pressure in MPa', domain=pyo.PositiveReals)
#     # model.w = pyo.Param(initialize=w, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
#     # # define the variable
#     # model.pr = pyo.Var(within=pyo.PositiveReals, initialize=0.5 * Pc, name='reduced pressure at 0.7 Tc in MPa')
#     #
#     #
#     # def eq_w(m):
#     #     """ Constraint for the equality of the acentric factor
#     #
#     #     :param m:
#     #     :return:
#     #     """
#     #     return m.w == - pyo.log10(m.pr / m.pc) - 1.00
#     #
#     #
#     # model.eq_w = pyo.Constraint(rule=eq_w, name='calculation rule for acentric factor')
#     #
#     # # define the solver
#     # opt = pyo.SolverFactory('ipopt')
#     # # define solver options
#     # opt.options["halt_on_ampl_error"] = "yes"
#     # opt.options["linear_solver"] = "MA27"
#     # opt.options["max_iter"] = 10000
#     # # ensurepyo_antoine_log exists
#     # ensure_file_exists_in_cwd("pyo_antoine_log", "")
#     # # solve the optimization problem
#     # solution = opt.solve(model, tee=True, symbolic_solver_labels=True, logfile="pyo_antoine_log")
#     # # extract solution and solver stats
#     # solv_stat = str(solution.solver.status)
#     # term_cond = str(solution.solver.termination_condition)
#     # Pw_pyo.append(model.pr.value)
#  #%%
#     # """
#     # Step 2: find the initial guess for the Antoine parameters using selected-points method
#     # """
#     # # extract the points of interest
#     # t1 = Tb
#     # t2 = 0.7 * Tc
#     # t3 = Tc
#     # p1 = P0
#     # p2 = Pw
#     # p3 = Pc
#     # y1 = np.log10(p1)
#     # y2 = np.log10(p2)
#     # y3 = np.log10(p3)
#     #
#     # # equation for C
#     # X = (y3 - y2) / (y2 - y1) * (t2 - t1) / (t3 - t2)
#     # Y = t1 - t3
#     # C0 = Y / (X - 1) - t3
#     # C0_clc.append(C0)
#     #
#     # # equation for B
#     # B0=(y3 - y1) / (t3 - t1) * (t1 + C0) * (t3 - C0)
#     # B0_clc.append(B0)
#     #
#     # # equation for A
#     # A0 = y2 + B0 / (t2 + C0)
#     # A0_clc.append(A0)
# #%% pyomo implementation
#     # # define the model type
#     # model = pyo.ConcreteModel('System for generating initial guess for the antoine parameters')
#     # # define the parameters
#     # model.t1 = pyo.Param(initialize=t1, mutable=False, name='t1 or Tb in K', domain=pyo.PositiveReals)
#     # model.t2 = pyo.Param(initialize=t2, mutable=False, name='t2 or Tr in K', domain=pyo.PositiveReals)
#     # model.t3 = pyo.Param(initialize=t3, mutable=False, name='t3 or Tc in K', domain=pyo.PositiveReals)
#     # model.y1 = pyo.Param(initialize=np.log10(p1), mutable=False, name='y1 or log10(Patm) in MPa', domain=pyo.Reals)
#     # model.y2 = pyo.Param(initialize=np.log10(p2), mutable=False, name='y2 or log10(Pr) in MPa', domain=pyo.Reals)
#     # model.y3 = pyo.Param(initialize=np.log10(p3), mutable=False, name='y3 or log10(Pc) in MPa', domain=pyo.Reals)
#     # # define the variable
#     # model.A = pyo.Var(within=pyo.Reals, initialize=10, name='Antoine Parameter A')
#     # model.B = pyo.Var(within=pyo.Reals, initialize=0, name='Antoine Parameter B')
#     # model.C = pyo.Var(within=pyo.Reals, initialize=0, name='Antoine Parameter C')
#     #
#     #
#     # # define constraints
#     #
#     # def eq_C(m):
#     #     """ Constraint for the equality of the C parameter for Antoine equation
#     #
#     #     :param m:
#     #     :return:
#     #     """
#     #     X = (m.y3 - m.y2) / (m.y2 - m.y1) * (m.t2 - m.t1) / (m.t3 - m.t2)
#     #     Y = m.t1 - m.t3
#     #     return m.C == Y / (X - 1) - m.t3
#     #
#     #
#     # model.eq_C = pyo.Constraint(rule=eq_C, name='calculation rule for Antoine C')
#     #
#     #
#     # def eq_B(m):
#     #     """ Constraint for the equality of the B parameter for Antoine equation
#     #
#     #     :param m:
#     #     :return:
#     #     """
#     #     return m.B == (m.y3 - m.y1) / (m.t3 - m.t1) * (m.t1 + m.C) * (m.t3 - m.C)
#     #
#     #
#     # model.eq_B = pyo.Constraint(rule=eq_B, name='calculation rule for Antoine B')
#     #
#     #
#     # def eq_A(m):
#     #     """ Constraint for the equality of the A parameter for Antoine equation
#     #
#     #     :param m:
#     #     :return:
#     #     """
#     #     return m.A == m.y2 + m.B / (m.t2 + m.C)
#     #
#     #
#     # model.eq_A = pyo.Constraint(rule=eq_A, name='calculation rule for Antoine A')
#     #
#     # # define the solver
#     # opt = pyo.SolverFactory('ipopt')
#     # # define solver options
#     # opt.options["halt_on_ampl_error"] = "yes"
#     # opt.options["linear_solver"] = "MA27"
#     # opt.options["max_iter"] = 10000
#     # # solve the optimization problem
#     # solution = opt.solve(model, tee=True, symbolic_solver_labels=True, logfile="pyo_vle_log")
#     # # extract solution and solver stats
#     # solv_stat = str(solution.solver.status)
#     # term_cond = str(solution.solver.termination_condition)
#     # A = model.A.value
#     # B = model.B.value
#     # C = model.C.value
#     # A0_pyo.append(A)
#     # B0_pyo.append(B)
#     # C0_pyo.append(C)
#     # if solv_stat != 'ok' or term_cond != 'optimal':
#     #     print('Something is wrong with compound while solving for Pr ')
#
# #%% final estimation
#     # """
#     # Step 3: find the final value for the Antoine parameters
#     # """
#     # # print('------------- Finding the final values for the Antoine parameters ----------')
#     #
#     # # define the model type
#     # model = pyo.ConcreteModel('system of equation for solving the Antoine parameters')
#     # # define the parameters
#     # model.tb = pyo.Param(initialize=Tb, mutable=False, name='normal boiling point in K', domain=pyo.PositiveReals)
#     # model.tc = pyo.Param(initialize=Tc, mutable=False, name='the critical temperature in K', domain=pyo.PositiveReals)
#     # model.pc = pyo.Param(initialize=Pc, mutable=False, name='the critical pressure in MPa', domain=pyo.PositiveReals)
#     # model.w = pyo.Param(initialize=w, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
#     # model.p0 = pyo.Param(initialize=P0, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
#     # # define the model variables
#     # model.A = pyo.Var(within=pyo.Reals, initialize=A0, name='Antoine parameter A')
#     # model.B = pyo.Var(within=pyo.Reals, initialize=B0, name='Antoine parameter B')
#     # model.C = pyo.Var(within=pyo.Reals, initialize=C0, name='Antoine parameter C')
#     # model.pr = pyo.Var(within=pyo.PositiveReals, initialize=10 ** (A0 - B0 / (C0 + Tc * 0.70)), name='Antoine parameter C')
#     #
#     # # define the constraints
#     # def eq_p0(m):
#     #     """ Constraint for the equality of the atmospheric pressure
#     #
#     #     :param m:
#     #     :return:
#     #     """
#     #     return pyo.log10(m.p0) == m.A - m.B / (m.C + m.tb)
#     #
#     # model.eq_p0 = pyo.Constraint(rule=eq_p0, name='calculation rule for p atmospheric')
#     #
#     # def eq_pc(m):
#     #     """ Constraint for the equality of the critical pressure
#     #
#     #     :param m:
#     #     :return:
#     #     """
#     #     return pyo.log10(m.pc) == m.A - m.B / (m.C + m.tc)
#     #
#     # model.eq_pc = pyo.Constraint(rule=eq_pc, name='calculation rule for p critical')
#     #
#     # def eq_w(m):
#     #     """ Constraint for the equality of the acentric factor
#     #
#     #     :param m:
#     #     :return:
#     #     """
#     #     pr = 10 ** (m.A - m.B / (m.C + m.tc * 0.70))
#     #     return m.w == - pyo.log10(pr / m.pc) - 1.00
#     #
#     # model.eq_w = pyo.Constraint(rule=eq_w, name='calculation rule for acentric factor')
#     # # define the solver
#     # opt = pyo.SolverFactory('ipopt')
#     # # show more stuff
#     # opt.options["halt_on_ampl_error"] = "yes"
#     # opt.options["linear_solver"] = "MA27"
#     # opt.options["max_iter"] = 10000
#     #
#     # # solve the optimization problem
#     # solution = opt.solve(model, tee=True, symbolic_solver_labels=True, logfile="pyomo_antoine_log")
#     #
#     # solv_stat = str(solution.solver.status)
#     # term_cond = str(solution.solver.termination_condition)
#     # if solv_stat != 'ok' or term_cond != 'optimal':
#     #     print('Something is wrong with compound while solving for the Antoine parameters')
#     #
#     # output = {'solv_stat': str(solution.solver.status),
#     #           'term_cond': str(solution.solver.termination_condition),
#     #           'num_iter': parse_pyomo_log('pyomo_antoine_log'),
#     #           'A': model.A.value,
#     #           'B': model.B.value,
#     #           'C': model.C.value}
#     # A_pyo.append(output['A'])
#     # B_pyo.append(output['B'])
#     # C_pyo.append(output['C'])
#
# #%% we implement this in a function
#
#
# # def solve_antoine(Tb: float, Tc: float, Pc: float, P0: float, w: float, verbose: bool) -> dict:
# #     """ Function that solves for the A, B and C parameters fom the antoine equation using pyomo
# #
# #     :param Tb: the melting point in Kelvin
# #     :param Tc: the critical temperature in Kelvin
# #     :param Pc: the critical pressure in MPa
# #     :param P0: the atmospheric pressure in MPa
# #     :param w: the acentric factor
# #     :param verbose: printing of solver progress
# #
# #     :return: a dict containing: solver_stat: status of the solver
# #                                 term_cond: termination condition of the solver
# #                                 num_iter: number of iterations
# #                                 obj_val: the objective values
# #                                 A: Antoine parameter A
# #                                 B: Antoine parameter B
# #                                 C: Antoine parameter C
# #     NB: the Antoine produces saturated pressures using the equation: P(MPa)=10 ** (A - B / (C + T))
# #     """
# #
# #     # initializations:
# #
# #     """
# #         Step 1: find the the saturated pressure at the acentric factor e.i. T=Tc*0.7
# #     """
# #     # calculate the saturated vapor pressure for Tr = 0.7
# #     Pw = np.power(10, -(w + 1)) * Pc
# #
# #     """
# #      Step 2: find the initial guess for the Antoine parameters using selected-points method
# #     """
# #     # extract the points of interest
# #     t1 = Tb
# #     t2 = 0.7 * Tc
# #     t3 = Tc
# #     p1 = P0
# #     p2 = Pw
# #     p3 = Pc
# #     y1 = np.log10(p1)
# #     y2 = np.log10(p2)
# #     y3 = np.log10(p3)
# #
# #     # equation for C
# #     X = (y3 - y2) / (y2 - y1) * (t2 - t1) / (t3 - t2)
# #     Y = t1 - t3
# #     C0 = Y / (X - 1) - t3
# #
# #     # equation for B
# #     B0 = (y3 - y1) / (t3 - t1) * (t1 + C0) * (t3 - C0)
# #
# #     # equation for A
# #     A0 = y2 + B0 / (t2 + C0)
# #
# #     """
# #     Step 3: find the final value for the Antoine parameters
# #     """
# #     # print('------------- Finding the final values for the Antoine parameters ----------')
# #
# #     # define the model type
# #     model = pyo.ConcreteModel('system of equation for solving the Antoine parameters')
# #     # define the parameters
# #     model.tb = pyo.Param(initialize=Tb, mutable=False, name='normal boiling point in K', domain=pyo.PositiveReals)
# #     model.tc = pyo.Param(initialize=Tc, mutable=False, name='the critical temperature in K', domain=pyo.PositiveReals)
# #     model.pc = pyo.Param(initialize=Pc, mutable=False, name='the critical pressure in MPa', domain=pyo.PositiveReals)
# #     model.w = pyo.Param(initialize=w, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
# #     model.p0 = pyo.Param(initialize=P0, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
# #     model.pw = pyo.Param(initialize=Pw, mutable=False, name='pressure at Tr=0.7', domain=pyo.PositiveReals)
# #     # define the model variables
# #     model.A = pyo.Var(within=pyo.Reals, initialize=A0, name='Antoine parameter A')
# #     model.B = pyo.Var(within=pyo.Reals, initialize=B0, name='Antoine parameter B')
# #     model.C = pyo.Var(within=pyo.Reals, initialize=C0, name='Antoine parameter C')
# #     #model.pr = pyo.Var(within=pyo.PositiveReals, initialize=10 ** (A0 - B0 / (C0 + Tc * 0.70)), name='Antoine parameter C')
# #
# #     # define the constraints
# #     def eq_p0(m):
# #         """ Constraint for the equality of the atmospheric pressure
# #
# #         :param m:
# #         :return:
# #         """
# #         return pyo.log10(m.p0) == m.A - m.B / (m.C + m.tb)
# #
# #     model.eq_p0 = pyo.Constraint(rule=eq_p0, name='calculation rule for p atmospheric')
# #
# #     def eq_pc(m):
# #         """ Constraint for the equality of the critical pressure
# #
# #         :param m:
# #         :return:
# #         """
# #         return pyo.log10(m.pc) == m.A - m.B / (m.C + m.tc)
# #
# #     model.eq_pc = pyo.Constraint(rule=eq_pc, name='calculation rule for p critical')
# #
# #     # def eq_w(m):
# #     #     """ Constraint for the equality of the acentric factor
# #     #
# #     #     :param m:
# #     #     :return:
# #     #     """
# #     #     pr = 10 ** (m.A - m.B / (m.C + m.tc * 0.70))
# #     #     return m.w == - pyo.log10(pr / m.pc) - 1.00
# #     #model.eq_w = pyo.Constraint(rule=eq_w, name='calculation rule for acentric factor')
# #
# #     def eq_pw(m):
# #         """
# #
# #         :param m:
# #         :return:
# #         """
# #
# #         return pyo.log10(m.pw) == m.A - m.B / (m.C + m.tc*0.7)
# #
# #     model.eq_pw = pyo.Constraint(rule=eq_pw, name='calculation rule for p at Tr=0.7')
# #
# #     # define the solver
# #     opt = pyo.SolverFactory('ipopt')
# #     # show more stuff
# #     opt.options["halt_on_ampl_error"] = "yes"
# #     opt.options["linear_solver"] = "MA27"
# #     opt.options["max_iter"] = 10000
# #
# #     # solve the optimization problem
# #     solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True, logfile="pyomo_antoine_log")
# #
# #     solv_stat = str(solution.solver.status)
# #     term_cond = str(solution.solver.termination_condition)
# #     if solv_stat != 'ok' or term_cond != 'optimal':
# #         print('Something is wrong with compound while solving for the Antoine parameters')
# #
# #     output = {'solv_stat': str(solution.solver.status),
# #               'term_cond': str(solution.solver.termination_condition),
# #               'num_iter': parse_pyomo_log('pyomo_antoine_log'),
# #               'A0': A0,
# #               'B0': B0,
# #               'C0': C0,
# #               'A': model.A.value,
# #               'B': model.B.value,
# #               'C': model.C.value}
# #
# #     return output
#
# # for i in df_space.index:
# #     # extract the data
# #     P0 = 0.101325
# #     Pc = df_props.loc[i, 'Pc']
# #     Tc = df_props.loc[i, 'Tc']
# #     Tb = df_props.loc[i, 'Tb']
# #     w = df_props.loc[i, 'Ac']
# #
# #     output = solve_antoine(Tb, Tc, Pc, P0, w, True)
# #     A_fun.append(output['A'])
# #     B_fun.append(output['B'])
# #     C_fun.append(output['C'])
# #
#
#
# # print(A_pyo)
# # print(A_fun)
# # print(B_pyo)
# # print(B_fun)
# # print(C_pyo)
# # print(C_fun)
# # all is good w.r.t. the algorithm, now for validation with plot. and calculations
# #%%
# # def antoine_correlation(A,B,C,T):
# #     """
# #     Function for the antoine correlation:
# #      log10(P*) = A-B/(C*T)
# #      the output is already made into the saturated pressure e.i.
# #      P* = 10^(A-B/(C*T))
# #
# #     :param A:
# #     :param B:
# #     :param C:
# #     :param T:
# #     :return:
# #     """
# #     return np.power(10, A-B/(T+C))
# #
# # def pitzer_acentric_factor(A, B, C, Tc, Pc):
# #     """
# #     function to calculate the Pitzer acentric factor
# #     :param A:
# #     :param B:
# #     :param C:
# #     :param Tc:
# #     :param Pc:
# #     :return:
# #     """
# #     #
# #     Tr = 0.7
# #     T = Tr * Tc
# #     P_sat = antoine_correlation(A,B,C,T)
# #     w = -np.log10(P_sat/Pc) - 1
# #     return w
# #
# #
# # def psat_validation(A, B, C, T, Tb, Tc, Pc, w, name):
# #     """
# #     Validates and plots saturation pressure (Psat) over a range of temperatures
# #     using Antoine's equation.
# #
# #     :param A: Antoine constant A
# #     :param B: Antoine constant B
# #     :param C: Antoine constant C
# #     :param T: Temperature array (if None, generates a range between 0.8*Tb to 1.2*Tc)
# #     :param Tb: Boiling temperature (K)
# #     :param Tc: Critical temperature (K)
# #     :param Pc: Critical pressure (bar)
# #     :param w: Acentric factor for the component
# #     :param name: Name of the component (for plot title)
# #     :return: Dictionary containing the figure, saturation pressure array, temperature array,
# #              and percentage errors for P0, Pc, and acentric factor w.
# #     """
# #
# #     # Calculate the saturation pressure at the boiling point (Tb) using Antoine's equation.
# #     P0_clc = antoine_correlation(A, B, C, Tb)
# #     # Calculate the Absolute Relative Error (ARE) between calculated and expected boiling pressure (0.101325 bar).
# #     are_P0 = (abs(0.101325 - P0_clc)) / 0.101325 * 100
# #
# #     # Calculate the saturation pressure at the critical point (Tc) using Antoine's equation.
# #     Pc_clc = antoine_correlation(A, B, C, Tc)
# #     # Calculate the Absolute Relative Error (ARE) between calculated and provided critical pressure (Pc).
# #     are_Pc = (abs(Pc - Pc_clc)) / Pc * 100
# #
# #     # Calculate the acentric factor using the Pitzer method (assuming pitzer_acentric_factor is implemented).
# #     w_clc = pitzer_acentric_factor(A, B, C, Tc, Pc)
# #     # Calculate the Absolute Relative Error (ARE) between calculated and provided acentric factor (w).
# #     are_w = (abs(w - w_clc)) / w * 100
# #
# #     # If the temperature array `T` is not provided, generate a default range from 0.8*Tb to 1.2*Tc.
# #     if T is None:
# #         T = np.arange(Tb * 0.8, Tc * 1.2, 5)
# #
# #     # Compute the saturation pressure for the temperature array `T` using Antoine's equation.
# #     psat = antoine_correlation(A, B, C, T)
# #
# #     # Calculate the saturation pressure at Tr = 0.7*Tc, an important reference point in the graph.
# #     Pw = antoine_correlation(A, B, C, 0.7 * Tc)
# #
# #     # Create a new figure for plotting the saturation pressure curve.
# #     fig = plt.figure()
# #
# #     # Plot the saturation pressure `psat` as a function of temperature `T`.
# #     plt.plot(T, psat, label='P$^{sat}$')
# #     plt.xlabel('Temperature (K)')  # Label the x-axis.
# #     plt.ylabel('Saturation Pressure (bar)')  # Label the y-axis.
# #     plt.title(f'Saturation Pressure vs Temperature for {name}')  # Set the title of the plot.
# #
# #     # Plot important points on the graph:
# #     # 1. Boiling point (Tb) with saturation pressure calculated from Antoine equation.
# #     plt.scatter(Tb, P0_clc, color='red', s=100, edgecolor='black', zorder=5, label='Tb: P0')
# #     # 2. The point at Tr = 0.7*Tc (Pw is the pressure at this reduced temperature).
# #     plt.scatter(0.7 * Tc, Pw, color='red', s=100, edgecolor='black', zorder=5, label='Tr=0.7')
# #     # 3. Critical point (Tc, Pc).
# #     plt.scatter(Tc, Pc, color='red', s=100, edgecolor='black', zorder=5, label='Tc: Pc')
# #
# #     # Add a legend to label the important points.
# #     plt.legend()
# #
# #     # Prepare the output dictionary:
# #     # - 'fig' contains the plotted figure.
# #     # - 'T' contains the temperature array.
# #     # - 'psat' contains the saturation pressure array.
# #     # - 'are_P0', 'are_Pc', 'are_w' contain the rounded percentage errors for P0, Pc, and w respectively.
# #     output = {
# #         'fig': fig,
# #         'T': T,
# #         'psat': psat,
# #         'are_P0': are_P0.round(),  # Rounded ARE for boiling point.
# #         'are_Pc': are_Pc.round(),  # Rounded ARE for critical point.
# #         'are_w': are_w.round()  # Rounded ARE for acentric factor.
# #     }
# #
# #     # Return the output dictionary.
# #     return output
#
#
#
#
#
#
# #%%
# # j = 0
# # for i in df_space.index:
# #     # extract the data
# #     P0 = 0.101325
# #     Pc = df_props.loc[i, 'Pc']
# #     Tc = df_props.loc[i, 'Tc']
# #     Tb = df_props.loc[i, 'Tb']
# #     w = df_props.loc[i, 'Ac']
# #     A = A_fun[j]
# #     B = B_fun[j]
# #     C = C_fun[j]
# #     # T = np.arange(Tb*0.9, Tc*1.1, 5)
# #     # #T = np.arange(Tb , Tc, 5)
# #     # #T = np.arange(Tb-8, Tc+8, 5)
# #     # print(P0)
# #     # print(antoine_correlation(A_fun[j],B_fun[j],C_fun[j],Tb))
# #     # print(Pc)
# #     # print(antoine_correlation(A_fun[j], B_fun[j], C_fun[j], Tc))
# #     # print(w)
# #     # print(pitzer_acentric_factor(A_fun[j], B_fun[j], C_fun[j], Tc, Pc))
# #     # psat = antoine_correlation(A_fun[j], B_fun[j], C_fun[j], T)
# #     # Pw = antoine_correlation(A_fun[j], B_fun[j], C_fun[j], Tc*0.7)
# #     # plt.figure()
# #     # plt.plot(T, psat, label=f'Component {i}')
# #     # plt.xlabel('Temperature (K)')
# #     # plt.ylabel('Saturation Pressure (bar)')
# #     # plt.title(f'Saturation Pressure vs Temperature for Component {i}')
# #     # # Highlight the point with a red circle
# #     # plt.scatter(Tb, P0, color='red', s=100, edgecolor='black', zorder=5, label='boiling point')
# #     # plt.scatter(Tc*0.7, Pw, color='red', s=100, edgecolor='black', zorder=5, label='Tr=0.7')
# #     # plt.scatter(Tc, Pc, color='red', s=100, edgecolor='black', zorder=5, label='critical point')
# #     # plt.legend()
# #     output = psat_validation(A, B, C, T=None ,Tb=Tb, Tc=Tc, Pc=Pc, w=w,name=i)
# #
# #     # Show the plot (optional if running in an interactive environment like Jupyter)
# #     #plt.show()
# #     output['fig'].show()
# #     j += 1
# #
# #
# # print(output['are_P0'])
# # print(output['are_Pc'])
# # print(output['are_w'])
#
# # make function to validate the antoines, plot Tb, Tc, Pc, omega and P0 and show these in a plot.l