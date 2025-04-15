########################################################################################################################
# Project: camd_iss
#
# Title: collection of tools for pure component property prediction using group-contribution methods
#
# Author: Adem Rosenkvist Nielsen Aouichaoui
#
# Email: arnaou@kt.dtu.dk
########################################################################################################################
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from utils.pyom import parse_pyomo_log, ensure_file_exists_in_cwd
import os
import warnings
import pickle
from collections import defaultdict
from matplotlib import pyplot as plt

def inspect_contribs(df, verbose=True):
    " function to expect contribution of selected groups and properties"
    # find zero entries
    zero_entries = df == 0

    # Step 2: Stack the DataFrame to convert it into a Series with multi-index
    zero_entries_stacked = zero_entries.stack()

    # Step 3: Filter where values are True (i.e., where the original values are 0)
    zero_tuples = zero_entries_stacked[zero_entries_stacked].index.tolist()

    # Step 4: Group by column
    grouped_by_column = defaultdict(list)
    for idx, col in zero_tuples:
        grouped_by_column[col].append(idx)
        # Step 5: Raise a warning if column is not "noc" or "ccsub"
        if col not in ["noc", "ccsub"] and verbose:
            warnings.warn(f"Warning: some groups have a 0 contribution for the property: Column '{col}' ")

    # Step 5: Convert defaultdict to regular dict for easy viewing (optional)
    grouped_by_column = dict(grouped_by_column)
    if bool(grouped_by_column):
        print('some properties and groups have missing interactions')
    else:
        print('all contribution for each group and property are present')


    return grouped_by_column



def inspect_interractions(df, verbose=True):
    """
    Inspects the interaction parameters in the given DataFrame. these are identified as the zero elements of groups not being from the same main group.
    A warning is raised in this case the the list of missing interraction is returned

    Parameters:
    df (DataFrame): The DataFrame containing interaction parameters an nxn shaped dataframe.

    Returns:
    missing_groups (list): A list of tuples containing the pairs of main groups with missing interactions.
    """

    # Initialize lists to store the locations of missing interaction parameters
    zero_locations = []
    missing_main_groups = []
    missing_sub_groups = []

    # Extract the groups from the DataFrame's index (rows)
    grp_list = list(df.index)

    # Load group mapping from a pickle file
    # unifac_dict will map each group to its main group for further comparisons
    try:
        with open('../data/unifac_groups.pkl', 'rb') as f:
            unifac_dict = pickle.load(f)
    except FileNotFoundError:
        with open('data/unifac_groups.pkl', 'rb') as f:
            unifac_dict = pickle.load(f)

    # Get the shape of the DataFrame (number of rows and columns)
    rows, cols = df.shape

    # Iterate through the DataFrame to find zeros above the diagonal
    for i in range(rows - 1):
        for j in range(i + 1, cols):
            # Check if the value is zero (indicating missing interaction)
            if df.iat[i, j] == 0: 
                # Retrieve the main groups for the groups at row `i` and column `j`
                main_g1 = next(value['main_g'] for value in unifac_dict.values() if value['grp'] == df.index[i])
                main_g2 = next(value['main_g'] for value in unifac_dict.values() if value['grp'] == df.columns[j])

                # If the main groups are different, log this missing interaction
                if main_g1 != main_g2:
                    missing_main_groups.append((main_g1, main_g2))  # Append the missing main group pair
                    zero_locations.append((i + 1, j + 1))  # Append the location with 1-based indexing
                    #sets_with_mg1 = [value['grp'] for key, value in unifac_dict.items() if value['main_g'] == main_g1]
                    #sets_with_mg2 = [value['grp'] for key, value in unifac_dict.items() if value['main_g'] == main_g2]
                    missing_sub_groups.append((df.columns[i], df.index[j]))

    # Calculate the total number of interaction parameters (upper triangular part of matrix)
    int_tot = rows * (cols - 1) // 2

    # If there are missing interaction parameters (zeros found above the diagonal)
    if len(zero_locations) != 0 and verbose:
        # Raise a warning indicating how many interaction parameters are missing
        warnings.warn(f"\nWarning, A total of {len(zero_locations)} out of {int_tot} interaction parameters are missing", UserWarning)
        # Raise a warning specifying which main groups have missing interaction parameters
        warnings.warn(f"\nWarning, the interaction between the following main groups {missing_main_groups} have been set to 0", UserWarning)
        #warnings.warn(f"\nWarning, the interaction between the following sub groups {missing_sub_groups} have been set to 0", UserWarning)
    

    # Return the list of missing main group interactions
    return missing_main_groups, missing_sub_groups



def load_gc(file_path, output_format):
    """
    This function reads an Excel file and returns the data in the specified format.
    This function loads the Constantinou-Gani group contributions (gc) as well as the UNIFAC interaction parameters (a)
     and returns them as either a dataframe or as a dict of dicts

    Parameters:
    file_path (str): The path to the Excel file.
    Output_format (str): The format to return the data in. Options are 'dataframe' or 'dict'.

    Returns:
    tuple: A tuple containing two elements () gc and a.
           If output_format is 'dataframe', the elements are two pandas DataFrames.
           If output_format is 'dict', the elements are two dictionaries.
           The first element contains data for the pure component properties from various sheets in the Excel file.
           The second element contains interaction parameters from the 'a' sheet in the Excel file.

    Raises:
    ValueError: If output_format is neither 'dataframe' nor 'dict'.
    """

    # Define a list of sheet names and corresponding column names
    sheet_columns = {
        'set': 'set',       # groups: main groups and subgroups
        'noc': 'noc',       # Number of connections
        'Rk': 'Rk',         # volume parameter of group k
        'Qk': 'Qk',         # surface area parameter of group k
        'Mw': 'Mw',         # Molecular weight in g/mol
        'Tm': 'Tm',         # Normal melting point in K
        'Tb': 'Tb',         # Normal boiling point in K
        'Tc': 'Tc',         # Critical temperature in K
        'Pc': 'Pc',         # Critical pressure in bar
        'Vc': 'Vc',         # Critical volume in cm3/mol
        'Gf': 'Gf',         # Gibbs free energy fo formation in kJ/mol
        'Hf': 'Hf',         # Enthalpy of formation in kJ/mol
        'Hv': 'Hv',         # Enthalpy of vaporization in kJ/mol
        'Ac': 'Ac',         # Acentric factor in -
        'Vm': 'Vm',         # Molar volume in cm3/mol
        'Gtt': 'Gtt',       #
        'LogP': 'LogP',     # Octanol/water partition coefficient
        'LogWs': 'LogWs',   # log water solubility
        'tfc': 'tfc',       # closed cup flash point in K
        'tfo': 'tfo',       # open cup flash point in K
        'ccsub': 'ccsub',   #
        'FDX': 'FDX',       # hansen dispersion solubility parameter MPa^1/2
        'FPX': 'FPX',       # hansen polar solubility parameter MPa^1/2
        'FHX': 'FHX',       # hansen hydrogen solubility parameter MPa^1/2
        'EttaA': 'EttaA',   # viscosity in cp
        'EttaB': 'EttaB',   # viscosity in cp
        'LC50': 'LC50'      # Acute toxicity (96h- LC50) to the fathead minnow
    }

    # Read the entire Excel file once
    xls = pd.ExcelFile(file_path)

    # Read each sheet and assign columns to gc_df
    #gc_df = pd.concat([pd.read_excel(xls, sheet_name=sheet, usecols="B") for sheet in sheet_columns.keys()], axis=1)
    gc_df = pd.concat([pd.read_excel(xls, sheet_name=sheet, usecols="C" if sheet == 'set' else "B") for sheet in sheet_columns.keys()], axis=1)
    # Rename columns
    gc_df.columns = sheet_columns.values()

    # make groups as indices
    gc_df.set_index('set', inplace=True)
    gc_df.rename_axis(None, inplace=True)

    # Construct dataframe for interaction params
    a_df = pd.read_excel(xls, sheet_name='a', index_col=0)
    # delete the first row that contains the main group data
    a_df.drop(a_df.index[0], inplace=True)
    # changing the type to float 64
    a_df = a_df.apply(pd.to_numeric, errors='coerce').astype('float64')

    if output_format == 'dataframe':
        return gc_df, a_df
    elif output_format == 'dict':
        # Convert into dicts
        gc_dict = gc_df.stack().to_dict()
        a_dict = a_df.stack().to_dict()
        return gc_dict, a_dict
    else:
        raise ValueError("Invalid output_format. Choose either 'dataframe' or 'dict'.")



def construct_mixture(tag, groups):
    """
    
    :param tag:
    :param groups:
    :return:
    """
    df_mix = pd.DataFrame(groups)
    df_mix['compound'] = tag
    df_mix.set_index('compound', inplace=True)
    return df_mix


def prop_calc(mol, prop_list, contrib):
    """ Function that calculates various thermophysical properties using the Contantinou-Gani model

    :param mol: a pandas dataframe with the occurrence of the groups
    :param prop_list:  list of properties of interest
    :param contrib: dataframe of GC contributions
    :return: y: dataframe with all the properties
    mw: Molar weight in mol/g
    tm: Normal melting point in K
    #TODO: add units

    """

    idx = mol.columns
    y = pd.DataFrame()
    y['mw'] = sum(contrib.Mw[i] * mol[i] for i in idx)
    y['Tm'] = 102.425 * np.log(sum(contrib.Tm[i] * mol[i] for i in idx))
    y['Tb'] = 204.359 * np.log(sum(contrib.Tb[i] * mol[i] for i in idx))
    y['Tc'] = 181.128 * np.log(sum(contrib.Tc[i] * mol[i] for i in idx))
    y['Pc'] = 1 / (sum(contrib.Pc[i] * mol[i] for i in idx) + 0.100220) ** 2 + 1.3705
    y['Vc'] = -4.35 + 1000 * sum(contrib.Vc[i] * mol[i] for i in idx)
    y['Gf'] = -14.828 + sum(contrib.Gf[i] * mol[i] for i in idx)
    y['Hf'] = 10.835 + sum(contrib.Hf[i] * mol[i] for i in idx)
    y['Hv'] = 6.829 + sum(contrib.Hv[i] * mol[i] for i in idx)
    y['Vm'] = 12.11 + sum(1000 * contrib.Vm[i] * mol[i] for i in idx)
    y['Ac'] = 0.4085 * np.log(sum(contrib.Ac[i] * mol[i] for i in idx) + 1.1507) ** 1.9802
    y['Zc'] = y['Pc'] * y['Vc'] / (83.14 * y['Tc'])
    y['Vm_Tb'] = 0.285 * y['Vc'] ** 1.048
    y['tfc'] = -2.03 * sum(contrib.tfc[i] * mol[i] for i in idx) + 0.659 * y['Tb'] + 20.00
    y['tfo'] = 3.63 * sum(contrib.tfo[i] * mol[i] for i in idx) + 0.409 * y['Tb'] + 88.43
    y['HDs'] = (sum(contrib.FDX[i] * mol[i] for i in idx)) / y['Vm']
    y['HPs'] = (sum(contrib.FPX[i] * contrib.FPX[i] * mol[i] for i in idx)) ** 0.5 / y['Vm']
    y['HHs'] = (sum(contrib.FHX[i] * mol[i] for i in idx) / y['Vm']) ** 0.5
    y['SolPar'] = ((1000 * y['Hv'] - 8.314 * 298.15) / y['Vm']) ** 0.5
    # y['visc'] = y['mw'] * 1000 * np.exp((-597.82 + sum(contrib.EttaA[i] * mol[i] for i in idx)) / 290 + sum(
    #     contrib.EttaB[i] * mol[i] for i in idx) - 11.202)
    y['LC50'] = sum(contrib.LC50[i] * mol[i] for i in idx)
    return y[prop_list]


def antoine_correlation(A, B, C, T):
    """
    Computes the saturation pressure (P*) using the Antoine equation:
    log10(P*) = A - B / (T + C), where P* is the saturation pressure in MPa.

    :param A: Antoine constant A
    :param B: Antoine constant B
    :param C: Antoine constant C
    :param T: Temperature in Kelvin (K)
    :return: Saturation pressure (P*) in bar
    """
    # Calculate the saturation pressure using the Antoine equation.
    return np.power(10, A - B / (T + C))


def pitzer_acentric_factor(A, B, C, Tc, Pc):
    """
    Calculates the Pitzer acentric factor based on the Antoine correlation.

    The acentric factor (w) is calculated using:
    w = -log10(P_sat / Pc) - 1,
    where P_sat is the saturation pressure at Tr = 0.7 (70% of Tc),
    and Pc is the critical pressure.

    :param A: Antoine constant A
    :param B: Antoine constant B
    :param C: Antoine constant C
    :param Tc: Critical temperature (K)
    :param Pc: Critical pressure (MPa)
    :return: Acentric factor (w)
    """

    # Reduced temperature Tr is 0.7 (as per Pitzer definition)
    Tr = 0.7
    T = Tr * Tc  # Calculate the temperature at Tr=0.7*Tc

    # Calculate saturation pressure at Tr = 0.7 * Tc
    P_sat = antoine_correlation(A, B, C, T)

    # Calculate acentric factor using Pitzer's equation
    w = -np.log10(P_sat / Pc) - 1

    return w

def solve_antoine(Tb: float, Tc: float, Pc: float, P0: float, w: float, verbose: bool) -> dict:
    """ Function that solves for the A, B and C parameters fom the antoine equation using pyomo

    :param Tb: the melting point in Kelvin
    :param Tc: the critical temperature in Kelvin
    :param Pc: the critical pressure in MPa
    :param P0: the atmospheric pressure in MPa
    :param w: the acentric factor
    :param verbose: printing of solver progress

    :return: a dict containing: solver_stat: status of the solver
                                term_cond: termination condition of the solver
                                num_iter: number of iterations
                                obj_val: the objective values
                                A: Antoine parameter A
                                B: Antoine parameter B
                                C: Antoine parameter C
    NB: the Antoine produces saturated pressures using the equation: P(MPa)=10 ** (A - B / (C + T))
    """

    # initializations:

    """
        Step 1: find the the saturated pressure at the acentric factor e.i. T=Tc*0.7
    """
    # calculate the saturated vapor pressure for Tr = 0.7
    Pw = np.power(10, -(w + 1)) * Pc

    """
     Step 2: find the initial guess for the Antoine parameters using selected-points method
    """
    # extract the points of interest
    t1 = Tb
    t2 = 0.7 * Tc
    t3 = Tc
    p1 = P0
    p2 = Pw
    p3 = Pc
    y1 = np.log10(p1)
    y2 = np.log10(p2)
    y3 = np.log10(p3)

    # equation for C
    X = (y3 - y2) / (y2 - y1) * (t2 - t1) / (t3 - t2)
    Y = t1 - t3
    C0 = Y / (X - 1) - t3

    # equation for B
    B0 = (y3 - y1) / (t3 - t1) * (t1 + C0) * (t3 - C0)

    # equation for A
    A0 = y2 + B0 / (t2 + C0)

    """
    Step 3: find the final value for the Antoine parameters
    """
    # print('------------- Finding the final values for the Antoine parameters ----------')

    # define the model type
    model = pyo.ConcreteModel('system of equation for solving the Antoine parameters')
    # define the parameters
    model.tb = pyo.Param(initialize=Tb, mutable=False, name='normal boiling point in K', domain=pyo.PositiveReals)
    model.tc = pyo.Param(initialize=Tc, mutable=False, name='the critical temperature in K', domain=pyo.PositiveReals)
    model.pc = pyo.Param(initialize=Pc, mutable=False, name='the critical pressure in MPa', domain=pyo.PositiveReals)
    model.w = pyo.Param(initialize=w, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
    model.p0 = pyo.Param(initialize=P0, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
    model.pw = pyo.Param(initialize=Pw, mutable=False, name='pressure at Tr=0.7', domain=pyo.PositiveReals)
    # define the model variables
    model.A = pyo.Var(within=pyo.Reals, initialize=A0, name='Antoine parameter A')
    model.B = pyo.Var(within=pyo.Reals, initialize=B0, name='Antoine parameter B')
    model.C = pyo.Var(within=pyo.Reals, initialize=C0, name='Antoine parameter C')
    #model.pr = pyo.Var(within=pyo.PositiveReals, initialize=10 ** (A0 - B0 / (C0 + Tc * 0.70)), name='Antoine parameter C')

    # define the constraints
    def eq_p0(m):
        """ Constraint for the equality of the atmospheric pressure

        :param m:
        :return:
        """
        return pyo.log10(m.p0) == m.A - m.B / (m.C + m.tb)

    model.eq_p0 = pyo.Constraint(rule=eq_p0, name='calculation rule for p atmospheric')

    def eq_pc(m):
        """ Constraint for the equality of the critical pressure

        :param m:
        :return:
        """
        return pyo.log10(m.pc) == m.A - m.B / (m.C + m.tc)

    model.eq_pc = pyo.Constraint(rule=eq_pc, name='calculation rule for p critical')

    def eq_pw(m):
        """

        :param m:
        :return:
        """

        return pyo.log10(m.pw) == m.A - m.B / (m.C + m.tc*0.7)

    model.eq_pw = pyo.Constraint(rule=eq_pw, name='calculation rule for p at Tr=0.7')

    # define the solver
    opt = pyo.SolverFactory('ipopt')
    # show more stuff
    opt.options["halt_on_ampl_error"] = "yes"
    opt.options["linear_solver"] = "MA27"
    opt.options["max_iter"] = 10000

    # solve the optimization problem
    solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True, logfile="pyomo_antoine_log")

    solv_stat = str(solution.solver.status)
    term_cond = str(solution.solver.termination_condition)
    if solv_stat != 'ok' or term_cond != 'optimal':
        print('Something is wrong with compound while solving for the Antoine parameters')

    output = {'solv_stat': str(solution.solver.status),
              'term_cond': str(solution.solver.termination_condition),
              'num_iter': parse_pyomo_log('pyomo_antoine_log'),
              'A0': A0,
              'B0': B0,
              'C0': C0,
              'A': model.A.value,
              'B': model.B.value,
              'C': model.C.value}

    return output


def psat_validation(A, B, C, T, Tb, Tc, Pc, w, name):
    """
    Validates and plots saturation pressure (Psat) over a range of temperatures
    using Antoine's equation.

    :param A: Antoine constant A
    :param B: Antoine constant B
    :param C: Antoine constant C
    :param T: Temperature array (if None, generates a range between 0.8*Tb to 1.2*Tc)
    :param Tb: Boiling temperature (K)
    :param Tc: Critical temperature (K)
    :param Pc: Critical pressure (bar)
    :param w: Acentric factor for the component
    :param name: Name of the component (for plot title)
    :return: Dictionary containing the figure, saturation pressure array, temperature array,
             and percentage errors for P0, Pc, and acentric factor w.
    """

    # Calculate the saturation pressure at the boiling point (Tb) using Antoine's equation.
    P0_clc = antoine_correlation(A, B, C, Tb)
    # Calculate the Absolute Relative Error (ARE) between calculated and expected boiling pressure (0.101325 bar).
    are_P0 = (abs(0.101325 - P0_clc)) / 0.101325 * 100

    # Calculate the saturation pressure at the critical point (Tc) using Antoine's equation.
    Pc_clc = antoine_correlation(A, B, C, Tc)
    # Calculate the Absolute Relative Error (ARE) between calculated and provided critical pressure (Pc).
    are_Pc = (abs(Pc - Pc_clc)) / Pc * 100

    # Calculate the acentric factor using the Pitzer method (assuming pitzer_acentric_factor is implemented).
    w_clc = pitzer_acentric_factor(A, B, C, Tc, Pc)
    # Calculate the Absolute Relative Error (ARE) between calculated and provided acentric factor (w).
    are_w = (abs(w - w_clc)) / w * 100

    # If the temperature array `T` is not provided, generate a default range from 0.8*Tb to 1.2*Tc.
    if T is None:
        T = np.arange(Tb * 0.8, Tc * 1.2, 5)

    # Compute the saturation pressure for the temperature array `T` using Antoine's equation.
    psat = antoine_correlation(A, B, C, T)

    # Calculate the saturation pressure at Tr = 0.7*Tc, an important reference point in the graph.
    Pw = antoine_correlation(A, B, C, 0.7 * Tc)

    # Create a new figure for plotting the saturation pressure curve.
    fig = plt.figure()

    # Plot the saturation pressure `psat` as a function of temperature `T`.
    plt.plot(T, psat, label='P$^{sat}$')
    plt.xlabel('Temperature (K)')  # Label the x-axis.
    plt.ylabel('Saturation Pressure (bar)')  # Label the y-axis.
    plt.title(f'Saturation Pressure vs Temperature for {name}')  # Set the title of the plot.

    # Plot important points on the graph:
    # 1. Boiling point (Tb) with saturation pressure calculated from Antoine equation.
    plt.scatter(Tb, P0_clc, color='red', s=100, edgecolor='black', zorder=5, label='Tb: P0')
    # 2. The point at Tr = 0.7*Tc (Pw is the pressure at this reduced temperature).
    plt.scatter(0.7 * Tc, Pw, color='red', s=100, edgecolor='black', zorder=5, label='Tr=0.7')
    # 3. Critical point (Tc, Pc).
    plt.scatter(Tc, Pc, color='red', s=100, edgecolor='black', zorder=5, label='Tc: Pc')

    # Add a legend to label the important points.
    plt.legend()

    # Prepare the output dictionary:
    # - 'fig' contains the plotted figure.
    # - 'T' contains the temperature array.
    # - 'psat' contains the saturation pressure array.
    # - 'are_P0', 'are_Pc', 'are_w' contain the rounded percentage errors for P0, Pc, and w respectively.
    output = {
        'fig': fig,
        'T': T,
        'psat': psat,
        'are_P0': are_P0.round(),  # Rounded ARE for boiling point.
        'are_Pc': are_Pc.round(),  # Rounded ARE for critical point.
        'are_w': are_w.round()  # Rounded ARE for acentric factor.
    }

    # Return the output dictionary.
    return output

def unifac(mixture: pd.DataFrame, x0, T0: float) -> np.array:
    """ implementation of the unifac method according to smith et al. (2001)
    slight modification for improved numerical stability according to
    https://doi.org/10.1016/j.ces.2022.118125

    :param mixture: dataframe with the compounds and respective groups considered
    :param x0: np.ndarray of composition
    :param T0: temperature
    :param df_gc: dataframe of group contribution parameter
    :param df_a: dataframe of interaction parameters
    :return: gamma: liquid activity coefficients
    :return:
    """
    # load data
    file_path = '../data/data_unifac.xlsx'
    df_gc, df_a = load_gc(file_path, output_format='dataframe')
    # extract data
    v = np.array(mixture)
    Rk = np.array(df_gc['Rk'][mixture.columns])
    Qk = np.array(df_gc['Qk'][mixture.columns])
    a = np.array(df_a[mixture.columns].loc[mixture.columns])
    # begin calculations: combinatorial term
    r = np.dot(v, Rk)
    q = np.dot(v, Qk)
    L = np.mat(q) / np.mat(np.dot(x0, q)).T
    j = np.mat(r) / np.mat(np.dot(x0, r)).T
    gamma_c = 1 - j + np.log(j) - 5 * np.multiply(q, (1 - j / L + np.log(j / L)))
    # begin calculation: residual term
    e = ((v * Qk).T / q).T
    tau = np.exp(-a / T0)
    beta = ((np.dot(v * Qk, tau)).T / q).T
    theta = np.dot(x0, v * Qk) / np.dot(x0, q)
    omega = np.dot(theta, tau)
    gamma_r = q * (1 - np.sum(theta * beta / omega - e * np.log(beta / omega), axis=1))
    gamma = np.array(np.exp(gamma_c + gamma_r))
    return gamma

def make_log_gamma_plot(mixture, x0, T0):
    """

    :param mixture:
    :param x0:
    :param T0:
    :return:
    """
    # extract data
    n_data = x0.shape[0]
    # initialize
    gamma = np.empty_like(x0)
    # evaluate the gamma
    for i in range(n_data):
        gamma[i, :] = unifac(mixture, x0[i, :], T0)
    # calculate the log
    log_gamma = np.log(gamma)
    # make the plot
    fig = plt.figure()
    plt.plot(x0[:, 0], log_gamma[:, 0], label='C1')
    plt.plot(x0[:, 0], log_gamma[:, 1], label='C2')
    plt.title('log $\gamma$ vs x')
    plt.xlabel('x')
    plt.ylabel('log $\gamma$')
    plt.xlim([0,1])
    plt.legend(loc='best')


    output = {'fig': fig, 'gamma': gamma}
    return output

def unifac_init(mixture: pd.DataFrame, x0: float, T0: float, df_gc: pd.DataFrame, df_a: pd.DataFrame) -> np.array:
    """ function to initialize unifac model calculations for pyomo
    NB: there are semilarities beetween this function and the function unifac
    However this function retains and respects the nomenclature and dimensions
    adopted in the pyomo bubble-t calculation

    :param mixture: dataframe with the compounds and rspective groups considered
    :param x0: np.ndarray of composition
    :param T0: temperature
    :param df_gc: dataframe of group contribution parameter
    :param df_a: dataframe of interaction parameters
    :return: gamma: liquid activity coefficients
    """
    v = np.array(mixture)
    Rk = np.array(df_gc['Rk'][mixture.columns])  # np.array(df_gc['Rk'][df_mix.columns])
    Qk = np.array(df_gc['Qk'][mixture.columns])
    a = np.array(df_a[mixture.columns].loc[mixture.columns])
    # begin calculations
    r = np.dot(v, Rk)
    q = np.dot(v, Qk)
    j = np.mat(r) / np.mat(np.dot(x0, r)).T
    L = np.mat(q) / np.mat(np.dot(x0, q)).T
    g = v * Qk
    theta = np.dot(x0, g)
    tau = np.transpose(np.exp(-a[..., None] / T0), (2, 0, 1))
    s = np.transpose(np.dot(g, tau), (1, 2, 0))
    eta = np.multiply(x0[:, :, None], np.transpose(s, (0, 2, 1))).sum(axis=1)
    gamma_c = 1 - j + np.log(j) - 5 * np.multiply(q, (1 - j / L + np.log(j / L)))
    gamma_r = np.multiply(q, (1 - np.log(L))) - np.sum(
        np.transpose(np.multiply(theta[:, :, None], s / eta[:, :, None]), (0, 2, 1)) - g * np.transpose(
            np.log(s / eta[:, :, None]), (0, 2, 1)), axis=2)
    gamma = np.array(np.exp(gamma_c + gamma_r))
    # make all the para,eters into dicts
    r = dict(enumerate(r.flatten(), 0))
    q = dict(enumerate(q.flatten(), 0))
    j = {index: v for index, v in np.ndenumerate(j)}
    L = {index: v for index, v in np.ndenumerate(L)}
    g = pd.DataFrame(g, columns=mixture.columns).T.stack().to_dict()
    theta = pd.DataFrame(theta, columns=mixture.columns).stack().to_dict()

    def custom_key(index, new_keys, v_name):
        """ multi-level dict (dim=3) dict constructor"""
        x1, x2, x3 = index
        if v_name == 'tau':
            return x1, new_keys[x2], new_keys[x3]
        if v_name == 's':
            return x1, new_keys[x2], x3

    tau = {custom_key(index, mixture.columns, 'tau'): value for index, value in np.ndenumerate(tau)}
    s = {custom_key(index, mixture.columns, 's'): value for index, value in np.ndenumerate(s)}
    eta = pd.DataFrame(eta, columns=mixture.columns).stack().to_dict()
    gamma_c = {index: v for index, v in np.ndenumerate(gamma_c)}
    gamma_r = {index: v for index, v in np.ndenumerate(gamma_r)}
    gamma = {index: v for index, v in np.ndenumerate(gamma)}
    return [r, q, j, L, g, theta, tau, s, eta, gamma_c, gamma_r, gamma]



def pyo_unifac_bubbleT(x, T0, P, antoine_params, mixture, verbose: bool):
    """ A helper or wrapper function over the pyomo procedure for conducting bubble-T calculations using unifac

    :param x: the liquid composition
    :param T0: initial guess for the saturation temperature in kelvin
    :param P: the total pressure in MPa
    :param antoine_params: array of Antoine parameters n_comp x 3
    :param mixture: dataframe of the mixture groups as columns
    :return:
    """
    # create the model
    model = pyo.ConcreteModel()
    # define the parameters of the model
    n_comp = x.shape[1]
    n_data = x.shape[0]
    # load the unifac parameters
    file_path = '../trash/unifac.xlsx'
    df_gc, df_a = load_gc(file_path, output_format='dataframe')
    # define sets
    model.k = pyo.Set(initialize=mixture.columns, name='Groups present')
    model.i = pyo.RangeSet(0, n_comp - 1, name=' no. of Compounds')
    model.dp = pyo.RangeSet(0, n_data - 1, name='no. of data points')
    model.grp = pyo.Set(initialize=df_a.columns, name='all Groups')

    # define the constants/parameters
    model.A = pyo.Param(model.i, initialize=antoine_params[:, 0], mutable=False, name='Antoine A', domain=pyo.Reals)
    model.B = pyo.Param(model.i, initialize=antoine_params[:, 1], mutable=False, name='Antoine B', domain=pyo.Reals)
    model.C = pyo.Param(model.i, initialize=antoine_params[:, 2], mutable=False, name='Antoine C', domain=pyo.Reals)
    model.a = pyo.Param(model.grp, model.grp, initialize=df_a.stack().to_dict(), mutable=False,
                        name='unifac interaction a_mk', domain=pyo.Reals)
    model.Rk = pyo.Param(model.grp, initialize=df_gc.Rk.to_dict(), mutable=False, name='unifac Rk parameters')
    model.Qk = pyo.Param(model.grp, initialize=df_gc.Qk.to_dict(), mutable=False, name='unifac Qk parameters')
    model.x = pyo.Param(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(x)}, mutable=False,
                        name='Liquid composition x', domain=pyo.Reals)
    model.P = pyo.Param(initialize=P, mutable=False, name='Total pressure in MPa', domain=pyo.PositiveReals)
    model.v = pyo.Param(model.i, model.k, initialize=mixture.stack().to_dict(), mutable=False, name='compounds')

    # perform unifac evaluation for better initialization
    [r_init, q_init, j_init, L_init, g_init, theta_init, tau_init, s_init, eta_init, gamma_c_init, gamma_r_init,
     gamma_init] = unifac_init(mixture, x, T0, df_gc, df_a)
    p_init = 10**(antoine_params[:, 0][:, None]-antoine_params[:, 1][:, None]/(antoine_params[:, 2][:, None]+T0))
    # M = np.zeros(shape=(n_data, n_comp))
    # M[tuple(zip(*gamma_init.keys()))] = list(gamma_init.values())
    # y_init = p_init * np.transpose(x, (1, 0)) * np.transpose(M, (1, 0)) / P
    # y_init = np.round(y_init, 3)
    # print(y_init)
    # y_init = {(index[1], index[0]): v for index, v in np.ndenumerate(y_init)}
    y_init = {index: v for index, v in np.ndenumerate(x)}
    p_init = {(index[1], index[0]): v for index, v in np.ndenumerate(p_init)}
    #model.T = pyo.Var(model.dp, initialize=T0, within=pyo.PositiveReals,      name='Temperature in K')
    # model.T = pyo.Var(model.dp, initialize=T0, within=pyo.PositiveReals, bounds=(T0.min() - 20, T0.max() + 10),
    #                    name='Temperature in K')
    model.T = pyo.Var(model.dp, initialize=T0, within=pyo.PositiveReals, bounds=(T0.min() - 20, T0.max() + 10),
                      name='Temperature in K')
    model.r = pyo.Param(model.i, initialize=r_init, mutable=False, name='unifac r_i')
    model.q = pyo.Param(model.i, initialize=q_init, mutable=False, name='unifac q_i')
    model.J = pyo.Param(model.dp, model.i, initialize=j_init, mutable=False, name='unifac J_i')
    model.L = pyo.Param(model.dp, model.i, initialize=L_init, mutable=False, name='unifac L_i')
    model.G = pyo.Param(model.k, model.i, initialize=g_init, mutable=False, name='unifac G_ki')
    model.theta = pyo.Param(model.dp, model.k, initialize=theta_init, mutable=False, name='unifac theta_k')

    # define the model variables


    model.tau = pyo.Var(model.dp, model.k, model.k, initialize=tau_init, within=pyo.PositiveReals, name='unifac tau_mk')
    model.s = pyo.Var(model.dp
                      , model.k, model.i, initialize=s_init, within=pyo.PositiveReals, name='unifac s_ki')
    model.eta = pyo.Var(model.dp, model.k, initialize=eta_init, within=pyo.PositiveReals, name='unifac eta_k')
    model.gamma_c = pyo.Var(model.dp, model.i, initialize=gamma_c_init, within=pyo.Reals,
                            name='unifac combinatorial gamma')
    model.gamma_r = pyo.Var(model.dp, model.i, initialize=gamma_r_init, within=pyo.Reals, name='unifac residual gamma')
    model.gamma = pyo.Var(model.dp, model.i, initialize=gamma_init, within=pyo.Reals, name='unifac total gamma')
    model.psat = pyo.Var(model.dp, model.i, initialize=p_init, within=pyo.PositiveReals, bounds=(None, pyo.value(3000)),
                         name='vapour pressure in bar')
    model.y = pyo.Var(model.dp, model.i, initialize=y_init, within=pyo.PercentFraction, name='vapour composition y')

    # define the constraints: god help me
    def sum_y(m, j):
        """
        Constrain for unity summation of the vapour fractions
        :param m: pyomo model
        :param j: index for datapoints
        :return: boolean
        """
        return sum(m.y[j, i] for i in m.i) == 1.0

    model.sum_y = pyo.Constraint(model.dp, rule=sum_y, name='Constraint for summation of the vapour fractions to unity')

    def eq_psat(m, j, i):
        """ Constraint for saturated vapour pressure equality
        :param m: pyomo model
        :param j: data point index
        :param i: compound index
        :return: boolean
        """
        return m.psat[j, i] == 10 ** (m.A[i] - m.B[i] / (m.C[i] + m.T[j]))

    model.eq_psat = pyo.Constraint(model.dp, model.i, rule=eq_psat, name='calculation rule for psat')


    def eq_tau(m, dp, k, l):
        """

        :param dp:
        :param m:
        :param k:
        :param l:
        :return:
        """
        return m.tau[dp, k, l] == pyo.exp(-m.a[k, l] / m.T[dp])

    model.eq_tau = pyo.Constraint(model.dp, model.k, model.k, rule=eq_tau,
                                  name='Constraint for calculating tau_k,m of unifac')

    def eq_s(m, dp, k, i):
        """

        :param m:
        :param dp:
        :param k:
        :param i:
        :return:
        """
        return m.s[dp, k, i] == sum(m.G[l, i] * m.tau[dp, l, k] for l in m.k)

    model.eq_s = pyo.Constraint(model.dp, model.k, model.i, rule=eq_s,
                                name='Constraint for calculating s_k,i of unifac')

    def eq_eta(m, dp, k):
        """

        :param m:
        :param dp:
        :param k:
        :return:
        """
        return m.eta[dp, k] == sum(m.s[dp, k, i] * m.x[dp, i] for i in m.i)

    model.eq_eta = pyo.Constraint(model.dp, model.k, rule=eq_eta,
                                  name='Constraint for calculating eta_k of unifac')

    def eq_gamma_c(m, dp, i):
        """

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.gamma_c[dp, i] == 1 - m.J[dp, i] + pyo.log(m.J[dp, i]) - 5 * m.q[i] * (
                1 - m.J[dp, i] / m.L[dp, i] + pyo.log(m.J[dp, i] / m.L[dp, i]))

    model.eq_gamma_c = pyo.Constraint(model.dp, model.i, rule=eq_gamma_c,
                                      name='Constraint for calculating gammaC_i of unifac')

    def eq_gamma_r(m, dp, i):
        """

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.gamma_r[dp, i] == m.q[i] * (1 - pyo.log(m.L[dp, i])) - sum(
            m.theta[dp, k] * m.s[dp, k, i] / m.eta[dp, k] - m.G[k, i] * pyo.log(m.s[dp, k, i] / m.eta[dp, k]) for k in
            m.k)

    model.eq_gamma_r = pyo.Constraint(model.dp, model.i, rule=eq_gamma_r,
                                      name='Constraint for calculating gammaR_i of unifac')

    def eq_gamma(m, dp, i):
        """

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.gamma[dp, i] == pyo.exp(m.gamma_r[dp, i] + m.gamma_c[dp, i])

    model.eq_gamma = pyo.Constraint(model.dp, model.i, rule=eq_gamma, name='Constraint for calculating gamma')

    def eq_y(m, dp, i):
        """ function that calculates y
        :param m: pyomo model
        :param dp: data point index
        :param i: compound index
        :return:
        """
        return m.y[dp, i] == m.psat[dp, i] * model.x[dp, i] * m.gamma[dp, i] / m.P

    model.eq_y = pyo.Constraint(model.dp, model.i, rule=eq_y, name='calculation rule for vapour fraction')

    # define the solver
    opt = pyo.SolverFactory('ipopt')
    # define solver options
#    opt.options["halt_on_ampl_error"] = "yes"
    opt.options["linear_solver"] = "MA27"
    opt.options["max_iter"] = 10000
#    opt.options["honor_original_bounds"] = "yes"
#    opt.options["bound_relax_factor"] = 0
    opt.options["expect_infeasible_problem"] = "yes"
#    opt.options["findiff_perturbation"] = 1e-2
    opt.options["warm_start_init_point"] = "yes"
    # solve the optimization problem
    solution = None
    solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True, logfile="pyo_vle_log")

    solv_stat = str(solution.solver.status)
    term_cond = str(solution.solver.termination_condition)
    if solv_stat != 'ok' or term_cond != 'optimal':
        print('Something is wrong with compound  while solving for bubble-T')

    return solution, model

def iss_method(x0, T0, P, antoine_params, mixture, verbose):
    """ A wrapper function around the pyomo model for the ISS shortcut method for evalouating entrainer performance

    :param x0:
    :param T0:
    :param P:
    :param antoine_params:
    :param mixture:
    :return:
    """

    # create the model
    model = pyo.ConcreteModel()
    # define the parameters of the model
    n_comp = x0.shape[1]
    n_data = x0.shape[0]
    # create initialization for K
    k_init = {('iso', 0): 0.7,
              ('iso', 1): 0.7,
              ('iso', 2): 0.7,
              ('min', 0): 1.2,
              ('min', 1): 0.6,
              ('min', 2): 0.7}
    # load the unifac parameters
    file_path = '../trash/unifac.xlsx'
    df_gc, df_a = load_gc(file_path, output_format='dataframe')
    # define sets
    model.k = pyo.Set(initialize=mixture.columns, name='Groups present')
    model.i = pyo.RangeSet(0, n_comp - 1, name=' no. of Compounds')
    model.dp = pyo.Set(initialize=['iso', 'min'], name='no. of data points')
    model.grp = pyo.Set(initialize=df_a.columns, name='all Groups')

    # define the constants/parameters
    model.A = pyo.Param(model.i, initialize=antoine_params[:, 0], mutable=False, name='Antoine A', domain=pyo.Reals)
    model.B = pyo.Param(model.i, initialize=antoine_params[:, 1], mutable=False, name='Antoine B', domain=pyo.Reals)
    model.C = pyo.Param(model.i, initialize=antoine_params[:, 2], mutable=False, name='Antoine C', domain=pyo.Reals)
    model.a = pyo.Param(model.grp, model.grp, initialize=df_a.stack().to_dict(), mutable=False,
                        name='unifac interaction a_mk', domain=pyo.Reals)
    model.Rk = pyo.Param(model.grp, initialize=df_gc.Rk.to_dict(), mutable=False, name='unifac Rk parameters')
    model.Qk = pyo.Param(model.grp, initialize=df_gc.Qk.to_dict(), mutable=False, name='unifac Qk parameters')
    model.P = pyo.Param(initialize=P, mutable=False, name='Total pressure in MPa', domain=pyo.PositiveReals)
    model.v = pyo.Param(model.i, model.k, initialize=mixture.stack().to_dict(), mutable=False, name='compounds')
    # define the model variables
    model.T = pyo.Var(model.dp, initialize=T0, within=pyo.PositiveReals, name='Temperature in K')
    model.r = pyo.Var(model.i, initialize=0.4, within=pyo.PositiveReals, name='unifac r_i')
    model.q = pyo.Var(model.i, initialize=4.5, within=pyo.PositiveReals, name='unifac q_i')
    model.J = pyo.Var(model.dp, model.i, initialize=1.13, within=pyo.PositiveReals, name='unifac J_i')
    model.L = pyo.Var(model.dp, model.i, initialize=1.15, within=pyo.PositiveReals, name='unifac L_i')
    model.G = pyo.Var(model.k, model.i, initialize=1.5, within=pyo.PositiveReals, name='unifac G_ki')
    model.theta = pyo.Var(model.dp, model.k, initialize=1.2, within=pyo.PositiveReals, name='unifac theta_k')
    model.tau = pyo.Var(model.dp, model.k, model.k, initialize=0.79, within=pyo.PositiveReals, name='unifac tau_mk')
    model.s = pyo.Var(model.dp, model.k, model.i, initialize=1.7, within=pyo.PositiveReals, name='unifac s_ki')
    model.eta = pyo.Var(model.dp, model.k, initialize=1.1, within=pyo.PositiveReals, name='unifac eta_k')
    model.gamma_c = pyo.Var(model.dp, model.i, initialize=0.7, within=pyo.Reals, name='unifac combinatorial gamma')
    model.gamma_r = pyo.Var(model.dp, model.i, initialize=1.5, within=pyo.Reals, name='unifac residual gamma')
    model.gamma = pyo.Var(model.dp, model.i, initialize=2.1, within=pyo.Reals, name='unifac total gamma')
    model.psat = pyo.Var(model.dp, model.i, initialize=0.9, within=pyo.PositiveReals, name='vapour pressure in bar')
    model.y = pyo.Var(model.dp, model.i, initialize=0.6, within=pyo.PercentFraction, name='vapour composition y')
    model.x = pyo.Var(model.dp, model.i, initialize=0.7, name='Liquid composition x', within=pyo.PercentFraction)
    model.K = pyo.Var(model.dp, model.i, initialize=k_init, name='Equilibrium constants', within=pyo.PositiveReals)
    model.ED = pyo.Var(initialize=3.2, name='Entrainer to distillate ratio', within=pyo.PositiveReals)
    model.R = pyo.Var(initialize=0.5, name='Reflux ratio', within=pyo.PositiveReals)

    # define the constraints: god help me
    def sum_y(m, j):
        """
        Constrain for unity summation of the vapour fractions
        :param m: pyomo model
        :param j: index for datapoints
        :return: boolean
        """
        return sum(m.y[j, i] for i in m.i) == 1.0

    model.sum_y = pyo.Constraint(model.dp, rule=sum_y, name='Constraint for summation of the vapour fractions to unity')

    def sum_x(m, j):
        """
        Constrain for unity summation of the liquid fractions
        :param m: pyomo model
        :param j: index for datapoints
        :return: boolean
        """
        return sum(m.x[j, i] for i in m.i) == 1.0

    model.sum_x = pyo.Constraint(model.dp, rule=sum_x, name='Constraint for summation of the liquid fractions to unity')

    def eq_psat(m, j, i):
        """ Constraint for saturated vapour pressure equality
        :param m: pyomo model
        :param j: data point index
        :param i: compound index
        :return: boolean
        """
        return m.psat[j, i] == 10 ** (m.A[i] - m.B[i] / (m.C[i] + m.T[j]))

    model.eq_psat = pyo.Constraint(model.dp, model.i, rule=eq_psat, name='calculation rule for psat')


    def eq_J(m, j, i):
        """

        :param m:
        :param j:
        :param i:
        :return:
        """
        return m.J[j, i] == m.r[i] / sum(m.r[i] * m.x[j, i] for i in m.i)

    model.eq_J = pyo.Constraint(model.dp, model.i, rule=eq_J, name='Constraint for calculating J_i of unifac')

    def eq_L(m, j, i):
        """

        :param m:
        :param j:
        :param i:
        :return:
        """
        return m.L[j, i] == m.q[i] / sum(m.q[i] * m.x[j, i] for i in m.i)

    model.eq_L = pyo.Constraint(model.dp, model.i, rule=eq_L, name='Constraint for calculating L_i of unifac')

    def eq_G(m, k, i):
        """

        :param m:
        :param k:
        :param j:
        :return:
        """
        return m.G[k, i] == m.v[i, k] * m.Qk[k]

    model.eq_G = pyo.Constraint(model.k, model.i, rule=eq_G, name='Constraint for calculating G_k,i of unifac')

    def eq_theta(m, dp, k):
        """

        :param m:
        :param i: dp
        :param j: k
        :return:
        """
        return m.theta[dp, k] == sum(m.G[k, i] * m.x[dp, i] for i in m.i)

    model.eq_theta = pyo.Constraint(model.dp, model.k, rule=eq_theta,
                                    name='Constraint for calculating theta_k of unifac')

    def eq_tau(m, dp, k, l):
        """

        :param dp:
        :param m:
        :param k:
        :param l:
        :return:
        """
        return m.tau[dp, k, l] == pyo.exp(-m.a[k, l] / m.T[dp])

    model.eq_tau = pyo.Constraint(model.dp, model.k, model.k, rule=eq_tau,
                                  name='Constraint for calculating tau_k,m of unifac')

    def eq_s(m, dp, k, i):
        """

        :param m:
        :param dp:
        :param k:
        :param i:
        :return:
        """
        return m.s[dp, k, i] == sum(m.G[l, i] * m.tau[dp, l, k] for l in m.k)

    model.eq_s = pyo.Constraint(model.dp, model.k, model.i, rule=eq_s,
                                name='Constraint for calculating s_k,i of unifac')

    def eq_eta(m, dp, k):
        """

        :param m:
        :param dp:
        :param k:
        :return:
        """
        return m.eta[dp, k] == sum(m.s[dp, k, i] * m.x[dp, i] for i in m.i)

    model.eq_eta = pyo.Constraint(model.dp, model.k, rule=eq_eta,
                                  name='Constraint for calculating eta_k of unifac')

    def eq_gamma_c(m, dp, i):
        """

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.gamma_c[dp, i] == 1 - m.J[dp, i] + pyo.log(m.J[dp, i]) - 5 * m.q[i] * (
                1 - m.J[dp, i] / m.L[dp, i] + pyo.log(m.J[dp, i] / m.L[dp, i]))

    model.eq_gamma_c = pyo.Constraint(model.dp, model.i, rule=eq_gamma_c,
                                      name='Constraint for calculating gammaC_i of unifac')

    def eq_gamma_r(m, dp, i):
        """

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.gamma_r[dp, i] == m.q[i] * (1 - pyo.log(m.L[dp, i])) - sum(
            m.theta[dp, k] * m.s[dp, k, i] / m.eta[dp, k] - m.G[k, i] * pyo.log(m.s[dp, k, i] / m.eta[dp, k]) for k in
            m.k)

    model.eq_gamma_r = pyo.Constraint(model.dp, model.i, rule=eq_gamma_r,
                                      name='Constraint for calculating gammaR_i of unifac')

    def eq_gamma(m, dp, i):
        """

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.gamma[dp, i] == pyo.exp(m.gamma_r[dp, i] + m.gamma_c[dp, i])

    model.eq_gamma = pyo.Constraint(model.dp, model.i, rule=eq_gamma, name='Constraint for calculating gamma')

    def eq_y(m, dp, i):
        """ function that calculates y
        :param m: pyomo model
        :param dp: data point index
        :param i: compound index
        :return:
        """
        return m.y[dp, i] == m.psat[dp, i] * model.x[dp, i] * m.gamma[dp, i] / m.P

    model.eq_y = pyo.Constraint(model.dp, model.i, rule=eq_y, name='calculation rule for vapour fraction')

    def eq_K(m, dp, i):
        """ Cons

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.K[dp, i] == m.gamma[dp, i] * m.psat[dp, i] / m.P

    model.eq_K = pyo.Constraint(model.dp, model.i, rule=eq_K, name='calculation rule for equilibrium constant')

    def eq_iso_vol(m):
        """

        :param m:
        :param dp:
        :return:
        """
        return 1.00 == m.K['iso', 0] / m.K['iso', 1]

    model.eq_iso_vol = pyo.Constraint(rule=eq_iso_vol, name='condition for iso-volatility')

    def eq_R(m):
        """

        :param m:
        :return:
        """
        return m.R == (m.K['min', 1] - m.ED) / (1.00 - m.K['min', 1])

    model.eq_R = pyo.Constraint(rule=eq_R, name='calculation of reflux ratio')

    def eq_ED(m):
        """

        :param m:
        :return:
        """
        return m.ED == 1 - ((1 - (m.K['min', 1])) / (m.x['min', 0] * (m.K['min', 0] - m.K['min', 1])))

    model.eq_ED = pyo.Constraint(rule=eq_ED, name='calculation of entrainer to distillate ratio')

    def con_x_min(m):
        """

        :param m:
        :return:
        """
        return m.x['min', 0] <= m.x['iso', 0] - 0.005

    model.con_x_min = pyo.Constraint(rule=con_x_min, name='xmin hs to be lower than the iso-volatility')

    def con_x_B(m, dp):
        """

        :param m:
        :param dp
        :return:
        """
        return m.x[dp, 1] == 0.0

    model.con_x_B = pyo.Constraint(model.dp, rule=con_x_B, name='xmin hs to be lower than the iso-volatility')

    model.obj = pyo.Objective(expr=model.ED, sense=pyo.minimize)

    # define the solver
    opt = pyo.SolverFactory('ipopt')
    # define solver options
    #opt.options["halt_on_ampl_error"] = "yes"
    opt.options["linear_solver"] = "MA27"
    opt.options["max_iter"] = 10000
    # solve the optimization problem
    solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True, logfile="pyo_vle_log")

    return solution, model



def iss_method_new(x0, T0, P, antoine_params, mixture, verbose):
    """ A wrapper function around the pyomo model for the ISS shortcut method for evalouating entrainer performance

    :param x0:
    :param T0:
    :param P:
    :param antoine_params:
    :param mixture:
    :return:
    """

    # create the model
    model = pyo.ConcreteModel()
    # define the parameters of the model
    n_comp = 3#x0.shape[1]
    n_data = 2#x0.shape[0]+1
    # create initialization for K
    k_init = {(0, 0): 0.7, #0: isovolatility 1: min
              (0, 1): 0.2,
              (0, 2): 0.7,
              (1, 0): 1.2,
              (1, 1): 0.6,
              (1, 2): 0.7}
    x_init= {(0, 0): 0.98,  # 0: isovolatility 1: min
            (0, 1): 0.02,
            (0, 2): 0.0,
            (1, 0): 0.5,
            (1, 1): 0.0,
            (1, 2): 0.5}
    # load the unifac parameters
    file_path = '../trash/unifac.xlsx'
    df_gc, df_a = load_gc(file_path, output_format='dataframe')
    # define sets
    model.k = pyo.Set(initialize=mixture.columns, name='Groups present')
    model.i = pyo.RangeSet(0, n_comp - 1, name=' no. of Compounds')
    model.dp = pyo.RangeSet(0, n_data - 1, name='no. of data points')
    model.grp = pyo.Set(initialize=df_a.columns, name='all Groups')

    # define the constants/parameters
    model.A = pyo.Param(model.i, initialize=antoine_params[:, 0], mutable=False, name='Antoine A', domain=pyo.Reals)
    model.B = pyo.Param(model.i, initialize=antoine_params[:, 1], mutable=False, name='Antoine B', domain=pyo.Reals)
    model.C = pyo.Param(model.i, initialize=antoine_params[:, 2], mutable=False, name='Antoine C', domain=pyo.Reals)
    model.a = pyo.Param(model.grp, model.grp, initialize=df_a.stack().to_dict(), mutable=False,
                        name='unifac interaction a_mk', domain=pyo.Reals)
    model.Rk = pyo.Param(model.grp, initialize=df_gc.Rk.to_dict(), mutable=False, name='unifac Rk parameters')
    model.Qk = pyo.Param(model.grp, initialize=df_gc.Qk.to_dict(), mutable=False, name='unifac Qk parameters')
    model.P = pyo.Param(initialize=P, mutable=False, name='Total pressure in MPa', domain=pyo.PositiveReals)
    model.v = pyo.Param(model.i, model.k, initialize=mixture.stack().to_dict(), mutable=False, name='compounds')
    # perform unifac evaluation for better initialization
    [r_init, q_init, j_init, L_init, g_init, theta_init, tau_init, s_init, eta_init, gamma_c_init, gamma_r_init,
     gamma_init] = unifac_init(mixture, x0, T0, df_gc, df_a)
    p_init = 10 ** (
                antoine_params[:, 0][:, None] - antoine_params[:, 1][:, None] / (antoine_params[:, 2][:, None] + T0))
    p_init = {(index[1], index[0]): v for index, v in np.ndenumerate(p_init)}
    model.r = pyo.Param(model.i, initialize=r_init, mutable=False, name='unifac r_i')
    model.q = pyo.Param(model.i, initialize=q_init, mutable=False, name='unifac q_i')
    model.G = pyo.Param(model.k, model.i, initialize=g_init, mutable=False, name='unifac G_ki')

    # define the model variables
    model.T = pyo.Var(model.dp, initialize=T0, within=pyo.PositiveReals, name='Temperature in K')
    model.J = pyo.Var(model.dp, model.i, initialize=j_init, within=pyo.PositiveReals, name='unifac J_i')
    model.L = pyo.Var(model.dp, model.i, initialize=L_init, within=pyo.PositiveReals, name='unifac L_i')
    model.theta = pyo.Var(model.dp, model.k, initialize=theta_init,within=pyo.PositiveReals, name='unifac theta_k')
    model.tau = pyo.Var(model.dp, model.k, model.k, initialize=tau_init, within=pyo.PositiveReals, name='unifac tau_mk')
    model.s = pyo.Var(model.dp, model.k, model.i, initialize=s_init, within=pyo.PositiveReals, name='unifac s_ki')
    model.eta = pyo.Var(model.dp, model.k, initialize=eta_init, within=pyo.PositiveReals, name='unifac eta_k')
    model.gamma_c = pyo.Var(model.dp, model.i, initialize=gamma_c_init, within=pyo.Reals, name='unifac combinatorial gamma')
    model.gamma_r = pyo.Var(model.dp, model.i, initialize=gamma_r_init, within=pyo.Reals, name='unifac residual gamma')
    model.gamma = pyo.Var(model.dp, model.i, initialize=gamma_init, within=pyo.Reals, name='unifac total gamma')
    model.psat = pyo.Var(model.dp, model.i, initialize=p_init, within=pyo.PositiveReals, name='vapour pressure in bar')
    model.y = pyo.Var(model.dp, model.i, initialize=0.33, within=pyo.PercentFraction, name='vapour composition y')
    model.x = pyo.Var(model.dp, model.i, initialize=x_init, name='Liquid composition x', within=pyo.PercentFraction)
    model.K = pyo.Var(model.dp, model.i, initialize=k_init, name='Equilibrium constants', within=pyo.PositiveReals)
    model.ED = pyo.Var(initialize=3.2, name='Entrainer to distillate ratio', within=pyo.PositiveReals)
    model.R = pyo.Var(initialize=3.5, name='Reflux ratio', within=pyo.PositiveReals)

    # define the constraints: god help me
    def sum_y(m, j):
        """
        Constrain for unity summation of the vapour fractions
        :param m: pyomo model
        :param j: index for datapoints
        :return: boolean
        """
        return sum(m.y[j, i] for i in m.i) == 1.0

    model.sum_y = pyo.Constraint(model.dp, rule=sum_y, name='Constraint for summation of the vapour fractions to unity')

    def sum_x(m, j):
        """
        Constrain for unity summation of the liquid fractions
        :param m: pyomo model
        :param j: index for datapoints
        :return: boolean
        """
        return sum(m.x[j, i] for i in m.i) == 1.0

    model.sum_x = pyo.Constraint(model.dp, rule=sum_x, name='Constraint for summation of the liquid fractions to unity')

    def eq_psat(m, j, i):
        """ Constraint for saturated vapour pressure equality
        :param m: pyomo model
        :param j: data point index
        :param i: compound index
        :return: boolean
        """
        return m.psat[j, i] == 10 ** (m.A[i] - m.B[i] / (m.C[i] + m.T[j]))

    model.eq_psat = pyo.Constraint(model.dp, model.i, rule=eq_psat, name='calculation rule for psat')


    def eq_J(m, j, i):
        """

        :param m:
        :param j:
        :param i:
        :return:
        """
        return m.J[j, i] == m.r[i] / sum(m.r[i] * m.x[j, i] for i in m.i)

    model.eq_J = pyo.Constraint(model.dp, model.i, rule=eq_J, name='Constraint for calculating J_i of unifac')

    def eq_L(m, j, i):
        """

        :param m:
        :param j:
        :param i:
        :return:
        """
        return m.L[j, i] == m.q[i] / sum(m.q[i] * m.x[j, i] for i in m.i)

    model.eq_L = pyo.Constraint(model.dp, model.i, rule=eq_L, name='Constraint for calculating L_i of unifac')


    def eq_theta(m, dp, k):
        """

        :param m:
        :param i: dp
        :param j: k
        :return:
        """
        return m.theta[dp, k] == sum(m.G[k, i] * m.x[dp, i] for i in m.i)

    model.eq_theta = pyo.Constraint(model.dp, model.k, rule=eq_theta,
                                    name='Constraint for calculating theta_k of unifac')

    def eq_tau(m, dp, k, l):
        """

        :param dp:
        :param m:
        :param k:
        :param l:
        :return:
        """
        return m.tau[dp, k, l] == pyo.exp(-m.a[k, l] / m.T[dp])

    model.eq_tau = pyo.Constraint(model.dp, model.k, model.k, rule=eq_tau,
                                  name='Constraint for calculating tau_k,m of unifac')

    def eq_s(m, dp, k, i):
        """

        :param m:
        :param dp:
        :param k:
        :param i:
        :return:
        """
        return m.s[dp, k, i] == sum(m.G[l, i] * m.tau[dp, l, k] for l in m.k)

    model.eq_s = pyo.Constraint(model.dp, model.k, model.i, rule=eq_s,
                                name='Constraint for calculating s_k,i of unifac')

    def eq_eta(m, dp, k):
        """

        :param m:
        :param dp:
        :param k:
        :return:
        """
        return m.eta[dp, k] == sum(m.s[dp, k, i] * m.x[dp, i] for i in m.i)

    model.eq_eta = pyo.Constraint(model.dp, model.k, rule=eq_eta,
                                  name='Constraint for calculating eta_k of unifac')

    def eq_gamma_c(m, dp, i):
        """

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.gamma_c[dp, i] == 1 - m.J[dp, i] + pyo.log(m.J[dp, i]) - 5 * m.q[i] * (
                1 - m.J[dp, i] / m.L[dp, i] + pyo.log(m.J[dp, i] / m.L[dp, i]))

    model.eq_gamma_c = pyo.Constraint(model.dp, model.i, rule=eq_gamma_c,
                                      name='Constraint for calculating gammaC_i of unifac')

    def eq_gamma_r(m, dp, i):
        """

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.gamma_r[dp, i] == m.q[i] * (1 - pyo.log(m.L[dp, i])) - sum(
            m.theta[dp, k] * m.s[dp, k, i] / m.eta[dp, k] - m.G[k, i] * pyo.log(m.s[dp, k, i] / m.eta[dp, k]) for k in
            m.k)

    model.eq_gamma_r = pyo.Constraint(model.dp, model.i, rule=eq_gamma_r,
                                      name='Constraint for calculating gammaR_i of unifac')

    def eq_gamma(m, dp, i):
        """

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.gamma[dp, i] == pyo.exp(m.gamma_r[dp, i] + m.gamma_c[dp, i])

    model.eq_gamma = pyo.Constraint(model.dp, model.i, rule=eq_gamma, name='Constraint for calculating gamma')

    def eq_y(m, dp, i):
        """ function that calculates y
        :param m: pyomo model
        :param dp: data point index
        :param i: compound index
        :return:
        """
        return m.y[dp, i] == m.psat[dp, i] * model.x[dp, i] * m.gamma[dp, i] / m.P

    model.eq_y = pyo.Constraint(model.dp, model.i, rule=eq_y, name='calculation rule for vapour fraction')

    def eq_K(m, dp, i):
        """ Cons

        :param m:
        :param dp:
        :param i:
        :return:
        """
        return m.K[dp, i] == m.gamma[dp, i] * m.psat[dp, i] / m.P

    model.eq_K = pyo.Constraint(model.dp, model.i, rule=eq_K, name='calculation rule for equilibrium constant')

    def eq_iso_vol(m):
        """

        :param m:
        :param dp:
        :return:
        """
        return 1.00 == m.K[0, 0] / m.K[0, 1]

    model.eq_iso_vol = pyo.Constraint(rule=eq_iso_vol, name='condition for iso-volatility')

    def eq_R(m):
        """

        :param m:
        :return:
        """
        return m.R == (m.K[1, 1] - m.ED) / (1.00 - m.K[1, 1])

    model.eq_R = pyo.Constraint(rule=eq_R, name='calculation of reflux ratio')

    def eq_ED(m):
        """

        :param m:
        :return:
        """
        return m.ED == 1 - ((1 - (m.K[1, 1])) / (m.x[1, 0] * (m.K[1, 0] - m.K[1, 1])))

    model.eq_ED = pyo.Constraint(rule=eq_ED, name='calculation of entrainer to distillate ratio')

    def con_x_min(m):
        """

        :param m:
        :return:
        """
        return m.x[1, 0] <= m.x[0, 0] - 0.005

    model.con_x_min = pyo.Constraint(rule=con_x_min, name='xmin hs to be lower than the iso-volatility')

    def con_x_B(m, dp):
        """

        :param m:
        :param dp
        :return:
        """
        return m.x[dp, 1] == 0.0

    model.con_x_B = pyo.Constraint(model.dp, rule=con_x_B, name='xmin hs to be lower than the iso-volatility')

    model.obj = pyo.Objective(expr=model.ED, sense=pyo.minimize)

    # define the solver
    opt = pyo.SolverFactory('ipopt')
    # define solver options
    #opt.options["halt_on_ampl_error"] = "yes"
    opt.options["linear_solver"] = "MA57"
    opt.options["max_iter"] = 1000
    #opt.options["expect_infeasible_problem"] = "yes"
    #opt.options["warm_start_init_point"] = "yes"
    # solve the optimization problem
    solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True, logfile="pyo_vle_log")

    return solution, model


########################################################################################################################
# Graveyard
########################################################################################################################


# def prop_calc(mol, prop_list, contrib):
#     """
#     This function calculates various thermochemical properties using the Constantinou-Gani model.
#
#     Parameters:
#     mol (pd.DataFrame): A pandas DataFrame with the occurrence of the groups.
#     prop_list (list): A list of properties to be calculated.
#     contrib (pd.DataFrame): A DataFrame of group contribution (GC) values for each property.
#
#     Returns:
#     y (pd.DataFrame): A DataFrame with all the calculated properties.
#
#     The function calculates the following properties:
#     - Mw: Molar weight in g/mol.
#     - Tm: Normal melting point in K.
#     - Tb: Normal boiling point in K.
#     - Tc: Critical temperature in K.
#     - Pc: Critical pressure in bar.
#     - Vc: Critical volume in cm3/mol.
#     - Gf: Gibbs energy of formation in kJ/mol.
#     - Hf: Enthalpy of formation in kJ/mol.
#     - Hv: Enthalpy of vaporization in kJ/mol.
#     - Vm: Molar volume in cm3/mol.
#     - Ac: Acentric factor (unitless).
#     - Zc: Critical compressibility factor (unitless).
#     - Vm_Tb: Molar volume at the boiling point in cm3/mol.
#     - tfc: Closed cup flash point in K.
#     - tfo: Open cup flash point in K.
#     - Hd: Hansen dispersion solubility parameter in MPa^1/2.
#     - Hp: Hansen polar solubility parameter in MPa^1/2.
#     - Hh: Hansen hydrogen solubility parameter in MPa^1/2.
#     - Solpar: Hildebrand solubility parameter in MPa^1/2.
#     - Visc: Viscosity in cp.
#     - LC50: Lethal concentration.
#     - Surf: Surface tension at 298K in dyn/cm.
#
#     The function uses the group contribution (GC) method,
#     where each group in a molecule contributes a certain amount to
#     the overall property of the molecule.
#     The contributions are summed up to calculate the property for the whole molecule.
#     """
#
#     # retrieve the groups and the occurrence
#     idx = mol.columns
#     mol_values = mol.values
#
#     # Ensure contrib DataFrame is aligned with mol's columns
#     contrib = contrib.loc[idx]
#
#     # initialize the dataframe
#     y = pd.DataFrame(index=mol.index)
#
#     # Vectorized summation for each property
#     y['Mw'] = np.sum(contrib['Mw'].values * mol_values, axis=1)
#     y['Tm'] = 102.425 * np.log(np.sum(contrib['Tm'].values * mol_values, axis=1))
#     y['Tb'] = 204.359 * np.log(np.sum(contrib['Tb'].values * mol_values, axis=1))
#     y['Tc'] = 181.128 * np.log(np.sum(contrib['Tc'].values * mol_values, axis=1))
#     y['Pc'] = 1 / (np.sum(contrib['Pc'].values * mol_values, axis=1) + 0.100220) ** 2 + 1.3705
#     y['Vc'] = -4.35 + 1000 * np.sum(contrib['Vc'].values * mol_values, axis=1)
#     y['Gf'] = -14.828 + np.sum(contrib['Gf'].values * mol_values, axis=1)
#     y['Hf'] = 10.835 + np.sum(contrib['Hf'].values * mol_values, axis=1)
#     y['Hv'] = 6.829 + np.sum(contrib['Hv'].values * mol_values, axis=1)
#     y['Vm'] = 12.11 + np.sum(1000 * contrib['Vm'].values * mol_values, axis=1)
#     y['Ac'] = 0.4085 * np.log(np.sum(contrib['Ac'].values * mol_values, axis=1) + 1.1507) ** 1.9802
#     y['Zc'] = y['Pc'] * y['Vc'] / (83.14 * y['Tc'])
#     y['Vm_Tb'] = 0.285 * y['Vc'] ** 1.048
#     y['tfc'] = -2.03 * np.sum(contrib['tfc'].values * mol_values, axis=1) + 0.659 * y['Tb'] + 20.00
#     y['tfo'] = 3.63 * np.sum(contrib['tfo'].values * mol_values, axis=1) + 0.409 * y['Tb'] + 88.43
#     y['Hd'] = np.sum(contrib['FDX'].values * mol_values, axis=1) / y['Vm']
#     y['Hp'] = np.sqrt(np.sum(contrib['FPX'].values ** 2 * mol_values, axis=1)) / y['Vm']
#     y['Hh'] = np.sqrt(np.sum(contrib['EHX'].values * mol_values, axis=1) / y['Vm'])
#     y['Solpar'] = np.sqrt((1000 * y['Hv'] - 8.314 * 298.15) / y['Vm'])
#     y['Visc'] = y['Mw'] * 1000 * np.exp((-597.82 + np.sum(contrib['EttaA'].values * mol_values, axis=1)) / 290 + np.sum(
#         contrib['EttaB'].values * mol_values, axis=1) - 11.202)
#     y['LC50'] = np.sum(contrib['LC50'].values * mol_values, axis=1)
#     y['Surf'] = 0.01707 * y['Solpar'] ** 2 * y['Vm'] ** 0.3333
#
#     return y[prop_list]


# def pyo_solve_antoine(Tb: float, Tc: float, Pc: float, P0: float, w: float, verbose: bool) -> dict:
#     """ Function that solves for the A, B and C parameters fom the antoine equation using pyomo
#
#     :param Tb: the melting point in Kelvin
#     :param Tc: the critical temperature in Kelvin
#     :param Pc: the critical pressure in MPa
#     :param P0: the atmospheric pressure in MPa
#     :param w: the acentric factor
#     :param verbose: printing of solver progress
#
#     :return: a dict containing: solver_stat: status of the solver
#                                 term_cond: termination condition of the solver
#                                 num_iter: number of iterations
#                                 obj_val: the objective values
#                                 A: Antoine parameter A
#                                 B: Antoine parameter B
#                                 C: Antoine parameter C
#     NB: the Antoine produces saturated pressures using the equation: P(MPa)=10 ** (A - B / (C + T))
#     """
#
#     """
#     Step 1: find the the saturated pressure at the acentric factor e.i. T=Tc*0.7
#     """
#     # print('------------- Finding the saturated pressure at at Tc*0.7 ----------')
#
#     # define the model type
#     model = pyo.ConcreteModel('Saturated pressure corresponding to the acentric factor')
#     # define the parameters
#     model.pc = pyo.Param(initialize=Pc, mutable=False, name='the critical pressure in MPa', domain=pyo.PositiveReals)
#     model.w = pyo.Param(initialize=w, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
#     # define the variable
#     model.pr = pyo.Var(within=pyo.PositiveReals, initialize=0.5 * Pc, name='reduced pressure at 0.7 Tc in MPa')
#
#     def eq_w(m):
#         """ Constraint for the equality of the acentric factor
#
#         :param m:
#         :return:
#         """
#         return m.w == - pyo.log10(m.pr / m.pc) - 1.00
#
#     model.eq_w = pyo.Constraint(rule=eq_w, name='calculation rule for acentric factor')
#
#     # define the solver
#     opt = pyo.SolverFactory('ipopt')
#     # define solver options
#     opt.options["halt_on_ampl_error"] = "yes"
#     opt.options["linear_solver"] = "MA27"
#     opt.options["max_iter"] = 10000
#     # ensurepyo_antoine_log exists
#     ensure_file_exists_in_cwd("pyo_antoine_log", "")
#     # solve the optimization problem
#     solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True, logfile="pyo_antoine_log")
#     # extract solution and solver stats
#     solv_stat = str(solution.solver.status)
#     term_cond = str(solution.solver.termination_condition)
#     Pw = model.pr.value
#     if solv_stat != 'ok' or term_cond != 'optimal':
#         print('Something is wrong with compound while solving for Pr ')
#
#     """
#     Step 2: find the initial guess for the Antoine parameters using selected-points method
#     """
#     # print('------------- Finding the initial guess for the Antoine parameters ----------')
#     # extract the points of interest
#     t1 = Tb
#     t2 = 0.7 * Tc
#     t3 = Tc
#     p1 = P0
#     p2 = Pw
#     p3 = Pc
#     # define the model type
#     model = pyo.ConcreteModel('System for generating initial guess for the antoine parameters')
#     # define the parameters
#     model.t1 = pyo.Param(initialize=t1, mutable=False, name='t1 or Tb in K', domain=pyo.PositiveReals)
#     model.t2 = pyo.Param(initialize=t2, mutable=False, name='t2 or Tr in K', domain=pyo.PositiveReals)
#     model.t3 = pyo.Param(initialize=t3, mutable=False, name='t3 or Tc in K', domain=pyo.PositiveReals)
#     model.y1 = pyo.Param(initialize=np.log10(p1), mutable=False, name='y1 or log10(Patm) in MPa', domain=pyo.Reals)
#     model.y2 = pyo.Param(initialize=np.log10(p2), mutable=False, name='y2 or log10(Pr) in MPa', domain=pyo.Reals)
#     model.y3 = pyo.Param(initialize=np.log10(p3), mutable=False, name='y3 or log10(Pc) in MPa', domain=pyo.Reals)
#     # define the variable
#     model.A = pyo.Var(within=pyo.Reals, initialize=10, name='Antoine Parameter A')
#     model.B = pyo.Var(within=pyo.Reals, initialize=0, name='Antoine Parameter B')
#     model.C = pyo.Var(within=pyo.Reals, initialize=0, name='Antoine Parameter C')
#
#     # define constraints
#
#     def eq_C(m):
#         """ Constraint for the equality of the C parameter for Antoine equation
#
#         :param m:
#         :return:
#         """
#         X = (m.y3 - m.y2) / (m.y2 - m.y1) * (m.t2 - m.t1) / (m.t3 - m.t2)
#         Y = m.t1 - m.t3
#         return m.C == Y / (X - 1) - m.t3
#
#     model.eq_C = pyo.Constraint(rule=eq_C, name='calculation rule for Antoine C')
#
#     def eq_B(m):
#         """ Constraint for the equality of the B parameter for Antoine equation
#
#         :param m:
#         :return:
#         """
#         return m.B == (m.y3 - m.y1) / (m.t3 - m.t1) * (m.t1 + m.C) * (m.t3 - m.C)
#
#     model.eq_B = pyo.Constraint(rule=eq_B, name='calculation rule for Antoine B')
#
#     def eq_A(m):
#         """ Constraint for the equality of the A parameter for Antoine equation
#
#         :param m:
#         :return:
#         """
#         return m.A == m.y2 + m.B / (m.t2 + m.C)
#
#     model.eq_A = pyo.Constraint(rule=eq_A, name='calculation rule for Antoine A')
#
#     # define the solver
#     opt = pyo.SolverFactory('ipopt')
#     # define solver options
#     opt.options["halt_on_ampl_error"] = "yes"
#     opt.options["linear_solver"] = "MA27"
#     opt.options["max_iter"] = 10000
#     # solve the optimization problem
#     solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True, logfile="pyo_vle_log")
#     # extract solution and solver stats
#     solv_stat = str(solution.solver.status)
#     term_cond = str(solution.solver.termination_condition)
#     A = model.A.value
#     B = model.B.value
#     C = model.C.value
#     if solv_stat != 'ok' or term_cond != 'optimal':
#         print('Something is wrong with compound while solving for Pr ')
#
#     """
#     Step 3: find the final value for the Antoine parameters
#     """
#     # print('------------- Finding the final values for the Antoine parameters ----------')
#
#     # define the model type
#     model = pyo.ConcreteModel('system of equation for solving the Antoine parameters')
#     # define the parameters
#     model.tb = pyo.Param(initialize=Tb, mutable=False, name='normal boiling point in K', domain=pyo.PositiveReals)
#     model.tc = pyo.Param(initialize=Tc, mutable=False, name='the critical temperature in K', domain=pyo.PositiveReals)
#     model.pc = pyo.Param(initialize=Pc, mutable=False, name='the critical pressure in MPa', domain=pyo.PositiveReals)
#     model.w = pyo.Param(initialize=w, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
#     model.p0 = pyo.Param(initialize=P0, mutable=False, name='acentric factor', domain=pyo.PositiveReals)
#     # define the model variables
#     model.A = pyo.Var(within=pyo.Reals, initialize=A, name='Antoine parameter A')
#     model.B = pyo.Var(within=pyo.Reals, initialize=B, name='Antoine parameter B')
#     model.C = pyo.Var(within=pyo.Reals, initialize=C, name='Antoine parameter C')
#     model.pr = pyo.Var(within=pyo.PositiveReals, initialize=10 ** (A - B / (C + Tc * 0.70)), name='Antoine parameter C')
#
#     # define the constraints
#     def eq_p0(m):
#         """ Constraint for the equality of the atmospheric pressure
#
#         :param m:
#         :return:
#         """
#         return pyo.log10(m.p0) == m.A - m.B / (m.C + m.tb)
#
#     model.eq_p0 = pyo.Constraint(rule=eq_p0, name='calculation rule for p atmospheric')
#
#     def eq_pc(m):
#         """ Constraint for the equality of the critical pressure
#
#         :param m:
#         :return:
#         """
#         return pyo.log10(m.pc) == m.A - m.B / (m.C + m.tc)
#
#     model.eq_pc = pyo.Constraint(rule=eq_pc, name='calculation rule for p critical')
#
#     def eq_w(m):
#         """ Constraint for the equality of the acentric factor
#
#         :param m:
#         :return:
#         """
#         pr = 10 ** (m.A - m.B / (m.C + m.tc * 0.70))
#         return m.w == - pyo.log10(pr / m.pc) - 1.00
#
#     model.eq_w = pyo.Constraint(rule=eq_w, name='calculation rule for acentric factor')
#     # define the solver
#     opt = pyo.SolverFactory('ipopt')
#     # show more stuff
#     opt.options["halt_on_ampl_error"] = "yes"
#     opt.options["linear_solver"] = "MA27"
#     opt.options["max_iter"] = 10000
#
#     # solve the optimization problem
#     solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True, logfile="pyomo_antoine_log")
#
#     solv_stat = str(solution.solver.status)
#     term_cond = str(solution.solver.termination_condition)
#     if solv_stat != 'ok' or term_cond != 'optimal':
#         print('Something is wrong with compound while solving for pr ')
#
#     output = {'solv_stat': str(solution.solver.status),
#               'term_cond': str(solution.solver.termination_condition),
#               'num_iter': parse_pyomo_log('pyomo_antoine_log'),
#               'A': model.A.value,
#               'B': model.B.value,
#               'C': model.C.value}
#
#     return output

# def unifac(mixture: pd.DataFrame, x0: float, T0: float, df_gc: pd.DataFrame, df_a: pd.DataFrame) -> np.array:
#     """ function to perform pt-flash calculations
#
#     :param mixture: dataframe with the compounds and rspective groups considered
#     :param x0: np.ndarray of composition
#     :param T0: temperature
#     :param df_gc: dataframe of group contribution parameter
#     :param df_a: dataframe of interaction parameters
#     :return: gamma: liquid activity coefficients
#     """
#     v = np.array(mixture)
#     Rk = np.array(df_gc['Rk'][mixture.columns])  # np.array(df_gc['Rk'][df_mix.columns])
#     Qk = np.array(df_gc['Qk'][mixture.columns])
#     a = np.array(df_a[mixture.columns].loc[mixture.columns])
#     # begin calculations
#     r = np.dot(v, Rk)
#     q = np.dot(v, Qk)
#     j = np.mat(r) / np.mat(np.dot(x0, r)).T
#     L = np.mat(q) / np.mat(np.dot(x0, q)).T
#     g = v * Qk
#     theta = np.dot(x0, g)
#     tau = np.exp(-a / T0)
#     s = np.dot(g, tau)
#     eta = np.dot(x0, s)
#     gamma_c = 1 - j + np.log(j) - 5 * np.multiply(q, (1 - j / L + np.log(j / L)))
#     gamma_r = np.multiply(q, (1 - np.log(L))) - np.sum(
#         np.multiply(theta[:, None, :], s / eta[:, None, :]) - g * np.log(s / eta[:, None, :]), axis=2)
#     gamma = np.array(np.exp(gamma_c + gamma_r))
#     return gamma  # , [r, q, j, L, g, theta, tau, s, eta, gamma_c, gamma_r, gamma]
