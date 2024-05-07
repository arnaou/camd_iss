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


def load_gc(file_path, output_format):
    """ function that load the unifac parameters as well as the Constantinou-Gani group contributions
     and return them as either a dataframe or as a dict of dicts

    :param file_path: the path to the file containing the data
    :param output_format: should the output be a dataframe or a dict of dicts
    :return: pd.Dataframe or dict
    """
    # construct the dataframe for gc
    gc_df = pd.read_excel(file_path, sheet_name='set')
    gc_df['noc'] = pd.read_excel(file_path, sheet_name='noc', usecols="B")
    gc_df['Rk'] = pd.read_excel(file_path, sheet_name='Rk', usecols="B")
    gc_df['Qk'] = pd.read_excel(file_path, sheet_name='Qk', usecols="B")
    gc_df['Mw'] = pd.read_excel(file_path, sheet_name='Mw', usecols="B")
    gc_df['Tm'] = pd.read_excel(file_path, sheet_name='Tm', usecols="B")
    gc_df['Tb'] = pd.read_excel(file_path, sheet_name='Tb', usecols="B")
    gc_df['Tc'] = pd.read_excel(file_path, sheet_name='Tc', usecols="B")
    gc_df['Pc'] = pd.read_excel(file_path, sheet_name='Pc', usecols="B")
    gc_df['Vc'] = pd.read_excel(file_path, sheet_name='Vc', usecols="B")
    gc_df['Gf'] = pd.read_excel(file_path, sheet_name='Gf', usecols="B")
    gc_df['Hf'] = pd.read_excel(file_path, sheet_name='Hf', usecols="B")
    gc_df['Hv'] = pd.read_excel(file_path, sheet_name='Hv', usecols="B")
    gc_df['Ac'] = pd.read_excel(file_path, sheet_name='Ac', usecols="B")
    gc_df['Vm'] = pd.read_excel(file_path, sheet_name='Vm', usecols="B")
    gc_df['Gtt'] = pd.read_excel(file_path, sheet_name='Gtt', usecols="B")
    gc_df['LogP'] = pd.read_excel(file_path, sheet_name='LogP', usecols="B")
    gc_df['LogWs'] = pd.read_excel(file_path, sheet_name='LogWs', usecols="B")
    gc_df['tfc'] = pd.read_excel(file_path, sheet_name='tfc', usecols="B")
    gc_df['tfo'] = pd.read_excel(file_path, sheet_name='tfo', usecols="B")
    gc_df['ccsub'] = pd.read_excel(file_path, sheet_name='ccsub', usecols="B")
    gc_df['Hd'] = pd.read_excel(file_path, sheet_name='FDX', usecols="B")
    gc_df['Hp'] = pd.read_excel(file_path, sheet_name='FPX', usecols="B")
    gc_df['Hh'] = pd.read_excel(file_path, sheet_name='EHX', usecols="B")
    gc_df['EttaA'] = pd.read_excel(file_path, sheet_name='EttaA', usecols="B")
    gc_df['EttaB'] = pd.read_excel(file_path, sheet_name='EttaB', usecols="B")
    gc_df['Lc'] = pd.read_excel(file_path, sheet_name='LC50', usecols="B")
    gc_df.set_index('set', inplace=True)
    gc_df.rename_axis(None, inplace=True)
    # construct dataframe for interaction params
    a_df = pd.read_excel(file_path, sheet_name='a', index_col=0)
    if output_format == 'dataframe':
        return gc_df, a_df
    else:
        # convert into dicts
        gc_dict = gc_df.stack().to_dict()
        a_dict = a_df.stack().to_dict()
        return gc_dict, a_dict


def prop_calc(mol, prop_list, contrib):
    """ Function that calculates various thermochemical properties using the Constantinou-Gani model

    :param mol: A pandas dataframe with the occurrence of the groups
    :param prop_list:  A list of properties
    :param contrib: A dataframe of GC contributions
    :return: y: A dataframe with all the properties

    # additional info: units
    mw: molar weight in mol/g
    tm: normal melting point in K
    tb: normal boiling point in K
    tc: critical temperature in K
    pc: critical pressure in bar
    vc: critical volume in cm3/mol
    gf: gibbs energy of formation kJ/mol
    hv: enthalpy of formation kJ/mol
    hv: enthalpy of vaporization kJ/mol
    vm: molar volume cm3/mol
    acc: acentric factor in  unitless
    zc: critical compressibility factor in unitless
    v_tb: molar volume at the boiling point cm3/mol
    tfc: closed cup flash point in K
    tfo: open cup flash point in K
    Ds: hansen dispersion solubility parameter MPa^1/2
    Ps: hansen polar solubility parameter MPa^1/2
    Hs: hansen hydrogen solubility parameter MPa^1/2
    solpar: hildebrand solubility parameter MPa^1/2
    visc: viscosity in cp
    lc50: lethal concentration
    surf: surface tension at 298K in dyn/cm
    """

    idx = mol.columns
    y = pd.DataFrame()
    y['mw'] = sum(contrib.Mw[i] * mol[i] for i in idx)
    y['tm'] = 102.425 * np.log(sum(contrib.Tm[i] * mol[i] for i in idx))
    y['tb'] = 204.359 * np.log(sum(contrib.Tb[i] * mol[i] for i in idx))
    y['tc'] = 181.128 * np.log(sum(contrib.Tc[i] * mol[i] for i in idx))
    y['pc'] = 1 / (sum(contrib.Pc[i] * mol[i] for i in idx) + 0.100220) ** 2 + 1.3705
    y['vc'] = -4.35 + 1000 * sum(contrib.Vc[i] * mol[i] for i in idx)
    y['gf'] = -14.828 + sum(contrib.Gf[i] * mol[i] for i in idx)
    y['hf'] = 10.835 + sum(contrib.Hf[i] * mol[i] for i in idx)
    y['hv'] = 6.829 + sum(contrib.Hv[i] * mol[i] for i in idx)
    y['vm'] = 12.11 + sum(1000 * contrib.Vm[i] * mol[i] for i in idx)
    y['ac'] = 0.4085 * np.log(sum(contrib.Ac[i] * mol[i] for i in idx) + 1.1507) ** 1.9802
    y['zc'] = y['pc'] * y['vc'] / (83.14 * y['tc'])
    y['vm_tb'] = 0.285 * y['vc'] ** 1.048
    y['tfc'] = -2.03 * sum(contrib.tfc[i] * mol[i] for i in idx) + 0.659 * y['tb'] + 20.00
    y['tfo'] = 3.63 * sum(contrib.tfo[i] * mol[i] for i in idx) + 0.409 * y['tb'] + 88.43
    y['Ds'] = (sum(contrib.Hd[i] * mol[i] for i in idx)) / y['vm']
    y['Ps'] = (sum(contrib.Hp[i] * contrib.Hp[i] * mol[i] for i in idx)) ** 0.5 / y['vm']
    y['Hs'] = (sum(contrib.Hh[i] * mol[i] for i in idx) / y['vm']) ** 0.5
    y['solpar'] = ((1000 * y['hv'] - 8.314 * 298.15) / y['vm']) ** 0.5
    y['visc'] = y['mw'] * 1000 * np.exp((-597.82 + sum(contrib.EttaA[i] * mol[i] for i in idx)) / 290 + sum(
        contrib.EttaB[i] * mol[i] for i in idx) - 11.202)
    y['lc50'] = sum(contrib.Lc[i] * mol[i] for i in idx)
    y['surf'] = 0.01707*y['SolPar']**2*y['vm']**0.3333
    return y[prop_list]
