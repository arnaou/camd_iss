# Project: camd_iss
#
# Title: testing the group-contribution calculation of pure component properties
#
# Author: Adem Rosenkvist Nielsen Aouichaoui
#
# Email: arnaou@kt.dtu.dk
########################################################################################################################

# import paths
from tests import _PATH_DATA

# import packages
import pandas as pd
import numpy as np
from utils.gc import load_gc, prop_calc
import pytest

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

@pytest.mark.parametrize("df_comps, target",
                         [(df_comps, np.array([273.15, 291.65, 176.00, 260.15, 175.55, 211.87]))])
def test_melting_point(df_comps, target):
    # Calculate the melting points
    pred = prop_calc(df_comps, ['Tm'], df_gc).to_numpy().round(2).flatten()
    # Assertion
    assert all(pred[i] == target[i] for i in range(len(pred))), "issue with melting point calculation"

@pytest.mark.parametrize("df_comps, target",
                         [(df_comps, np.array([373.15, 463.1, 337.85, 470.45, 379.07, 341.35]))])
def test_boiling_point(df_comps, target):
    # Calculate the melting points
    pred = prop_calc(df_comps, ['Tb'], df_gc).to_numpy().round(2).flatten()
    # Assertion
    assert all(pred[i] == target[i] for i in range(len(pred))), "issue with boiling point calculation"

@pytest.mark.parametrize("df_comps, target",
                         [(df_comps, np.array([647.00, 729.00, 513.00, 719.7, 549.86, 516.51]))])

def test_critical_point_T(df_comps, target):
    # Calculate the melting points
    pred = prop_calc(df_comps, ['Tc'], df_gc).to_numpy().round(2).flatten()
    # Assertion
    assert all(pred[i] == target[i] for i in range(len(pred))), "issue with critical temperature calculation"


@pytest.mark.parametrize("df_comps, target",
                         [(df_comps, np.array([220.64, 56.50, 81.00, 85.38, 28.27, 39.96]))])
def test_critical_point_P(df_comps, target):
    # Calculate the melting points
    pred = prop_calc(df_comps, ['Pc'], df_gc).to_numpy().round(2).flatten()
    # Assertion
    assert all(pred[i] == target[i] for i in range(len(pred))), "issue with critical pressure calculation"

@pytest.mark.parametrize("df_comps, target",
                         [(df_comps, np.array([0.34, 0.28, 0.56, 0.49, 0.34, 0.33]))])
def test_critical_point_V(df_comps, target):
    # Calculate the melting points
    pred = prop_calc(df_comps, ['Ac'], df_gc).to_numpy().round(2).flatten()
    # Assertion
    assert all(pred[i] == target[i] for i in range(len(pred))), "issue with critical volume calculation"

@pytest.mark.parametrize("df_comps, target",
                         [(df_comps, np.array([43.99, 52.46, 37.47, 66.26, 38.31, 34.19]))])
def test_enthalpy_vaporization(df_comps, target):
    # Calculate the melting points
    pred = prop_calc(df_comps, ['Hv'], df_gc).to_numpy().round(2).flatten()
    # Assertion
    assert all(pred[i] == target[i] for i in range(len(pred))), "issue with enthalpy of vaporization calculation"

