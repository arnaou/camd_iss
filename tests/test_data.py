########################################################################################################################
# Project: camd_iss
#
# Title: testing data loading, structure and availability
#
# Author: Adem Rosenkvist Nielsen Aouichaoui
#
# Email: arnaou@kt.dtu.dk
########################################################################################################################


# import python native packages
import pandas as pd

# import custom packages
from utils.gc import load_gc, inspect_interractions, inspect_contribs
from tests import _PATH_DATA

def test_loading_group_contributions():
    df_gc, df_a = load_gc(_PATH_DATA+'/data_unifac.xlsx', 'dataframe')
    gc_rows,gc_cols = df_gc.shape
    a_row, a_cols = df_a.shape
    assert isinstance(df_gc, pd.DataFrame), "data for group contributions and R is not a dataframe"
    assert isinstance(df_a, pd.DataFrame), "data for unifac interactions is not a dataframe"
    assert gc_rows == 119, "number of groups for the group-contribution is not 119"
    assert gc_cols == 26
    assert a_row == a_cols == 119, "number of groups for the unifac data is not 119"
    assert all(dtype in ['float64', 'int64'] for dtype in set(df_gc.dtypes)), "gc data is not float or integers"
    assert all(dtype in ['float64'] for dtype in set(df_a.dtypes)), "unifac data is not float"



def test_unifac_data_availability():
    # load the interaction parameters
    _, df_a = load_gc(_PATH_DATA + '/data_unifac.xlsx', 'dataframe')
    # a select subset of groups
    grp_lst = ['CH3', 'CH2', 'ACH', 'AC-Br', 'C4H4S', 'CH2=CH']
    # select ubset of the matrix
    matrix = df_a.loc[grp_lst,grp_lst]
    mg,sg = inspect_interractions(matrix, verbose=False)

    assert len(mg) == len(sg) == 5, " no. of missing interactions is wrong"
    for i in range(len(mg)):
        assert matrix.loc[sg[i][0],sg[i][1]] == 0, "indicated missing group has a non 0 interaction"


def test_gc_data_availability():
    # load the interaction parameters
    df_gc, _ = load_gc(_PATH_DATA + '/data_unifac.xlsx', 'dataframe')
    # a select subset of groups
    grp_lst = ['CH3','CF2', 'SiO', 'COC', 'CH3', 'CH2']
    # select a subset of properties
    prop_lst = ["Tm", "Tb", "Pc"]
    # check missing contribs
    missing_contribs = inspect_contribs(df_gc.loc[grp_lst, prop_lst], verbose=False)
    assert missing_contribs['Tm'] == ["CF2", "SiO", "COC"]
    assert missing_contribs['Tb'] == ["SiO", "COC"]
    assert missing_contribs['Pc'] == ["SiO", "COC"]


















# # define a mixture
# mix_dict = {'CH3': [2, 2],
#             'CH2': [5, 1],
#             'CH2NH': [0, 1]
#             }
# tags = ['HEP', 'DEA']
# df_mix = construct_mixture(tags, mix_dict)
#
# """
# test 1: is the data loaded?
# """
# if df_a.empty or df_gc.empty or df_mix.empty:
#     print("one of the DataFrame is empty")
#
# """
# test 2: are the data loaded available too? do we risk choosing groups that are missing some parameters?
# it should raise a warning to indicate that some groups not of the same main group are missing interactions
# and that these have been set to 0. these groups should then be shown.
# """
# # select a list of properties
# prop_list = ['Tb', 'Tc', 'Ac', 'Pc', 'Tm']
# grp_list = ['CH3', 'CH3CO', 'CH3OH', 'H2O', 'DMSO', 'DOH', 'CH2', 'OH', 'CH', 'CH2CO', 'CHNH2']
#
# df_test = df_a.loc[grp_list,grp_list]
# # we inspect the interaction fo the unifac
# miss_int = inspect_interractions(df_test)
# print(f' following groups have missing unifac interactions {miss_int}. These have been set to 0')
#
# # we inspect the contributions of the selected groups and properties
# miss_contrib=inspect_contribs(df_gc.loc[grp_list, prop_list])
# print(f' following groups have missing pure component contributions{miss_contrib}')
