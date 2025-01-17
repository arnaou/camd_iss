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

# import custom packages
from utils.gc import load_gc, construct_mixture, inspect_interractions, inspect_contribs


# load the data
df_gc, df_a = load_gc('../data/data_unifac.xlsx', 'dataframe')

# define a mixture
mix_dict = {'CH3': [2, 2],
            'CH2': [5, 1],
            'CH2NH': [0, 1]
            }
tags = ['HEP', 'DEA']
df_mix = construct_mixture(tags, mix_dict)

"""
test 1: is the data loaded?
"""
if df_a.empty or df_gc.empty or df_mix.empty:
    print("one of the DataFrame is empty")

"""
test 2: are the data loaded available too? do we risk choosing groups that are missing some parameters?
it should raise a warning to indicate that some groups not of the same main group are missing interactions
and that these have been set to 0. these groups should then be shown.
"""
# select a list of properties
prop_list = ['Tb', 'Tc', 'Ac', 'Pc', 'Tm']
grp_list = ['CH3', 'CH3CO', 'CH3OH', 'H2O', 'DMSO', 'DOH', 'CH2', 'OH', 'CH', 'CH2CO', 'CHNH2']

df_test = df_a.loc[grp_list,grp_list]
# we inspect the interaction fo the unifac
miss_int = inspect_interractions(df_test)
print(f' following groups have missing unifac interactions {miss_int}. These have been set to 0')

# we inspect the contributions of the selected groups and properties
miss_contrib=inspect_contribs(df_gc.loc[grp_list, prop_list])
print(f' following groups have missing pure component contributions{miss_contrib}')
