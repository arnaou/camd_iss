# Project: camd_iss
#
# Title: testing the calculations pf pure component properties group-contribution calculation
#
# Author: Adem Rosenkvist Nielsen Aouichaoui
#
# Email: arnaou@kt.dtu.dk
########################################################################################################################

# import packages
import pandas as pd
from utils.gc import load_gc, prop_calc, inspect_contribs


# load the data
df_gc, _ = load_gc('../data/data_unifac.xlsx', 'dataframe')
# loading the calculations from propred
df_propred = pd.read_excel('../data/propred_calculation.xlsx')
df_propred.set_index('property', inplace=True)
df_propred.index.name = None
# extract property list
prop_list =df_propred.index.to_list()

# define the mixture: heptane: CCCCCCC & diethylamine: CCNCC
df_mix = pd.DataFrame({'CH3': [2, 2],
                       'CH2': [5, 1],
                       'CH2NH': [0, 1]})
df_mix.index = ['HEP', 'DEA']

# inspect missing contributions
grp_list = df_mix.columns
print(inspect_contribs(df_gc.loc[grp_list,:]))

# calculate the properties
df_props = prop_calc(df_mix, prop_list, df_gc)
df_props = df_props.T

# calculate difference
df_diff = df_propred-df_props
df_diff




