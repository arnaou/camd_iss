########################################################################################################################
#   Test units to ensure that data are read correctly.
#
#
#
#
#
#
#   Adem R.N. Aouichaoui, arnaou@kt.dtu.dk
########################################################################################################################
import numpy as np

# import packages
from utils.gc import load_gc

# define data path
file_path = '../data/unifac.xlsx'
# load the data
df_gc, df_unifac = load_gc(file_path, output_format='dataframe')


def test_gc_parameters():
    """ Function that tests the loading procedure of the Constantinou-Gani group-contributions

    :return:
    """
    # define data path
    file_path = 'data/unifac.xlsx' # this path is relative to the root directory
    # load the data
    df_gc, _ = load_gc(file_path, output_format='dataframe')
    # perform tests on random properties and groups
    assert df_gc.loc['CH3']['Tc'] == 1.67810  # critical point
    assert df_gc.loc['CH2']['Mw'] == 14.02600  # molar weight
    assert df_gc.loc['C']['Tm'] == 1.64790  # melting point
    assert df_gc.loc['OH']['Tb'] == 3.2152  # boiling point
    assert df_gc.loc['AC']['Pc'] == 0.002136  # critical point
    assert df_gc.loc['ACCH']['Vc'] == 0.07120  # critical volume
    assert df_gc.loc['HCO']['Gf'] == -107.858  # gibbs free energy
    assert df_gc.loc['C=C']['Hf'] == 104.293  # energy of formation
    assert df_gc.loc['CH=C']['Hv'] == 8.178  # enthalpy of vaporization
    assert df_gc.loc['CH']['Ac'] == -0.07063  # acentric factor
    assert df_gc.loc['CH2=CH']['Vm'] == 0.03727  # molar volume

#
#
def test_unifac_parameters():
    """

    :return:
    """
    # define data path
    file_path = 'data/unifac.xlsx'  # this path is relative to the root directory
    # load the data
    df_gc, df_unifac = load_gc(file_path, output_format='dataframe')
    # perform tests
    assert df_gc.loc['CH2']['Rk'] == 0.6744
    assert df_gc.loc['CH2']['Qk'] == 0.54
    assert df_unifac.loc[['CH2','ACH','ACCH2', 'OH', 'CH2NH']][['CH2','ACH','ACCH2', 'OH', 'CH2NH']].values == np.array()



#
#
# def test_data_format():


# TODO
# make sure dict and dataframe are identical
# make sure unifac params are correct
# make sure cg parameters are correct
