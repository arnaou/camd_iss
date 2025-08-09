# Project: camd_iss
#
# Title: testing the  installation of Pyomo and additional solvers
#
# Author: Adem Rosenkvist Nielsen Aouichaoui
#
# Email: arnaou@kt.dtu.dk
########################################################################################################################


########################################################################################################################
#%% Import packages
########################################################################################################################
from pyomo.environ import  Var, NonNegativeReals, ConcreteModel, Objective, maximize, Constraint, SolverFactory

########################################################################################################################
#%% Define functions
########################################################################################################################

def test_pyomo():
    """
    unit test to check if pyomo is installed correctly
    test from here:
    https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/02.01-Production-Models-with-Linear-Constraints.html
    :return:
    """
    model = ConcreteModel()

    # declare decision variables
    model.x = Var(domain=NonNegativeReals)

    # declare objective
    model.profit = Objective(
        expr=40 * model.x,
        sense=maximize)

    # declare constraints
    model.demand = Constraint(expr=model.x <= 40)
    model.laborA = Constraint(expr=model.x <= 80)
    model.laborB = Constraint(expr=2 * model.x <= 100)

    # solve
    SolverFactory('glpk').solve(model).write()

    print(f"Profit = {model.profit()} per week") #1600.0
    print(f"X = {model.x()} units per week") #40.0

    assert model.profit() == 1600.0
    assert model.x() == 40.0

def test_ipopt():
    """
    unit test to check if some of the solvers are installed correctly
    :return:
    """

    assert SolverFactory('ipopt').available() == True
    opt = SolverFactory('ipopt')
    opt.options["linear_solver"] = "MA57"
    opt.options["hsllib"] = "libcoinhsl.so"

test_pyomo()
test_ipopt()


