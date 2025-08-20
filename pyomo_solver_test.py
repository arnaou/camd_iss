from pyomo.environ import *

V = 40     # liters
kA = 0.5   # 1/min
kB = 0.1   # l/min
CAf = 2.0  # moles/liter

# create a model instance
model = ConcreteModel()

# create x and y variables in the model
model.q = Var()

# add a model objective
model.objective = Objective(expr = model.q*V*kA*CAf/(model.q + V*kB)/(model.q + V*kA), sense=maximize)


opt = SolverFactory('ipopt')
# opt.options["halt_on_ampl_error"] = "yes"
# opt.options["hsllib"] = "../libHSL_binaries.v2025.7.21/lib/libhsl.dylib"
# opt.options["linear_solver"] = "MA57"
opt.options["max_iter"] = 10000
opt.options['print_level'] = 5
opt.options['output_file'] = "my_ipopt_log.txt"
# compute a solution using ipopt for nonlinear optimization
results = opt.solve(model, tee=True)#, symbolic_solver_labels=True)
# model.pprint()


# # print solutions
# qmax = model.q()
# CBmax = model.objective()
# print('\nFlowrate at maximum CB = ', qmax, 'liters per minute.')
# print('\nMaximum CB =', CBmax, 'moles per liter.')
# print('\nProductivity = ', qmax*CBmax, 'moles per minute.')