########################################################################################################################
# Project: camd_iss
#
# Title: UNIFAC Bubble Temperature Calculator Class
#
# Author: Adem Rosenkvist Nielsen Aouichaoui
#
# Email: arnaou@kt.dtu.dk
########################################################################################################################
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from typing import Dict, Any, Tuple, Union
import json
import os
import warnings


class UnifacBubbleTCalculator:
    """
    A class for conducting bubble temperature calculations using UNIFAC method with Pyomo optimization.
    
    This class loads UNIFAC parameters from JSON files and provides a clean interface for bubble T calculations.
    """
    
    def __init__(self, data_path='data/json', solver='ipopt', solver_options=None):
        """
        Initialize the UNIFAC Bubble Temperature Calculator.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing JSON data files
        solver : str
            Solver to use for optimization (default: 'ipopt')
        solver_options : dict
            Dictionary of solver options
        """
        self.data_path = data_path
        self.solver_name = solver
        self.solver_options = solver_options or self._default_solver_options()
        
        # Load UNIFAC parameters from JSON files
        self._load_unifac_data()
        
        # Initialize model and solution storage
        self.model = None
        self.solution = None
        
    def _default_solver_options(self):
        """Default solver options for IPOPT."""
        return {
            #"linear_solver": "mumps",
            "max_iter": 3000,
            "tol": 1e-6,
            "constr_viol_tol": 1e-6,
            "acceptable_tol": 1e-4,
            "acceptable_constr_viol_tol": 1e-4
        }
    
    def _load_unifac_data(self):
        """Load UNIFAC parameters from JSON files."""
        # Load Rk parameters
        with open(os.path.join(self.data_path, 'rk.json'), 'r') as f:
            rk_data = json.load(f)
        self.rk_params = rk_data['data']
        
        # Load Qk parameters
        with open(os.path.join(self.data_path, 'qk.json'), 'r') as f:
            qk_data = json.load(f)
        self.qk_params = qk_data['data']
        
        # Load interaction parameters
        with open(os.path.join(self.data_path, 'a.json'), 'r') as f:
            a_data = json.load(f)
        self.a_params = a_data['data']
        
    def _unifac_init(self, mixture, x, T0):
        """
        Initialize UNIFAC calculations for better starting point.
        
        Parameters:
        -----------
        mixture : pandas.DataFrame
            Mixture groups as columns
        x : numpy.ndarray
            Liquid composition matrix
        T0 : float or numpy.ndarray
            Initial temperature guess
            
        Returns:
        --------
        dict
            Dictionary containing initialized UNIFAC parameters
        """
        n_comp = x.shape[1]
        n_data = x.shape[0]
        
        # Calculate r and q for each component
        r = np.zeros(n_comp)
        q = np.zeros(n_comp)
        
        for i in range(n_comp):
            for group in mixture.columns:
                v_ik = mixture.iloc[i][group]
                if v_ik > 0:
                    if group in self.rk_params and group in self.qk_params:
                        r[i] += v_ik * self.rk_params[group]
                        q[i] += v_ik * self.qk_params[group]
                    else:
                        print(f"Warning: Group '{group}' not found in UNIFAC parameters")
        
        # Calculate J and L
        J = np.zeros((n_data, n_comp))
        L = np.zeros((n_data, n_comp))
        
        for dp in range(n_data):
            r_sum = np.sum(x[dp, :] * r)
            q_sum = np.sum(x[dp, :] * q)
            
            for i in range(n_comp):
                J[dp, i] = r[i] / r_sum
                L[dp, i] = q[i] / q_sum
        
        # Calculate group activity coefficients (simplified initialization)
        g = {}
        for group in mixture.columns:
            for i in range(n_comp):
                g[(group, i)] = mixture.iloc[i][group] if mixture.iloc[i][group] > 0 else 0
        
        # Calculate theta
        theta = {}
        for dp in range(n_data):
            for group in mixture.columns:
                theta_sum = 0
                for i in range(n_comp):
                    theta_sum += x[dp, i] * g.get((group, i), 0)
                theta[(dp, group)] = theta_sum
        
        # Initialize tau (temperature dependent)
        tau = {}
        for dp in range(n_data):
            T_val = T0[dp] if hasattr(T0, '__len__') else T0
            for group1 in mixture.columns:
                for group2 in mixture.columns:
                    a_val = self.a_params.get(group1, {}).get(group2, 0)
                    tau[(dp, group1, group2)] = np.exp(-a_val / T_val)
        
        # Initialize other variables (simplified)
        s = {(dp, group, i): 1.0 for dp in range(n_data) for group in mixture.columns for i in range(n_comp)}
        eta = {(dp, group): 1.0 for dp in range(n_data) for group in mixture.columns}
        gamma_c = {(dp, i): 1.0 for dp in range(n_data) for i in range(n_comp)}
        gamma_r = {(dp, i): 1.0 for dp in range(n_data) for i in range(n_comp)}
        gamma = {(dp, i): 1.0 for dp in range(n_data) for i in range(n_comp)}
        
        return {
            'r': {i: r[i] for i in range(n_comp)},
            'q': {i: q[i] for i in range(n_comp)},
            'J': {(dp, i): J[dp, i] for dp in range(n_data) for i in range(n_comp)},
            'L': {(dp, i): L[dp, i] for dp in range(n_data) for i in range(n_comp)},
            'g': g,
            'theta': theta,
            'tau': tau,
            's': s,
            'eta': eta,
            'gamma_c': gamma_c,
            'gamma_r': gamma_r,
            'gamma': gamma
        }
    
    def setup_model(self, x, T0, P, antoine_params, mixture):
        """
        Set up the Pyomo model for bubble temperature calculation.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Liquid composition matrix (n_data x n_comp)
        T0 : float or numpy.ndarray
            Initial guess for saturation temperature in Kelvin
        P : float
            Total pressure in MPa
        antoine_params : numpy.ndarray
            Antoine parameters array (n_comp x 3)
        mixture : pandas.DataFrame
            Mixture groups as columns
        """
        # Create the model
        self.model = pyo.ConcreteModel()
        
        # Define model dimensions
        n_comp = x.shape[1]
        n_data = x.shape[0]
        
        # Define sets
        self.model.k = pyo.Set(initialize=mixture.columns, name='Groups present')
        self.model.i = pyo.RangeSet(0, n_comp - 1, name='no. of Compounds')
        self.model.dp = pyo.RangeSet(0, n_data - 1, name='no. of data points')
        self.model.grp = pyo.Set(initialize=list(self.a_params.keys()), name='all Groups')
        
        # Define constants/parameters
        self._setup_parameters(x, P, antoine_params, mixture)
        
        # Perform UNIFAC initialization
        init_data = self._unifac_init(mixture, x, T0)
        
        # Initialize pressure and composition
        if hasattr(T0, '__len__'):
            T_vals = T0
        else:
            T_vals = np.full(n_data, T0)
        
        p_init = {}
        for dp in range(n_data):
            for i in range(n_comp):
                p_init[(dp, i)] = 10**(antoine_params[i, 0] - antoine_params[i, 1] / 
                                     (antoine_params[i, 2] + T_vals[dp]))
        
        y_init = {(dp, i): x[dp, i] for dp in range(n_data) for i in range(n_comp)}
        
        # Define variables
        self._setup_variables(T0, init_data, p_init, y_init)
        
        # Define constraints
        self._setup_constraints()
    
    def _setup_parameters(self, x, P, antoine_params, mixture):
        """Set up model parameters."""
        self.model.A = pyo.Param(self.model.i, initialize=antoine_params[:, 0], 
                                mutable=False, name='Antoine A', domain=pyo.Reals)
        self.model.B = pyo.Param(self.model.i, initialize=antoine_params[:, 1], 
                                mutable=False, name='Antoine B', domain=pyo.Reals)
        self.model.C = pyo.Param(self.model.i, initialize=antoine_params[:, 2], 
                                mutable=False, name='Antoine C', domain=pyo.Reals)
        
        # Flatten interaction parameters for Pyomo
        a_flat = {}
        for group1 in self.a_params:
            for group2 in self.a_params[group1]:
                a_flat[(group1, group2)] = self.a_params[group1][group2]
        
        self.model.a = pyo.Param(self.model.grp, self.model.grp, 
                                initialize=a_flat, mutable=False,
                                name='unifac interaction a_mk', domain=pyo.Reals)
        self.model.Rk = pyo.Param(self.model.grp, initialize=self.rk_params, 
                                 mutable=False, name='unifac Rk parameters')
        self.model.Qk = pyo.Param(self.model.grp, initialize=self.qk_params, 
                                 mutable=False, name='unifac Qk parameters')
        self.model.x = pyo.Param(self.model.dp, self.model.i, 
                                initialize={index: v for index, v in np.ndenumerate(x)}, 
                                mutable=False, name='Liquid composition x', domain=pyo.Reals)
        self.model.P = pyo.Param(initialize=P, mutable=False, name='Total pressure in MPa', 
                                domain=pyo.PositiveReals)
        # Convert mixture DataFrame to proper indexing for Pyomo
        mixture_dict = {}
        for i, compound in enumerate(mixture.index):
            for group in mixture.columns:
                mixture_dict[(i, group)] = mixture.loc[compound, group]
        
        self.model.v = pyo.Param(self.model.i, self.model.k, 
                                initialize=mixture_dict, mutable=False, 
                                name='compounds')
    
    def _setup_variables(self, T0, init_data, p_init, y_init):
        """Set up model variables."""
        # Temperature variable with wider bounds
        if hasattr(T0, 'min'):
            T_bounds = (max(250.0, T0.min() - 50), T0.max() + 50)
        else:
            T_bounds = (max(250.0, T0 - 50), T0 + 50)
        self.model.T = pyo.Var(self.model.dp, initialize=T0, within=pyo.PositiveReals, 
                              bounds=T_bounds, name='Temperature in K')
        
        # UNIFAC parameters (fixed)
        self.model.r = pyo.Param(self.model.i, initialize=init_data['r'], 
                                mutable=False, name='unifac r_i')
        self.model.q = pyo.Param(self.model.i, initialize=init_data['q'], 
                                mutable=False, name='unifac q_i')
        self.model.J = pyo.Param(self.model.dp, self.model.i, initialize=init_data['J'], 
                                mutable=False, name='unifac J_i')
        self.model.L = pyo.Param(self.model.dp, self.model.i, initialize=init_data['L'], 
                                mutable=False, name='unifac L_i')
        self.model.G = pyo.Param(self.model.k, self.model.i, initialize=init_data['g'], 
                                mutable=False, name='unifac G_ki')
        self.model.theta = pyo.Param(self.model.dp, self.model.k, initialize=init_data['theta'], 
                                    mutable=False, name='unifac theta_k')
        
        # UNIFAC variables
        self.model.tau = pyo.Var(self.model.dp, self.model.k, self.model.k, 
                                initialize=init_data['tau'], within=pyo.PositiveReals, 
                                name='unifac tau_mk')
        self.model.s = pyo.Var(self.model.dp, self.model.k, self.model.i, 
                              initialize=init_data['s'], within=pyo.PositiveReals, 
                              name='unifac s_ki')
        self.model.eta = pyo.Var(self.model.dp, self.model.k, initialize=init_data['eta'], 
                                within=pyo.PositiveReals, name='unifac eta_k')
        self.model.gamma_c = pyo.Var(self.model.dp, self.model.i, initialize=init_data['gamma_c'], 
                                    within=pyo.Reals, name='unifac combinatorial gamma')
        self.model.gamma_r = pyo.Var(self.model.dp, self.model.i, initialize=init_data['gamma_r'], 
                                    within=pyo.Reals, name='unifac residual gamma')
        self.model.gamma = pyo.Var(self.model.dp, self.model.i, initialize=init_data['gamma'], 
                                  within=pyo.Reals, name='unifac total gamma')
        self.model.psat = pyo.Var(self.model.dp, self.model.i, initialize=p_init, 
                                 within=pyo.PositiveReals, bounds=(None, pyo.value(3000)),
                                 name='vapour pressure in bar')
        self.model.y = pyo.Var(self.model.dp, self.model.i, initialize=y_init, 
                              within=pyo.PercentFraction, name='vapour composition y')
    
    def _setup_constraints(self):
        """Set up model constraints."""
        # Vapour fraction summation constraint
        def sum_y(m, j):
            return sum(m.y[j, i] for i in m.i) == 1.0
        self.model.sum_y = pyo.Constraint(self.model.dp, rule=sum_y, 
                                         name='Constraint for summation of vapour fractions')
        
        # Antoine equation constraint
        def eq_psat(m, j, i):
            return m.psat[j, i] == 10 ** (m.A[i] - m.B[i] / (m.C[i] + m.T[j]))
        self.model.eq_psat = pyo.Constraint(self.model.dp, self.model.i, rule=eq_psat, 
                                           name='Antoine equation')
        
        # UNIFAC tau constraint
        def eq_tau(m, dp, k, l):
            return m.tau[dp, k, l] == pyo.exp(-m.a[k, l] / m.T[dp])
        self.model.eq_tau = pyo.Constraint(self.model.dp, self.model.k, self.model.k, 
                                          rule=eq_tau, name='UNIFAC tau calculation')
        
        # UNIFAC s constraint
        def eq_s(m, dp, k, i):
            return m.s[dp, k, i] == sum(m.G[l, i] * m.tau[dp, l, k] for l in m.k)
        self.model.eq_s = pyo.Constraint(self.model.dp, self.model.k, self.model.i, 
                                        rule=eq_s, name='UNIFAC s calculation')
        
        # UNIFAC eta constraint
        def eq_eta(m, dp, k):
            return m.eta[dp, k] == sum(m.s[dp, k, i] * m.x[dp, i] for i in m.i)
        self.model.eq_eta = pyo.Constraint(self.model.dp, self.model.k, rule=eq_eta, 
                                          name='UNIFAC eta calculation')
        
        # UNIFAC combinatorial activity coefficient constraint
        def eq_gamma_c(m, dp, i):
            return m.gamma_c[dp, i] == 1 - m.J[dp, i] + pyo.log(m.J[dp, i]) - 5 * m.q[i] * (
                1 - m.J[dp, i] / m.L[dp, i] + pyo.log(m.J[dp, i] / m.L[dp, i]))
        self.model.eq_gamma_c = pyo.Constraint(self.model.dp, self.model.i, rule=eq_gamma_c, 
                                              name='UNIFAC combinatorial gamma')
        
        # UNIFAC residual activity coefficient constraint
        def eq_gamma_r(m, dp, i):
            return m.gamma_r[dp, i] == m.q[i] * (1 - pyo.log(m.L[dp, i])) - sum(
                m.theta[dp, k] * m.s[dp, k, i] / m.eta[dp, k] - m.G[k, i] * 
                pyo.log(m.s[dp, k, i] / m.eta[dp, k]) for k in m.k)
        self.model.eq_gamma_r = pyo.Constraint(self.model.dp, self.model.i, rule=eq_gamma_r, 
                                              name='UNIFAC residual gamma')
        
        # Total activity coefficient constraint
        def eq_gamma(m, dp, i):
            return m.gamma[dp, i] == pyo.exp(m.gamma_r[dp, i] + m.gamma_c[dp, i])
        self.model.eq_gamma = pyo.Constraint(self.model.dp, self.model.i, rule=eq_gamma, 
                                            name='Total activity coefficient')
        
        # VLE equilibrium constraint
        def eq_y(m, dp, i):
            # Both psat and P are in the same units (MPa) from Antoine equation
            return m.y[dp, i] == m.psat[dp, i] * m.x[dp, i] * m.gamma[dp, i] / m.P
        self.model.eq_y = pyo.Constraint(self.model.dp, self.model.i, rule=eq_y, 
                                        name='VLE equilibrium')
    
    def solve(self, verbose=False):
        """
        Solve the optimization problem.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print solver output
            
        Returns:
        --------
        tuple
            (solution, model) - Pyomo solution object and model
        """
        if self.model is None:
            raise ValueError("Model must be set up before solving. Call setup_model() first.")
        
        # Set up solver
        opt = pyo.SolverFactory('ipopt')
        for key, value in self.solver_options.items():
            opt.options[key] = value
        
        # Solve the problem
        self.solution = opt.solve(self.model, tee=verbose, symbolic_solver_labels=True, 
                                 logfile="unifac_bubble_T_log")
        
        # Check solution status
        solv_stat = str(self.solution.solver.status)
        term_cond = str(self.solution.solver.termination_condition)
        
        if solv_stat != 'ok' or term_cond != 'optimal':
            warnings.warn('Solution may not be optimal. Solver status: {}, '
                         'Termination condition: {}'.format(solv_stat, term_cond))
        
        return self.solution, self.model
    
    def calculate_bubble_temperature(self, x, T0, P, antoine_params, mixture, verbose=False):
        """
        Complete bubble temperature calculation workflow.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Liquid composition matrix (n_data x n_comp)
        T0 : float or numpy.ndarray
            Initial guess for saturation temperature in Kelvin
        P : float
            Total pressure in MPa
        antoine_params : numpy.ndarray
            Antoine parameters array (n_comp x 3)
        mixture : pandas.DataFrame
            Mixture groups as columns
        verbose : bool
            Whether to print solver output
            
        Returns:
        --------
        tuple
            (solution, model) - Pyomo solution object and model
        """
        self.setup_model(x, T0, P, antoine_params, mixture)
        return self.solve(verbose)
    
    def get_results(self):
        """
        Extract results from the solved model.
        
        Returns:
        --------
        dict
            Dictionary containing temperature, vapour fractions, and activity coefficients
        """
        if self.model is None or self.solution is None:
            raise ValueError("Model must be solved before extracting results.")
        
        results = {}
        
        # Extract temperatures
        temperatures = []
        for dp in self.model.dp:
            temperatures.append(pyo.value(self.model.T[dp]))
        results['temperature'] = np.array(temperatures)
        
        # Extract vapour fractions
        y_values = []
        for dp in self.model.dp:
            y_row = []
            for i in self.model.i:
                y_row.append(pyo.value(self.model.y[dp, i]))
            y_values.append(y_row)
        results['vapour_fractions'] = np.array(y_values)
        
        # Extract activity coefficients
        gamma_values = []
        for dp in self.model.dp:
            gamma_row = []
            for i in self.model.i:
                gamma_row.append(pyo.value(self.model.gamma[dp, i]))
            gamma_values.append(gamma_row)
        results['activity_coefficients'] = np.array(gamma_values)
        
        # Extract vapour pressures
        psat_values = []
        for dp in self.model.dp:
            psat_row = []
            for i in self.model.i:
                psat_row.append(pyo.value(self.model.psat[dp, i]))
            psat_values.append(psat_row)
        results['vapour_pressures'] = np.array(psat_values)
        
        return results


def generate_xy_diagram(calculator, mixture, antoine_params, P=0.101325, n_points=20):
    """
    Generate x-y diagram for binary mixture at constant pressure.
    
    Parameters:
    -----------
    calculator : UnifacBubbleTCalculator
        Initialized calculator instance
    mixture : pd.DataFrame
        Mixture group composition
    antoine_params : np.ndarray
        Antoine parameters for components
    P : float
        Pressure (MPa)
    n_points : int
        Number of points for diagram
        
    Returns:
    --------
    dict
        Dictionary with x1, y1, T arrays for plotting
    """
    # Generate composition range
    x1_range = np.linspace(0.00, 1.00, n_points)
    
    results = {'x1': [], 'y1': [], 'T': []}
    
    for x1 in x1_range:
        x2 = 1.0 - x1
        x = np.array([[x1, x2]])
        
        # Improved initial temperature guess
        # Calculate pure component boiling points at given pressure
        # P is already in MPa, which matches Antoine parameter units
        try:
            T_pure1 = antoine_params[0, 1] / (antoine_params[0, 0] - np.log10(P)) - antoine_params[0, 2]
            T_pure2 = antoine_params[1, 1] / (antoine_params[1, 0] - np.log10(P)) - antoine_params[1, 2]
            
            # Linear interpolation with reasonable bounds
            T_guess = x1 * T_pure1 + x2 * T_pure2
            T_guess = max(250.0, min(450.0, T_guess))  # Keep within reasonable bounds
            T0 = np.array([T_guess])
        except:
            # Fallback temperature if Antoine calculation fails
            T0 = np.array([320.0])
        
        try:
            solution, _ = calculator.calculate_bubble_temperature(
                x, T0, P, antoine_params, mixture, verbose=False
            )
            
            if str(solution.solver.status) == 'ok':
                calc_results = calculator.get_results()
                
                results['x1'].append(x1)
                results['y1'].append(calc_results['vapour_fractions'][0, 0])
                results['T'].append(calc_results['temperature'][0])
                
        except Exception as e:
            print(f"Warning: Failed calculation for x1={x1:.3f}: {e}")
            continue
    
    return {k: np.array(v) for k, v in results.items()}


# Example usage with acetone and methanol using existing AntoineCalculator
if __name__ == "__main__":
    
    print("UNIFAC Bubble Temperature Calculation with xy-Diagram")
    print("=" * 60)
    print("System: Acetone + Methanol")
    
    # Import the existing AntoineCalculator and PropertyCalculator
    try:
        import sys
        from pathlib import Path
        
        # Add src directory to path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            
        from antoine_calculator import AntoineCalculator
        from property_calculator import PropertyCalculator
        
        # Define compounds using group contributions
        acetone = {'CH3': 1, 'CH3CO': 1}
        methanol = {'CH3OH': 1}
        
        print(f"\nCompound definitions:")
        print(f"Acetone:  {acetone}")
        print(f"Methanol: {methanol}")
        
        # Calculate critical properties using PropertyCalculator
        prop_calc = PropertyCalculator(missing_group_action="remove_compounds")
        
        print(f"\nCalculating critical properties...")
        acetone_props = prop_calc.calculate_all_properties(acetone)
        methanol_props = prop_calc.calculate_all_properties(methanol)
        
        if acetone_props and methanol_props:
            # Convert pc from bar to Pa for AntoineCalculator
            acetone_props['pc'] *= 1e5  # bar to Pa
            methanol_props['pc'] *= 1e5  # bar to Pa
            
            print(f"Acetone properties:  Tc={acetone_props['tc']:.1f}K, Pc={acetone_props['pc']/1e6:.2f}MPa, Tb={acetone_props['tb']:.1f}K")
            print(f"Methanol properties: Tc={methanol_props['tc']:.1f}K, Pc={methanol_props['pc']/1e6:.2f}MPa, Tb={methanol_props['tb']:.1f}K")
            
            # Calculate Antoine parameters using existing AntoineCalculator
            antoine_calc = AntoineCalculator(verbose=False)
            
            print(f"\nCalculating Antoine parameters...")
            acetone_antoine = antoine_calc.calculate_antoine_parameters(acetone, acetone_props)
            methanol_antoine = antoine_calc.calculate_antoine_parameters(methanol, methanol_props)
            
            if acetone_antoine and methanol_antoine:
                antoine_params = np.array([
                    [acetone_antoine['A'], acetone_antoine['B'], acetone_antoine['C']],
                    [methanol_antoine['A'], methanol_antoine['B'], methanol_antoine['C']]
                ])
                
                print(f"Acetone Antoine:  A={acetone_antoine['A']:.4f}, B={acetone_antoine['B']:.2f}, C={acetone_antoine['C']:.2f}")
                print(f"Methanol Antoine: A={methanol_antoine['A']:.4f}, B={methanol_antoine['B']:.2f}, C={methanol_antoine['C']:.2f}")
                
                # Define the mixture groups for UNIFAC
                # Acetone: CH3 (2) + CH3CO (1)
                # Methanol: CH3 (1) + OH (1)
                mixture_data = {
                    'CH3': [1, 0],      # 2 CH3 groups in acetone, 1 in methanol
                    'CH3CO': [1, 0],    # 1 CH3CO group in acetone, 0 in methanol
                    'CH3OH': [0, 1]        # 0 OH groups in acetone, 1 in methanol
                }
                
                mixture = pd.DataFrame(mixture_data, index=['acetone', 'methanol'])
                
                print(f"\nMixture group contributions:")
                print(mixture)
                
                # Operating conditions
                P = 0.101325  # MPa (atmospheric pressure)
                print(f"\nPressure: {P} MPa")
                
                try:
                    # Create calculator instance
                    calculator = UnifacBubbleTCalculator()
                    
                    print(f"\nGenerating x-y diagram...")
                    
                    # Generate x-y diagram (reduce points for testing)
                    xy_data = generate_xy_diagram(calculator, mixture, antoine_params, P, n_points=100)
                    
                    if len(xy_data['x1']) > 0:
                        print(f"Successfully calculated {len(xy_data['x1'])} points")
                        
                        # Import matplotlib
                        import matplotlib.pyplot as plt
                        
                        # Create x-y diagram
                        plt.figure(figsize=(12, 10))
                        
                        # Plot x-y curve
                        plt.subplot(2, 1, 1)
                        plt.plot(xy_data['x1'], xy_data['y1'], 'b-o', linewidth=2, markersize=6, label='UNIFAC')
                        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='x=y line')
                        plt.xlabel('x₁ (acetone liquid mole fraction)')
                        plt.ylabel('y₁ (acetone vapor mole fraction)')
                        plt.title('x-y Diagram for Acetone-Methanol System (UNIFAC)')
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.xlim(0, 1)
                        plt.ylim(0, 1)
                        
                        # Plot T-x-y diagram
                        plt.subplot(2, 1, 2)
                        plt.plot(xy_data['x1'], xy_data['T'], 'r-o', linewidth=2, markersize=6, label='Bubble T (liquid)')
                        plt.plot(xy_data['y1'], xy_data['T'], 'b-s', linewidth=2, markersize=6, label='Dew T (vapor)')
                        plt.xlabel('Mole fraction of acetone')
                        plt.ylabel('Temperature (K)')
                        plt.title('T-x-y Diagram for Acetone-Methanol System')
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.xlim(0, 1)
                        
                        plt.tight_layout()
                        plt.savefig('acetone_methanol_xy_diagram.png', dpi=300, bbox_inches='tight')
                        plt.show()
                        
                        print(f"\nDiagram saved as 'acetone_methanol_xy_diagram.png'")
                        
                        # Print some key results
                        print(f"\nKey results:")
                        print(f"Pure acetone boiling point: {xy_data['T'][0]:.2f} K ({xy_data['T'][0]-273.15:.2f} °C)")
                        print(f"Pure methanol boiling point: {xy_data['T'][-1]:.2f} K ({xy_data['T'][-1]-273.15:.2f} °C)")
                        
                        # Find azeotrope if exists
                        x_diff = np.abs(xy_data['x1'] - xy_data['y1'])
                        min_idx = np.argmin(x_diff)
                        if x_diff[min_idx] < 0.05:  # Potential azeotrope
                            print(f"Potential azeotrope near x₁={xy_data['x1'][min_idx]:.3f}, "
                                  f"T={xy_data['T'][min_idx]:.2f} K ({xy_data['T'][min_idx]-273.15:.2f} °C)")
                        
                        # Print data table
                        print(f"\nDetailed results:")
                        print(f"{'x₁':>6} {'y₁':>6} {'T (K)':>8} {'T (°C)':>8}")
                        print("-" * 30)
                        for i in range(len(xy_data['x1'])):
                            print(f"{xy_data['x1'][i]:6.3f} {xy_data['y1'][i]:6.3f} {xy_data['T'][i]:8.2f} {xy_data['T'][i]-273.15:8.2f}")
                        
                    else:
                        print("No successful calculations - check solver and data files")
                        
                except Exception as e:
                    print(f"\nError during UNIFAC calculation: {e}")
                    print("This might be due to:")
                    print("1. Missing IPOPT solver")
                    print("2. Missing JSON data files in data/json/ directory")
                    print("3. Missing matplotlib for plotting")
                    print("4. Pyomo installation issues")
                    
            else:
                print("Failed to calculate Antoine parameters")
                
        else:
            print("Failed to calculate critical properties - missing groups in PropertyCalculator")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Required modules not found. Make sure you have:")
        print("1. AntoineCalculator in src/antoine_calculator.py")
        print("2. PropertyCalculator in src/property_calculator.py")
        print("3. All dependencies installed")
        
    print("\n" + "=" * 60)