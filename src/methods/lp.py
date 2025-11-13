"""Linear Programming (LP) pricing method implementation.

Based on Gupta & Nagarajan's reduction from convex Myerson program to linear program.
This is a NEW method not in the original Hikima et al. paper.
"""

import numpy as np
import pulp
from typing import Dict, Any, List, Tuple

from .base import BasePricingMethod


class LPMethod(BasePricingMethod):
    """
    Linear Programming pricing method using Gupta-Nagarajan approach.
    
    This method linearizes the Myerson revenue optimization problem by:
    1. Discretizing the price space into a finite grid
    2. Using probing variables y_c,π for each rider-price pair
    3. Using allocation variables x_c,t,π for rider-taxi-price triples
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LP pricing method."""
        super().__init__(config)
        
        # LP-specific parameters
        # Trade-off: Slightly less granular pricing, but still adequate
        self.price_grid_size = config.get('lp_price_grid_size', 50)
        self.price_min_multiplier = config.get('lp_price_min_mult', 0.5)
        self.price_max_multiplier = config.get('lp_price_max_mult', 2.0)
        
        # Solver selection (default to CBC for compatibility, can override with 'highs', 'gurobi', 'cplex')
        self.solver_name = config.get('lp_solver', 'cbc')  # 'cbc', 'highs', 'gurobi', 'cplex'
        
    def get_method_name(self) -> str:
        """Get method name."""
        return "LP"
    
    def compute_prices(
        self,
        scenario_data: Dict[str, Any],
        acceptance_function: str
    ) -> Tuple[np.ndarray, float]:
        """
        Compute prices using the Gupta-Nagarajan LP formulation optimized for specific acceptance function.
        
        Args:
            scenario_data: Dictionary with scenario data
            acceptance_function: 'PL' or 'Sigmoid' - which function to optimize for
            
        Returns:
            Tuple of (prices array, optimal objective value)
        """
        n_requesters = scenario_data['num_requesters']
        n_taxis = scenario_data['num_taxis']
        edge_weights = scenario_data['edge_weights']
        trip_amounts = scenario_data['trip_amounts']
        
        # Handle edge cases
        if n_requesters == 0:
            return np.array([]), 0.0
        if n_taxis == 0:
            return np.zeros(n_requesters), 0.0
        
        # Generate price grids for each requester
        price_grids = self._generate_price_grids(trip_amounts)
        
        # Calculate acceptance probabilities for each price
        acceptance_probs = self._calculate_acceptance_probabilities(
            price_grids, trip_amounts, acceptance_function
        )
        
        # Build and solve the LP
        prob, x_vars, y_vars = self._build_lp(
            n_requesters, n_taxis, edge_weights,
            price_grids, acceptance_probs
        )
        
        # Solve the LP with selected solver
        solver = self._get_solver()
        prob.solve(solver)
        
        if prob.status != pulp.LpStatusOptimal:
            self.logger.warning(f"LP did not find optimal solution: {pulp.LpStatus[prob.status]}")
            # Return baseline prices if LP fails
            return trip_amounts * 1.2, 0.0
        
        # Extract optimal objective value
        opt_value = pulp.value(prob.objective) if prob.objective else 0.0
        
        # Extract prices from solution
        prices = self._extract_prices_from_solution(
            y_vars, price_grids, n_requesters
        )
        
        self.logger.info(f"{acceptance_function} LP optimal value: ${opt_value:.2f}")
        
        return prices, opt_value
    
    def _get_solver(self):
        """Get LP solver based on configuration with fallback to CBC."""
        solver_map = {
            'cbc': pulp.PULP_CBC_CMD,
            'highs': pulp.HiGHS_CMD,
            'gurobi': pulp.GUROBI_CMD,
            'cplex': pulp.CPLEX_CMD
        }
        
        solver_class = solver_map.get(self.solver_name.lower(), pulp.PULP_CBC_CMD)
        
        try:
            # Try to create solver with msg=False (suppress output)
            solver = solver_class(msg=False)
            
            # Check if solver is actually available
            if not solver.available():
                raise Exception(f"Solver binary not found")
            
            return solver
        except Exception as e:
            # Fallback to CBC if solver not available
            if self.solver_name.lower() != 'cbc':
                self.logger.warning(f"Solver '{self.solver_name}' not available ({e}), falling back to CBC")
            return pulp.PULP_CBC_CMD(msg=False)
    
    def _generate_price_grids(self, trip_amounts: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Generate discretized price grids for each requester.
        
        Args:
            trip_amounts: Array of trip valuations
            
        Returns:
            Dictionary mapping requester index to price grid
        """
        price_grids = {}
        
        for i, amount in enumerate(trip_amounts):
            # Generate price grid based on trip amount
            min_price = amount * self.price_min_multiplier
            max_price = amount * self.price_max_multiplier
            
            # Create log-spaced grid for better coverage
            grid = np.logspace(
                np.log10(max(min_price, 0.1)),
                np.log10(max_price),
                self.price_grid_size
            )
            price_grids[i] = grid
        
        return price_grids
    
    def _calculate_acceptance_probabilities(
        self,
        price_grids: Dict[int, np.ndarray],
        trip_amounts: np.ndarray,
        acceptance_function: str
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate acceptance probability for each (requester, price_index) pair using specified acceptance function.
        
        PERFORMANCE: Partially vectorized for efficiency.
        
        Returns:
            Dictionary mapping (requester_id, price_index) to acceptance probability
        """
        acceptance_probs = {}
        
        # Process each requester (can't fully vectorize due to different grid sizes per requester)
        for i, amount in enumerate(trip_amounts):
            prices = price_grids[i]  # Array of prices for this requester
            
            if acceptance_function == 'PL':
                # Vectorized piecewise linear: P(accept) = max(0, min(1, -2/v * p + 3))
                probs = np.clip(-2.0 / amount * prices + 3.0, 0.0, 1.0)
            else:  # Sigmoid
                # Vectorized sigmoid acceptance
                exponents = (-prices + self.sigmoid_beta * amount) / (self.sigmoid_gamma * abs(amount))
                probs = 1.0 - (1.0 / (1.0 + np.exp(exponents)))
            
            # Store in dictionary format expected by LP builder
            for j, prob in enumerate(probs):
                acceptance_probs[(i, j)] = float(prob)
        
        return acceptance_probs
    
    def _build_lp(
        self,
        n_requesters: int,
        n_taxis: int,
        edge_weights: np.ndarray,
        price_grids: Dict[int, np.ndarray],
        acceptance_probs: Dict[Tuple[int, int], float]
    ) -> Tuple[pulp.LpProblem, Dict, Dict]:
        """
        Build the Gupta-Nagarajan LP formulation.
        
        Returns:
            Tuple of (problem, x_variables, y_variables)
        """
        # Create LP problem
        prob = pulp.LpProblem("RideHailing_GN_LP", pulp.LpMaximize)
        
        # Create variables
        # y[c, pi_idx]: probability we offer price pi to requester c
        y_vars = {}
        for c in range(n_requesters):
            for pi_idx in range(len(price_grids[c])):
                y_vars[(c, pi_idx)] = pulp.LpVariable(
                    f"y_{c}_{pi_idx}", lowBound=0, upBound=1
                )
        
        # x[c, t, pi_idx]: probability requester c accepts price pi and is matched to taxi t
        x_vars = {}
        for c in range(n_requesters):
            for t in range(n_taxis):
                for pi_idx in range(len(price_grids[c])):
                    x_vars[(c, t, pi_idx)] = pulp.LpVariable(
                        f"x_{c}_{t}_{pi_idx}", lowBound=0, upBound=1
                    )
        
        # Objective: maximize expected profit
        objective = 0
        for c in range(n_requesters):
            for t in range(n_taxis):
                for pi_idx in range(len(price_grids[c])):
                    price = price_grids[c][pi_idx]
                    cost = -edge_weights[c, t]  # Cost is negative of weight
                    profit = price - cost
                    objective += profit * x_vars[(c, t, pi_idx)]
        
        prob += objective, "Total_expected_profit"
        
        # Constraints
        
        # (1) Probe at most one price per requester
        for c in range(n_requesters):
            constraint = pulp.lpSum(
                y_vars[(c, pi_idx)] 
                for pi_idx in range(len(price_grids[c]))
            ) <= 1
            prob += constraint, f"Offer_once_{c}"
        
        # (2) Matching only after acceptance
        for c in range(n_requesters):
            for pi_idx in range(len(price_grids[c])):
                taxi_sum = pulp.lpSum(
                    x_vars[(c, t, pi_idx)]
                    for t in range(n_taxis)
                )
                prob += (
                    taxi_sum <= acceptance_probs[(c, pi_idx)] * y_vars[(c, pi_idx)],
                    f"Link_{c}_{pi_idx}"
                )
        
        # (3) Taxi capacity: one requester per taxi
        for t in range(n_taxis):
            constraint = pulp.lpSum(
                x_vars[(c, t, pi_idx)]
                for c in range(n_requesters)
                for pi_idx in range(len(price_grids[c]))
            ) <= 1
            prob += constraint, f"Taxi_cap_{t}"
        
        return prob, x_vars, y_vars
    
    def _extract_prices_from_solution(
        self,
        y_vars: Dict,
        price_grids: Dict[int, np.ndarray],
        n_requesters: int
    ) -> np.ndarray:
        """
        Extract prices from LP solution.
        
        Args:
            y_vars: Dictionary of y variables from LP
            price_grids: Price grids for each requester
            n_requesters: Number of requesters
            
        Returns:
            Array of prices for each requester
        """
        prices = np.zeros(n_requesters)
        
        for c in range(n_requesters):
            # Find which price was selected (highest y value)
            best_price_idx = -1
            best_y_value = -1
            
            for pi_idx in range(len(price_grids[c])):
                y_value = y_vars[(c, pi_idx)].varValue
                if y_value is not None and y_value > best_y_value:
                    best_y_value = y_value
                    best_price_idx = pi_idx
            
            if best_price_idx >= 0:
                prices[c] = price_grids[c][best_price_idx]
            else:
                # Default price if no price was selected
                prices[c] = np.mean(price_grids[c])
        
        return prices 