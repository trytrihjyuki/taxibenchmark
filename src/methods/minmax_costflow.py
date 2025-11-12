"""MinMaxCostFlow pricing method - properly optimized for each acceptance function."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx

from .base import BasePricingMethod


class MinMaxCostFlowMethod(BasePricingMethod):
    """
    MinMaxCostFlow method from Hikima et al. paper.
    Optimizes separately for each acceptance function.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MinMaxCostFlow method."""
        super().__init__(config)
        
        # Method-specific parameters
        self.delta = config.get('delta', 0.001)  # For delta-scaling
        self.max_iterations = config.get('max_iterations', 1000)
    
    def get_method_name(self) -> str:
        """Return method name."""
        return "MinMaxCostFlow"
    
    def compute_prices(
        self,
        scenario_data: Dict[str, Any],
        acceptance_function: str
    ) -> np.ndarray:
        """
        Compute prices using MinMaxCostFlow algorithm optimized for specific acceptance function.
        
        This implements the delta-scaling min-cost flow algorithm from Hikima et al.'s paper,
        with separate handling for PL and Sigmoid acceptance functions.
        
        Args:
            scenario_data: Scenario data with requesters, taxis, etc.
            acceptance_function: 'PL' or 'Sigmoid' - which function to optimize for
            
        Returns:
            Array of prices for each requester
        """
        # Extract data
        n_requesters = scenario_data['num_requesters']
        n_taxis = scenario_data['num_taxis']
        edge_weights = scenario_data['edge_weights']
        trip_amounts = scenario_data['trip_amounts']
        
        self.logger.info(f"{acceptance_function}: N={n_requesters}, M={n_taxis}")
        
        # Handle edge cases
        if n_requesters == 0:
            self.logger.debug(f"{acceptance_function}: No requesters, returning empty array")
            return np.array([])
        if n_taxis == 0:
            self.logger.debug(f"{acceptance_function}: No taxis, returning zeros")
            return np.zeros(n_requesters)
        
        # Use the appropriate algorithm based on acceptance function
        if acceptance_function == 'PL':
            return self._solve_piecewise_linear(n_requesters, n_taxis, edge_weights, trip_amounts)
        else:  # Sigmoid
            return self._solve_sigmoid(n_requesters, n_taxis, edge_weights, trip_amounts)
    
    def _solve_piecewise_linear(
        self,
        n_requesters: int,
        n_taxis: int, 
        edge_weights: np.ndarray,
        trip_amounts: np.ndarray
    ) -> np.ndarray:
        """
        Solve for PL acceptance function using delta-scaling min-cost flow.
        Based on Hikima's experiment_PL.py implementation.
        """
        n = n_requesters
        m = n_taxis
        
        # PL function coefficients: P(accept) = -c*price + d
        c = 2.0 / trip_amounts  # coefficient for PL function  
        d = 3.0  # constant for PL function
        
        # Build flow network (following Hikima's structure)
        # Nodes: requesters [0...n-1], taxis [n...n+m-1], source [n+m], sink [n+m+1]
        num_nodes = n + m + 2
        source = n + m
        sink = n + m + 1
        
        # Initialize flow, capacity, and cost matrices
        Flow = np.zeros((num_nodes, num_nodes))
        Cap_matrix = np.zeros((num_nodes, num_nodes))
        Cost_matrix = np.ones((num_nodes, num_nodes)) * np.inf
        
        # Set up edges (i,j) for requesters to taxis
        for i in range(n):
            for j in range(m):
                Cap_matrix[i, n+j] = np.inf
                Cost_matrix[i, n+j] = -edge_weights[i, j]  # Cost from edge weights
                Cost_matrix[n+j, i] = edge_weights[i, j]
        
        # Initialize delta
        delta = n
        
        # Source to requesters - initial setup
        for i in range(n):
            Cap_matrix[source, i] = 1
            # Initial cost calculation for PL
            val = (1/c[i]*(delta**2) - (d/c[i])*delta - 0) / delta
            Cost_matrix[source, i] = val
        
        # Taxis to sink
        for j in range(m):
            Cap_matrix[n+j, sink] = 1
            Cost_matrix[n+j, sink] = 0
        
        # Source to sink
        Cap_matrix[source, sink] = n
        Cost_matrix[source, sink] = 0
        
        # Excess and potential arrays
        excess = np.zeros(num_nodes)
        excess[source] = n
        excess[sink] = -n
        potential = np.zeros(num_nodes)
        
        # Delta-scaling algorithm
        epsilon = 1e-10
        
        # First iteration - delta-scaling phase
        for i in range(n):
            for j in range(m):
                if Cost_matrix[i, n+j] < 0:
                    flow_amount = min(delta, Cap_matrix[i, n+j])
                    Flow[i, n+j] += flow_amount
                    excess[i] -= flow_amount
                    excess[n+j] += flow_amount
                    Cap_matrix[i, n+j] -= flow_amount
                    Cap_matrix[n+j, i] += flow_amount
        
        # Main delta-scaling loop
        while delta > 0.001:
            # Process source-requester edges
            for i in range(n):
                if Cost_matrix[source, i] - potential[source] + potential[i] < -epsilon and Cap_matrix[source, i] >= delta:
                    Flow[source, i] += delta
                    excess[source] -= delta  
                    excess[i] += delta
                    Cap_matrix[source, i] -= delta
                    Cap_matrix[i, source] += delta
                    
                    # Update cost for PL function
                    val = (1/c[i]*((Flow[source, i] + delta)**2) - (d/c[i])*(Flow[source, i] + delta) - 
                          (1/c[i]*(Flow[source, i]**2) - (d/c[i])*Flow[source, i])) / delta
                    Cost_matrix[source, i] = val
                    val = (1/c[i]*((Flow[source, i] - delta)**2) - (d/c[i])*(Flow[source, i] - delta) - 
                          (1/c[i]*(Flow[source, i]**2) - (d/c[i])*Flow[source, i])) / delta
                    Cost_matrix[i, source] = val
                    
                if Cost_matrix[i, source] - potential[i] + potential[source] < -epsilon and Cap_matrix[i, source] >= delta:
                    Flow[source, i] -= delta
                    excess[source] += delta
                    excess[i] -= delta  
                    Cap_matrix[source, i] += delta
                    Cap_matrix[i, source] -= delta
                    
                    # Update cost
                    val = (1/c[i]*((Flow[source, i] + delta)**2) - (d/c[i])*(Flow[source, i] + delta) - 
                          (1/c[i]*(Flow[source, i]**2) - (d/c[i])*Flow[source, i])) / delta
                    Cost_matrix[source, i] = val
                    val = (1/c[i]*((Flow[source, i] - delta)**2) - (d/c[i])*(Flow[source, i] - delta) - 
                          (1/c[i]*(Flow[source, i]**2) - (d/c[i])*Flow[source, i])) / delta
                    Cost_matrix[i, source] = val
            
            # Process requester-taxi and taxi-sink edges similar to original
            for i in range(n):
                for j in range(m):
                    if (Cost_matrix[i, n+j] - potential[i] + potential[n+j] < -epsilon and 
                        Cap_matrix[i, n+j] >= delta):
                        Flow[i, n+j] += delta
                        excess[i] -= delta
                        excess[n+j] += delta
                        Cap_matrix[i, n+j] -= delta
                        Cap_matrix[n+j, i] += delta
                    
                    if (Cost_matrix[n+j, i] - potential[n+j] + potential[i] < -epsilon and 
                        Cap_matrix[n+j, i] >= delta):
                        Flow[i, n+j] -= delta
                        excess[i] += delta
                        excess[n+j] -= delta
                        Cap_matrix[i, n+j] += delta
                        Cap_matrix[n+j, i] -= delta
            
            for j in range(m):
                if (-potential[n+j] + potential[sink] < -epsilon and 
                    Cap_matrix[n+j, sink] >= delta):
                    Flow[n+j, sink] += delta
                    excess[n+j] -= delta
                    excess[sink] += delta
                    Cap_matrix[n+j, sink] -= delta
                    Cap_matrix[sink, n+j] += delta
                    
                if (-potential[sink] + potential[n+j] < -epsilon and 
                    Cap_matrix[sink, n+j] >= delta):
                    Flow[n+j, sink] -= delta
                    excess[sink] -= delta
                    excess[n+j] += delta
                    Cap_matrix[n+j, sink] += delta
                    Cap_matrix[sink, n+j] -= delta
            
            # Source-sink edge
            if (-potential[source] + potential[sink] < -epsilon and 
                Cap_matrix[source, sink] >= delta):
                Flow[source, sink] += delta
                excess[source] -= delta
                excess[sink] += delta
                Cap_matrix[source, sink] -= delta
                Cap_matrix[sink, source] += delta
                
            if (-potential[sink] + potential[source] < -epsilon and 
                Cap_matrix[sink, source] >= delta):
                Flow[source, sink] -= delta
                excess[sink] -= delta
                excess[source] += delta
                Cap_matrix[source, sink] += delta
                Cap_matrix[sink, source] -= delta
            
            # Shortest path phase - update potentials using Dijkstra's algorithm
            # This maintains reduced cost optimality conditions
            potential = self._update_potentials_dijkstra(
                num_nodes, source, sink, Cost_matrix, Cap_matrix, potential, delta, epsilon
            )
            
            # Update delta
            delta = 0.5 * delta
            
            # Update costs for new delta
            for i in range(n):
                val = (1/c[i]*((Flow[source, i] + delta)**2) - (d/c[i])*(Flow[source, i] + delta) - 
                      (1/c[i]*(Flow[source, i]**2) - (d/c[i])*Flow[source, i])) / delta
                Cost_matrix[source, i] = val
                val = (1/c[i]*((Flow[source, i] - delta)**2) - (d/c[i])*(Flow[source, i] - delta) - 
                      (1/c[i]*(Flow[source, i]**2) - (d/c[i])*Flow[source, i])) / delta
                Cost_matrix[i, source] = val
        
        # Extract prices from final flow
        prices = np.zeros(n)
        for i in range(n):
            prices[i] = -(1/c[i]) * Flow[source, i] + d/c[i]
        
        return prices
    
    def _update_potentials_dijkstra(
        self,
        num_nodes: int,
        source: int,
        sink: int,
        Cost_matrix: np.ndarray,
        Cap_matrix: np.ndarray,
        potential: np.ndarray,
        delta: float,
        epsilon: float
    ) -> np.ndarray:
        """
        Update node potentials using Dijkstra's algorithm on residual network.
        This maintains reduced cost optimality for delta-scaling.
        """
        # Compute reduced costs for all edges
        reduced_costs = np.copy(Cost_matrix)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if Cap_matrix[i, j] >= delta:
                    reduced_costs[i, j] = Cost_matrix[i, j] - potential[i] + potential[j]
                else:
                    reduced_costs[i, j] = np.inf
        
        # Run Dijkstra from source to compute shortest path distances
        dist = np.full(num_nodes, np.inf)
        dist[source] = 0
        visited = np.zeros(num_nodes, dtype=bool)
        
        for _ in range(num_nodes):
            # Find unvisited node with minimum distance
            u = -1
            min_dist = np.inf
            for v in range(num_nodes):
                if not visited[v] and dist[v] < min_dist:
                    min_dist = dist[v]
                    u = v
            
            if u == -1 or min_dist == np.inf:
                break
                
            visited[u] = True
            
            # Update distances to neighbors
            for v in range(num_nodes):
                if not visited[v] and Cap_matrix[u, v] >= delta:
                    new_dist = dist[u] + reduced_costs[u, v]
                    if new_dist < dist[v]:
                        dist[v] = new_dist
        
        # Update potentials: potential[v] = potential[v] - dist[v]
        for v in range(num_nodes):
            if dist[v] < np.inf:
                potential[v] = potential[v] - dist[v]
        
        return potential
    
    def _solve_sigmoid(
        self,
        n_requesters: int,
        n_taxis: int,
        edge_weights: np.ndarray,
        trip_amounts: np.ndarray
    ) -> np.ndarray:
        """
        Solve for Sigmoid acceptance function using delta-scaling min-cost flow.
        Based on Hikima's experiment_Sigmoid.py implementation.
        """
        n = n_requesters
        m = n_taxis
        beta = self.sigmoid_beta
        gamma = self.sigmoid_gamma
        
        # Build flow network
        num_nodes = n + m + 2
        source = n + m
        sink = n + m + 1
        
        # Initialize matrices
        Flow = np.zeros((num_nodes, num_nodes))
        Cap_matrix = np.zeros((num_nodes, num_nodes))
        Cost_matrix = np.ones((num_nodes, num_nodes)) * np.inf
        
        # Set up edges
        for i in range(n):
            for j in range(m):
                Cap_matrix[i, n+j] = np.inf
                Cost_matrix[i, n+j] = -edge_weights[i, j]
                Cost_matrix[n+j, i] = edge_weights[i, j]
        
        # Initialize with 0.5 flow from source to each requester (Hikima's approach)
        for i in range(n):
            Cap_matrix[source, i] = 0.5
            Cap_matrix[i, source] = 0.5
            Cost_matrix[source, i] = np.inf
            Flow[source, i] = 0.5
        
        for j in range(m):
            Cap_matrix[n+j, sink] = 1
            Cost_matrix[n+j, sink] = 0
        
        Cap_matrix[source, sink] = n
        Cost_matrix[source, sink] = 0
        
        # Initial excess
        excess = np.zeros(num_nodes)
        excess[source] = n - 0.5 * n  # Adjusted for initial flows
        excess[sink] = -n
        for i in range(n):
            excess[i] += 0.5  # From initial flow
        
        potential = np.zeros(num_nodes)
        delta = n
        epsilon = 1e-10
        
        # Helper function for sigmoid cost calculation (from Hikima's val_calc function)
        def sigmoid_cost_calc(gamma_val, beta_val, amount, flow_val, delta_val, direction):
            """Calculate sigmoid cost following Hikima's val_calc function"""
            if direction == 1:  # Positive flow
                if flow_val != 0:
                    if 1 - (flow_val + delta_val) > 0:
                        val = ((gamma_val * amount * np.log((flow_val + delta_val)/(1-(flow_val + delta_val))) * (flow_val + delta_val) - 
                               beta_val * amount * (flow_val + delta_val)) - 
                              (gamma_val * amount * np.log(flow_val/(1-flow_val)) * flow_val - 
                               beta_val * amount * flow_val)) / delta_val
                    else:
                        val = np.inf
                else:
                    if 1 - (flow_val + delta_val) > 0:
                        val = (gamma_val * amount * np.log((flow_val + delta_val)/(1-(flow_val + delta_val))) * (flow_val + delta_val) - 
                              beta_val * amount * (flow_val + delta_val)) / delta_val
                    else:
                        val = np.inf
            else:  # Negative flow
                if flow_val - delta_val > 0:
                    if flow_val != 0:
                        if 1 > flow_val - delta_val:
                            val = ((gamma_val * amount * np.log((flow_val - delta_val)/(1-(flow_val - delta_val))) * (flow_val - delta_val) - 
                                   beta_val * amount * (flow_val - delta_val)) - 
                                  (gamma_val * amount * np.log(flow_val/(1-flow_val)) * flow_val - 
                                   beta_val * amount * flow_val)) / delta_val
                        else:
                            val = np.inf
                    else:
                        val = np.inf
                elif flow_val - delta_val == 0:
                    val = (0 - (gamma_val * amount * np.log(flow_val/(1-flow_val)) * flow_val - 
                              beta_val * amount * flow_val)) / delta_val
                else:
                    val = np.inf
            return val
        
        # Delta-scaling loop for sigmoid
        while delta > 0.001:
            # Process edges similar to PL but with sigmoid cost function
            for i in range(n):
                for j in range(m):
                    if (Cost_matrix[i, n+j] - potential[i] + potential[n+j] < -epsilon and 
                        Cap_matrix[i, n+j] >= delta):
                        Flow[i, n+j] += delta
                        excess[i] -= delta
                        excess[n+j] += delta
                        Cap_matrix[i, n+j] -= delta
                        Cap_matrix[n+j, i] += delta
                    
                    if (Cost_matrix[n+j, i] - potential[n+j] + potential[i] < -epsilon and 
                        Cap_matrix[n+j, i] >= delta):
                        Flow[i, n+j] -= delta
                        excess[i] += delta
                        excess[n+j] -= delta
                        Cap_matrix[i, n+j] += delta
                        Cap_matrix[n+j, i] -= delta
            
            for i in range(n):
                if (Cost_matrix[source, i] - potential[source] + potential[i] < -epsilon and 
                    Cap_matrix[source, i] >= delta):
                    Flow[source, i] += delta
                    excess[source] -= delta
                    excess[i] += delta
                    Cap_matrix[source, i] -= delta
                    Cap_matrix[i, source] += delta
                    
                    # Update sigmoid costs
                    if Flow[source, i] != 0:
                        Cost_matrix[source, i] = sigmoid_cost_calc(gamma, beta, trip_amounts[i], Flow[source, i], delta, 1)
                        Cost_matrix[i, source] = sigmoid_cost_calc(gamma, beta, trip_amounts[i], Flow[source, i], delta, -1)
                
                if (Cost_matrix[i, source] - potential[i] + potential[source] < -epsilon and 
                    Cap_matrix[i, source] >= delta):
                    Flow[source, i] -= delta
                    excess[source] += delta
                    excess[i] -= delta
                    Cap_matrix[source, i] += delta
                    Cap_matrix[i, source] -= delta
                    
                    Cost_matrix[source, i] = sigmoid_cost_calc(gamma, beta, trip_amounts[i], Flow[source, i], delta, 1)
                    Cost_matrix[i, source] = sigmoid_cost_calc(gamma, beta, trip_amounts[i], Flow[source, i], delta, -1)
            
            # Process taxi-sink edges
            for j in range(m):
                if (-potential[n+j] + potential[sink] < -epsilon and 
                    Cap_matrix[n+j, sink] >= delta):
                    Flow[n+j, sink] += delta
                    excess[n+j] -= delta
                    excess[sink] += delta
                    Cap_matrix[n+j, sink] -= delta
                    Cap_matrix[sink, n+j] += delta
                    
                if (-potential[sink] + potential[n+j] < -epsilon and 
                    Cap_matrix[sink, n+j] >= delta):
                    Flow[n+j, sink] -= delta
                    excess[sink] -= delta
                    excess[n+j] += delta
                    Cap_matrix[n+j, sink] += delta
                    Cap_matrix[sink, n+j] -= delta
            
            # Source-sink edge
            if (-potential[source] + potential[sink] < -epsilon and 
                Cap_matrix[source, sink] >= delta):
                Flow[source, sink] += delta
                excess[source] -= delta
                excess[sink] += delta
                Cap_matrix[source, sink] -= delta
                Cap_matrix[sink, source] += delta
            
            if (-potential[sink] + potential[source] < -epsilon and 
                Cap_matrix[sink, source] >= delta):
                Flow[source, sink] -= delta
                excess[sink] -= delta
                excess[source] += delta
                Cap_matrix[source, sink] += delta
                Cap_matrix[sink, source] -= delta
            
            # Shortest path phase - update potentials using Dijkstra's algorithm
            potential = self._update_potentials_dijkstra(
                num_nodes, source, sink, Cost_matrix, Cap_matrix, potential, delta, epsilon
            )
            
            # Update delta
            delta = 0.5 * delta
            
            # Update sigmoid costs for new delta
            for i in range(n):
                Cost_matrix[source, i] = sigmoid_cost_calc(gamma, beta, trip_amounts[i], Flow[source, i], delta, 1)
                Cost_matrix[i, source] = sigmoid_cost_calc(gamma, beta, trip_amounts[i], Flow[source, i], delta, -1)
        
        # Extract prices from final flow (sigmoid formula)
        prices = np.zeros(n)
        for i in range(n):
            if Flow[source, i] > 0 and Flow[source, i] < 1:
                prices[i] = -gamma * trip_amounts[i] * np.log(Flow[source, i]/(1-Flow[source, i])) + beta * trip_amounts[i]
            else:
                # Fallback for edge cases
                prices[i] = beta * trip_amounts[i]
        
        return prices 