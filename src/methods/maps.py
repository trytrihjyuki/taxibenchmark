"""MAPS pricing method implementation (Baseline from Tong et al.).

This implements the MAPS (Matching And Pricing in Shared economy) algorithm
used as a baseline in Hikima et al.'s experiments.
"""

import numpy as np
from typing import Dict, Any, Set, List, Tuple
from .base import BasePricingMethod


class MAPSMethod(BasePricingMethod):
    """
    MAPS (Matching And Pricing in Shared economy) pricing method.
    
    This is a greedy algorithm that iteratively matches requesters to taxis
    while optimizing prices for each area/region.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MAPS pricing method."""
        super().__init__(config)
        
        # MAPS-specific parameters from Hikima
        self.s0_rate = config.get('maps_s0_rate', 1.5)
        self.price_delta = config.get('maps_price_delta', 0.05)  # d_rate in Hikima
        self.matching_radius = config.get('maps_matching_radius', 2.0)  # km
        
    def get_method_name(self) -> str:
        """Get method name."""
        return "MAPS"
    
    def compute_prices(
        self,
        scenario_data: Dict[str, Any],
        acceptance_function: str
    ) -> np.ndarray:
        """
        Compute prices using the MAPS algorithm optimized for specific acceptance function.
        
        This exactly replicates Hikima's MAPS implementation from experiment_PL.py and experiment_Sigmoid.py
        (lines 530-733 in PL, lines 475-683 in Sigmoid).
        
        Args:
            scenario_data: Dictionary with scenario data
            acceptance_function: 'PL' or 'Sigmoid' - which function to optimize for
            
        Returns:
            Array of prices for each requester
        """
        n_requesters = scenario_data['num_requesters']
        n_taxis = scenario_data['num_taxis']
        trip_amounts = scenario_data['trip_amounts']
        trip_distances = scenario_data.get('trip_distances', np.ones(n_requesters))
        location_ids = scenario_data.get('location_ids', np.arange(n_requesters))
        
        # Handle edge cases
        if n_requesters == 0:
            return np.array([])
        if n_taxis == 0:
            return np.zeros(n_requesters)
        
        # Get distance matrix (convert from edge weights if needed)
        if 'distance_matrix' in scenario_data:
            distance_matrix = scenario_data['distance_matrix']
        else:
            # Convert edge weights to distances (approximation)
            edge_weights = scenario_data['edge_weights']
            distance_matrix = -edge_weights / self.alpha * self.s_taxi
        
        # Run MAPS algorithm exactly like Hikima
        prices = self._maps_algorithm_hikima(
            n_requesters, n_taxis, distance_matrix,
            trip_amounts, trip_distances, location_ids, acceptance_function
        )
        
        return prices
    
    def _maps_algorithm_hikima(
        self,
        n_requesters: int,
        n_taxis: int,
        distance_matrix: np.ndarray,
        trip_amounts: np.ndarray,
        trip_distances: np.ndarray,
        location_ids: np.ndarray,
        acceptance_function: str
    ) -> np.ndarray:
        """
        Execute MAPS algorithm exactly following Hikima's implementation.
        
        This replicates lines 530-733 from experiment_PL.py and lines 475-683 from experiment_Sigmoid.py
        """
        n = n_requesters
        m = n_taxis
        
        # Hikima's data structure setup (lines 542-569 in PL, 487-515 in Sigmoid)
        ID_set = list(set(location_ids))  # Set of area IDs where at least one requester exists
        
        # Parameters to calculate acceptance rate (Hikima's lines 544-546)
        S_0_rate = self.s0_rate  # 1.5
        S_a = 1 / (S_0_rate - 1)
        S_b = 1 + 1 / (S_0_rate - 1)
        
        # Upper and lower bounds of price (Hikima's lines 548-549)
        p_max = np.amax(trip_amounts / trip_distances) * S_0_rate
        p_min = np.amin(trip_amounts / trip_distances)
        
        # Safety checks to prevent infinite loops
        if p_max <= p_min or p_max <= 0 or p_min <= 0:
            self.logger.warning(f"Invalid price bounds: p_max={p_max}, p_min={p_min}. Using fallback pricing.")
            # Fallback: simple uniform pricing
            return trip_amounts * 1.2
        
        if np.any(trip_distances <= 0) or np.any(trip_amounts <= 0):
            self.logger.warning(f"Invalid trip data found. Using fallback pricing.")
            return trip_amounts * 1.2
        
        # Initialize MAPS data structures (Hikima's lines 551-570)
        p_current = np.ones(len(ID_set)) * p_max
        current_count = np.zeros(len(ID_set))
        Nr = np.zeros(len(ID_set))
        Nr_max = np.zeros(len(ID_set))
        
        for i in range(len(ID_set)):
            Nr_max[i] += sum(location_ids == ID_set[i])
        
        dr_sum = np.zeros(len(ID_set))
        p_new = np.ones(len(ID_set)) * p_max
        new_count = np.zeros(len(ID_set))
        delta_new = np.zeros(len(ID_set))
        
        # Matching structures (Hikima's lines 566-570)
        num_requesters = len(trip_amounts)
        num_taxis = n_taxis
        edges = [set() for _ in range(num_requesters)]
        matched = [-1] * num_taxis
        matched_r = [-1] * num_requesters
        D_pre = np.zeros(len(ID_set))
        
        # Price discretization (Hikima's lines 572-575)
        d_rate = self.price_delta  # 0.05
        d_number = int(np.trunc(np.log(p_max / p_min) / np.log(1 + d_rate))) + 1
        
        # Safety check for price discretization
        if d_number <= 0 or d_number > 1000:
            self.logger.warning(f"Invalid price discretization: d_number={d_number}. Using simplified approach.")
            d_number = min(max(1, d_number), 50)  # Clamp to reasonable range
        
        # Calculate acceptance rates for each price (Hikima's lines 577-590)
        S = self._calculate_acceptance_matrix(
            ID_set, location_ids, trip_distances, trip_amounts, 
            p_max, d_rate, d_number, acceptance_function, S_a, S_b
        )
        
        # First iteration delta calculation (Hikima's lines 593-622)
        self._calculate_initial_deltas(
            ID_set, location_ids, trip_distances, Nr, dr_sum, p_max, p_min, d_rate, 
            S, p_current, current_count, delta_new, p_new, new_count, D_pre
        )
        
        # Build matching edges within 2km (Hikima's lines 624-628)
        for i in range(num_requesters):
            for j in range(num_taxis):
                if distance_matrix[i, j] <= 2.0:
                    edges[i].add(j)
        
        # Main MAPS algorithm (Hikima's lines 630-726)
        # First iteration
        feasible_flag = False
        max_index = np.argmax(delta_new)
        
        # Find augmenting path (Hikima's lines 632-638)
        for i in range(n):
            if location_ids[i] == ID_set[max_index] and matched_r[i] == -1:
                feasible_flag = self._dfs_hikima(i, set(), edges, matched)
                if feasible_flag:
                    matched_r[i] = 1
                    break
        
        # Update if feasible path found (Hikima's lines 640-676)
        if feasible_flag:
            Nr[max_index] += 1
            p_current[max_index] = p_new[max_index]
            current_count[max_index] = new_count[max_index]
            
            # Recalculate delta for next iteration (Hikima's lines 649-672)
            if Nr[max_index] + 1 <= Nr_max[max_index]:
                self._update_delta_for_area(
                    max_index, ID_set, location_ids, trip_distances, Nr, dr_sum,
                    p_max, p_min, d_rate, S, p_current, current_count, 
                    delta_new, p_new, new_count
                )
            else:
                delta_new[max_index] = -1
                p_new[max_index] = -1
                new_count[max_index] = -1
        else:
            delta_new[max_index] = -1
        
        # Main iteration loop (Hikima's lines 678-725)
        iter_num = 0
        max_iterations = 10000  # Safety limit to prevent infinite loops
        
        while iter_num < max_iterations:
            feasible_flag = False
            max_index = np.argmax(delta_new)
            
            if delta_new[max_index] <= 0:
                break
                
            # Find augmenting path
            for i in range(n):
                if location_ids[i] == ID_set[max_index] and matched_r[i] == -1:
                    feasible_flag = self._dfs_hikima(i, set(), edges, matched)
                    if feasible_flag:
                        matched_r[i] = 1
                        break
            
            if feasible_flag:
                Nr[max_index] += 1
                p_current[max_index] = p_new[max_index]
                current_count[max_index] = new_count[max_index]
                
                if Nr[max_index] + 1 <= Nr_max[max_index]:
                    self._update_delta_for_area(
                        max_index, ID_set, location_ids, trip_distances, Nr, dr_sum,
                        p_max, p_min, d_rate, S, p_current, current_count,
                        delta_new, p_new, new_count
                    )
                else:
                    delta_new[max_index] = -1
                    p_new[max_index] = -1
                    new_count[max_index] = -1
            else:
                delta_new[max_index] = -1
                
            iter_num += 1
        
        # Set final prices (Hikima's lines 727-733)
        price_MAPS = np.zeros(n)
        for i in range(n):
            r = location_ids[i]
            for h in range(len(ID_set)):
                if ID_set[h] == r:
                    price_MAPS[i] = p_current[h] * trip_distances[i]
                    break
        
        return price_MAPS
    
    def _calculate_acceptance_matrix(
        self, ID_set, location_ids, trip_distances, trip_amounts, 
        p_max, d_rate, d_number, acceptance_function, S_a, S_b
    ):
        """Calculate acceptance rate matrix S exactly like Hikima (lines 577-590)."""
        S = np.ones([len(ID_set), d_number]) * np.inf
        
        for r in range(len(ID_set)):
            p_tmp = p_max
            for k in range(d_number):
                accept_sum = 0
                
                # Get requesters in this area
                area_mask = location_ids == ID_set[r]
                area_distances = trip_distances[area_mask]
                area_amounts = trip_amounts[area_mask]
                
                for o_dist, to_am in zip(area_distances, area_amounts):
                    if acceptance_function == 'PL':
                        # Hikima's PL acceptance (lines 583-587)
                        acceptance_rate = -S_a / to_am * p_tmp * o_dist + S_b
                        if acceptance_rate > 0:
                            if acceptance_rate < 1:
                                accept_sum += acceptance_rate
                            else:
                                accept_sum += 1
                    else:  # Sigmoid
                        # Hikima's Sigmoid acceptance (line 531)
                        exponent = (-p_tmp * o_dist + self.sigmoid_beta * to_am) / (self.sigmoid_gamma * to_am)
                        accept_sum += 1 - (1 / (1 + np.exp(exponent)))
                
                if len(area_distances) > 0:
                    S[r, k] = accept_sum / len(area_distances)
                else:
                    S[r, k] = 0
                    
                p_tmp = p_tmp / (1 + d_rate)
        
        return S
    
    def _calculate_initial_deltas(
        self, ID_set, location_ids, trip_distances, Nr, dr_sum, p_max, p_min, d_rate,
        S, p_current, current_count, delta_new, p_new, new_count, D_pre
    ):
        """Calculate initial deltas exactly like Hikima (lines 593-622)."""
        for r in range(len(ID_set)):
            dr_sum[r] = np.sum(trip_distances * (location_ids == ID_set[r]))
            
            # Calculate dr_Nr_sum (Hikima's lines 595-606)
            dr_Nr_sum = 0
            r_count = 0
            count = 0
            
            while count < len(location_ids):
                if location_ids[count] == ID_set[r]:
                    dr_Nr_sum += trip_distances[count]
                    r_count += 1
                if r_count == Nr[r] + 1:
                    break
                count += 1
            
            # Find optimal price (Hikima's lines 607-618)
            value_tmp = 0.0
            p_tmp = p_max
            d_count = 0
            p_opt = p_max
            opt_d_count = 0
            
            while p_tmp >= p_min:
                C = dr_sum[r]
                D = dr_Nr_sum
                current_value = np.amin([
                    C * (p_tmp - self.alpha / self.s_taxi) * S[r, d_count],
                    D * (p_tmp - self.alpha / self.s_taxi)
                ])
                
                if value_tmp < current_value:
                    value_tmp = current_value
                    p_opt = p_tmp
                    opt_d_count = d_count
                    
                p_tmp = p_tmp / (1 + d_rate)
                d_count += 1
            
            # Calculate delta (Hikima's line 619)
            delta_new[r] = ((p_opt - self.alpha / self.s_taxi) * S[r, opt_d_count] - 
                           (p_current[r] - self.alpha / self.s_taxi) * S[r, int(current_count[r])])
            p_new[r] = p_opt
            new_count[r] = opt_d_count
            D_pre[r] = D
    
    def _update_delta_for_area(
        self, max_index, ID_set, location_ids, trip_distances, Nr, dr_sum,
        p_max, p_min, d_rate, S, p_current, current_count, delta_new, p_new, new_count
    ):
        """Update delta for specific area exactly like Hikima (lines 649-672)."""
        # Calculate D for Nr[max_index] + 1 (Hikima's lines 651-659)
        C = dr_sum[max_index]
        D = 0
        sum_num = 0
        
        for i in range(len(location_ids)):
            if sum_num > Nr[max_index] + 1:
                break
            if location_ids[i] == ID_set[max_index]:
                D += trip_distances[i]
                sum_num += 1
        
        # Find optimal price (Hikima's lines 660-666)
        value_tmp = 0.0
        p_tmp = p_max
        d_count = 0
        p_opt = p_max
        opt_d_count = 0
        
        while p_tmp >= p_min:
            current_value = np.amin([
                C * (p_tmp - self.alpha / self.s_taxi) * S[max_index, d_count],
                D * (p_tmp - self.alpha / self.s_taxi)
            ])
            
            if value_tmp < current_value:
                value_tmp = current_value
                p_opt = p_tmp
                opt_d_count = d_count
                
            p_tmp = p_tmp / (1 + d_rate)
            d_count += 1
        
        # Update delta (Hikima's line 667)
        delta_new[max_index] = ((p_opt - self.alpha / self.s_taxi) * S[max_index, opt_d_count] - 
                               (p_current[max_index] - self.alpha / self.s_taxi) * S[max_index, int(current_count[max_index])])
        p_new[max_index] = p_opt
        new_count[max_index] = opt_d_count
    
    def _dfs_hikima(self, v, visited, edges, matched):
        """DFS for augmenting path exactly like Hikima (lines 161-169)."""
        for u in edges[v]:
            if u in visited:
                continue
            visited.add(u)
            if matched[u] == -1 or self._dfs_hikima(matched[u], visited, edges, matched):
                matched[u] = v
                return True
        return False 