"""Base class for pricing methods with dual acceptance evaluation."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from time import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import networkx as nx
import traceback

from ..core import get_logger


class BasePricingMethod(ABC):
    """Abstract base class for all pricing methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pricing method.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(f"methods.{self.get_method_name()}")
        
        # Extract common parameters
        self.alpha = config.get('alpha', 18.0)
        self.s_taxi = config.get('s_taxi', 25.0)
        
        # Sigmoid parameters - aligned with Hikima paper
        self.sigmoid_beta = config.get('sigmoid_beta', 1.3)
        # gamma = (0.3*sqrt(3)/pi) from experiment_Sigmoid.py line 49 - EXACT match
        import math
        self.sigmoid_gamma = config.get('sigmoid_gamma', (0.3 * math.sqrt(3) / math.pi))
    
    @abstractmethod
    def compute_prices(
        self,
        scenario_data: Dict[str, Any],
        acceptance_function: str
    ):
        """
        Compute prices for all requesters optimized for specific acceptance function.
        
        Args:
            scenario_data: Dictionary with scenario data
            acceptance_function: 'PL' or 'Sigmoid' - which function to optimize for
            
        Returns:
            Either:
            - np.ndarray: Array of prices for each requester (backward compatible)
            - Tuple[np.ndarray, float]: (prices, optimal_objective_value) for methods that compute optimal
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of the pricing method."""
        pass
    
    def execute(
        self,
        scenario_data: Dict[str, Any],
        num_simulations: int = 1
    ) -> Dict[str, Any]:
        """
        Execute the pricing method separately for each acceptance function.
        This ensures proper alignment with Hikima's experimental setup.
        
        Args:
            scenario_data: Dictionary with scenario data
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with results for both acceptance functions
        """
        # Results dictionary
        results = {
            'method': self.get_method_name(),
            'num_requesters': scenario_data.get('num_requesters', 0),
            'num_taxis': scenario_data.get('num_taxis', 0),
        }
        
        # Get valuations
        valuations = scenario_data.get('trip_amounts', np.array([]))
        
        # CRITICAL: Optimize and evaluate separately for each acceptance function
        for acceptance_function in ['PL', 'Sigmoid']:
            func_start = time()
            self.logger.info(f"Starting {acceptance_function} optimization (N={len(valuations)}, sims={num_simulations})")
            
            # Compute prices optimized for this specific acceptance function
            price_start = time()
            compute_result = self.compute_prices(scenario_data, acceptance_function)
            
            # Handle both return types: just prices or (prices, opt_value)
            if isinstance(compute_result, tuple):
                prices, opt_value = compute_result
            else:
                prices = compute_result
                opt_value = None  # Method doesn't compute optimal value
            
            price_time = time() - price_start
            
            self.logger.debug(f"{acceptance_function} price computation: {price_time:.3f}s")
            if opt_value is not None:
                self.logger.info(f"{acceptance_function} optimal objective value: ${opt_value:.2f}")
            
            # Record computation time
            computation_time = time() - func_start
            
            # Compute acceptance probabilities with the same function
            prob_start = time()
            acceptance_probs = self._compute_acceptance_probs_specific(
                prices, valuations, acceptance_function
            )
            prob_time = time() - prob_start
            
            self.logger.info(f"{acceptance_function} probability computation: {prob_time:.3f}s")
            
            # Run Monte Carlo simulations using Hikima's approach with optional parallelization
            sim_start = time()
            if num_simulations > 1:
                # Determine if we should parallelize simulations
                num_workers = scenario_data.get('num_workers', 1)
                use_parallel = num_simulations > 1 and num_workers > 1
                
                if use_parallel:
                    self.logger.info(f"{acceptance_function} using PARALLEL simulations: {num_simulations} sims across {num_workers} workers")
                    revenues, matching_rates, acceptance_rates = self._run_parallel_simulations(
                        prices, acceptance_probs, scenario_data, num_simulations, num_workers, acceptance_function
                    )
                else:
                    # Sequential simulations (original approach)
                    if num_workers > 1:
                        self.logger.info(f"{acceptance_function} using SEQUENTIAL simulations: {num_simulations} <= {num_workers} workers threshold")
                    else:
                        self.logger.info(f"{acceptance_function} using SEQUENTIAL simulations: single worker mode")
                    revenues = []
                    matching_rates = []
                    acceptance_rates = []
                    
                    for sim_idx in range(num_simulations):
                        # Simulate acceptance decisions exactly like Hikima
                        accepted = np.zeros(len(acceptance_probs))
                        for i in range(len(acceptance_probs)):
                            tmp = np.random.rand()
                            if tmp < acceptance_probs[i]:
                                accepted[i] = 1
                        
                        # Calculate objective value using Hikima's value_eval function
                        opt_value, matched_edges, rewards = self._compute_objective_value_hikima(
                            prices, accepted, scenario_data['edge_weights'],
                            scenario_data['num_requesters'], scenario_data['num_taxis']
                        )
                        
                        # Calculate metrics like Hikima
                        revenue = opt_value  # This is the total objective value
                        matching_rate = len(matched_edges) / len(accepted) if len(accepted) > 0 else 0
                        acceptance_rate = np.mean(accepted)
                        
                        revenues.append(revenue)
                        matching_rates.append(matching_rate)
                        acceptance_rates.append(acceptance_rate)
                        
                        # Log progress for long simulations
                        if sim_idx > 0 and (sim_idx + 1) % 10 == 0:
                            elapsed = time() - sim_start
                            self.logger.debug(f"{acceptance_function} simulation {sim_idx+1}/{num_simulations} ({elapsed:.1f}s)")
                
                # Aggregate results - use the last simulation's matching for decision tracking
                # Get matching results from the last simulation
                final_accepted = np.zeros(len(acceptance_probs))
                for i in range(len(acceptance_probs)):
                    tmp = np.random.rand()
                    if tmp < acceptance_probs[i]:
                        final_accepted[i] = 1
                
                _, final_matched_edges, _ = self._compute_objective_value_hikima(
                    prices, final_accepted, scenario_data['edge_weights'],
                    scenario_data['num_requesters'], scenario_data['num_taxis']
                )
                
                # Create matching results array for decision tracking
                matching_results = np.full(len(prices), -1, dtype=int)  # -1 means not matched
                for (i, j) in final_matched_edges:
                    # Handle edge order like Hikima
                    if i > j:
                        requester_idx = j
                    else:
                        requester_idx = i
                    if requester_idx < len(matching_results):
                        matching_results[requester_idx] = j if i <= j else (i - scenario_data['num_requesters'])
                
                func_results = {
                    'prices': prices.tolist() if len(prices) > 0 else [],
                    'acceptance_probs': acceptance_probs.tolist() if len(acceptance_probs) > 0 else [],
                    'matching_results': matching_results.tolist(),  # Add matching results for decision tracking
                    'avg_revenue': float(np.mean(revenues)),
                    'std_revenue': float(np.std(revenues)),
                    'avg_matching_rate': float(np.mean(matching_rates)),
                    'std_matching_rate': float(np.std(matching_rates)),
                    'avg_acceptance_rate': float(np.mean(acceptance_rates)),
                    'computation_time': computation_time,
                    'num_simulations': num_simulations
                }
                
                # NEW: Store opt_value array for LP method (for aggregation later)
                if opt_value is not None:
                    func_results['opt_value'] = float(opt_value)
                    func_results['revenues'] = revenues  # Store individual simulation revenues
                    # Note: Optimality metrics will be computed during aggregation
            else:
                # Single run using Hikima's approach
                accepted = np.zeros(len(acceptance_probs))
                for i in range(len(acceptance_probs)):
                    tmp = np.random.rand()
                    if tmp < acceptance_probs[i]:
                        accepted[i] = 1
                
                # Calculate objective value using Hikima's value_eval function
                realized_value, matched_edges, rewards = self._compute_objective_value_hikima(
                    prices, accepted, scenario_data['edge_weights'],
                    scenario_data['num_requesters'], scenario_data['num_taxis']
                )
                
                # Create matching results array for decision tracking
                matching_results = np.full(len(prices), -1, dtype=int)  # -1 means not matched
                for (i, j) in matched_edges:
                    # Handle edge order like Hikima
                    if i > j:
                        requester_idx = j
                    else:
                        requester_idx = i
                    if requester_idx < len(matching_results):
                        matching_results[requester_idx] = j if i <= j else (i - scenario_data['num_requesters'])
                
                func_results = {
                    'prices': prices.tolist() if len(prices) > 0 else [],
                    'acceptance_probs': acceptance_probs.tolist() if len(acceptance_probs) > 0 else [],
                    'matching_results': matching_results.tolist(),  # Add matching results for decision tracking
                    'avg_revenue': float(realized_value),
                    'std_revenue': 0.0,
                    'avg_matching_rate': float(len(matched_edges) / len(accepted)) if len(accepted) > 0 else 0,
                    'std_matching_rate': 0.0,
                    'avg_acceptance_rate': float(np.mean(accepted)) if len(accepted) > 0 else 0,
                    'computation_time': computation_time,
                    'num_simulations': 1
                }
                
                # NEW: Store opt_value for LP method (for aggregation later)
                if opt_value is not None:
                    func_results['opt_value'] = float(opt_value)
                    func_results['revenues'] = [realized_value]  # Single simulation revenue
                    # Note: Optimality metrics will be computed during aggregation
            
            sim_time = time() - sim_start
            func_total_time = time() - func_start
            self.logger.info(f"{acceptance_function} completed in {func_total_time:.2f}s (sims: {sim_time:.2f}s, avg_revenue: ${func_results['avg_revenue']:.2f})")
            
            # Store results for this acceptance function
            results[acceptance_function] = func_results
        
        # Store common prices info (from PL optimization for backward compatibility)
        results['prices'] = results['PL']['prices']
        results['computation_time'] = results['PL']['computation_time'] + results['Sigmoid']['computation_time']
        
        return results
    
    def _compute_acceptance_probs(
        self,
        prices: np.ndarray,
        valuations: np.ndarray
    ) -> np.ndarray:
        """
        Compute acceptance probabilities (deprecated - use _compute_acceptance_probs_specific).
        """
        # Default to PL for backward compatibility
        return self._compute_acceptance_probs_specific(prices, valuations, 'PL')
    
    def _compute_acceptance_probs_specific(
        self,
        prices: np.ndarray,
        valuations: np.ndarray,
        acceptance_function: str
    ) -> np.ndarray:
        """
        Compute acceptance probabilities for specific acceptance function.
        
        Args:
            prices: Offered prices
            valuations: Customer valuations (trip amounts)
            acceptance_function: 'PL' or 'Sigmoid'
            
        Returns:
            Array of acceptance probabilities
        """
        if len(prices) == 0 or len(valuations) == 0:
            return np.array([])
        
        if acceptance_function == 'PL':
            # Piecewise linear acceptance function
            # P(accept) = -2/valuation * price + 3
            probs = -2.0 / valuations * prices + 3.0
            probs = np.clip(probs, 0, 1)
            
        elif acceptance_function == 'Sigmoid':
            # Sigmoid acceptance function - exact match to Hikima implementation
            # From experiment_Sigmoid.py line 739 and 782-783:
            # P(accept) = 1 - (1 / (1 + exp((-price + beta*valuation) / (gamma*valuation))))
            beta = self.sigmoid_beta
            gamma = self.sigmoid_gamma
            
            exponent = (-prices + beta * valuations) / (gamma * np.abs(valuations))
            probs = 1.0 - (1.0 / (1.0 + np.exp(exponent)))
            
        else:
            raise ValueError(f"Unknown acceptance function: {acceptance_function}")
        
        return probs
    
    def _simulate_matching(
        self,
        n_requesters: int,
        n_taxis: int,
        edge_weights: np.ndarray,
        acceptance_decisions: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Simulate matching between requesters and taxis.
        
        Args:
            n_requesters: Number of requesters
            n_taxis: Number of taxis
            edge_weights: Edge weight matrix
            acceptance_decisions: Binary acceptance decisions
            
        Returns:
            Tuple of (matched_requesters, matched_taxis)
        """
        # Simple greedy matching
        matched_requesters = []
        matched_taxis = []
        available_taxis = set(range(n_taxis))
        
        # Sort requesters by acceptance decision and edge weight
        requester_order = []
        for i in range(n_requesters):
            if acceptance_decisions[i]:
                best_taxi = max(available_taxis, key=lambda j: edge_weights[i, j]) if available_taxis else None
                if best_taxi is not None:
                    requester_order.append((edge_weights[i, best_taxi], i, best_taxi))
        
        requester_order.sort(reverse=True)
        
        for _, requester, taxi in requester_order:
            if taxi in available_taxis:
                matched_requesters.append(requester)
                matched_taxis.append(taxi)
                available_taxis.remove(taxi)
        
        return matched_requesters, matched_taxis
    
    def _calculate_metrics(
        self,
        prices: np.ndarray,
        acceptance_decisions: np.ndarray,
        matched_requesters: List[int]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            prices: Offered prices
            acceptance_decisions: Binary acceptance decisions
            matched_requesters: List of matched requester indices
            
        Returns:
            Dictionary with metrics
        """
        n_requesters = len(prices)
        
        # Revenue: sum of prices for matched requesters who accepted
        revenue = sum(prices[r] for r in matched_requesters if acceptance_decisions[r])
        
        # Matching rate: fraction of requesters matched
        matching_rate = len(matched_requesters) / n_requesters if n_requesters > 0 else 0
        
        # Acceptance rate: fraction of requesters who accepted
        acceptance_rate = np.mean(acceptance_decisions) if len(acceptance_decisions) > 0 else 0
        
        # Profit (simplified - could include costs)
        profit = revenue
        
        return {
            'revenue': float(revenue),
            'profit': float(profit),
            'matching_rate': float(matching_rate),
            'acceptance_rate': float(acceptance_rate),
            'num_matched': len(matched_requesters)
        } 

    def _compute_objective_value_hikima(
        self,
        prices: np.ndarray,
        acceptance_decisions: np.ndarray,
        edge_weights: np.ndarray,
        n_requesters: int,
        n_taxis: int
    ) -> Tuple[float, List, np.ndarray]:
        """
        Compute objective value using Hikima's exact value_eval function.
        
        This replicates the value_eval function from experiment_PL.py and experiment_Sigmoid.py
        (lines 172-195) to ensure 1:1 alignment.
        """
        import networkx as nx
        
        # Build bipartite graph exactly like Hikima
        group1 = range(n_requesters)
        group2 = range(n_requesters, n_requesters + n_taxis)
        g_post = nx.Graph()
        g_post.add_nodes_from(group1, bipartite=1)
        g_post.add_nodes_from(group2, bipartite=0)
        
        # Add edges for accepted requesters only
        for i in range(n_requesters):
            if acceptance_decisions[i] == 1:
                for j in range(n_taxis):
                    val = prices[i] + edge_weights[i, j]  # Hikima's line 182
                    g_post.add_edge(i, j + n_requesters, weight=val)
        
        # Find maximum weight matching (Hikima's line 183)
        matched_edges = nx.max_weight_matching(g_post)
        
        # Calculate objective value and rewards exactly like Hikima (lines 184-194)
        opt_value = 0.0
        reward = np.zeros(n_requesters)
        
        for (i, j) in matched_edges:
            # Handle edge order (Hikima's lines 186-192)
            if i > j:
                jtmp = j
                j = i - n_requesters
                i = jtmp
            else:
                j = j - n_requesters
            
            # Calculate value: price + edge_weight (Hikima's line 193)
            opt_value += prices[i] + edge_weights[i, j]
            reward[i] = prices[i] + edge_weights[i, j]
        
        return opt_value, matched_edges, reward
    
    def _run_parallel_simulations(
        self,
        prices: np.ndarray,
        acceptance_probs: np.ndarray,
        scenario_data: Dict[str, Any],
        num_simulations: int,
        num_workers: int,
        acceptance_function: str
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Run Monte Carlo simulations in parallel across workers.
        
        Args:
            prices: Computed prices for requesters
            acceptance_probs: Acceptance probabilities 
            scenario_data: Scenario data including edge weights
            num_simulations: Total number of simulations to run
            num_workers: Number of parallel workers
            acceptance_function: 'PL' or 'Sigmoid'
            
        Returns:
            Tuple of (revenues, matching_rates, acceptance_rates) lists
        """
        
        # Distribute simulations across workers
        sims_per_worker = math.ceil(num_simulations / num_workers)
        worker_tasks = []
        
        # Create worker tasks with serializable data
        base_seed = 42  # Base seed for reproducibility
        current_sim = 0
        
        for worker_id in range(num_workers):
            if current_sim >= num_simulations:
                break
                
            # Calculate how many simulations this worker should run
            worker_sims = min(sims_per_worker, num_simulations - current_sim)
            
            # Create unique seed range for this worker to avoid duplicates
            worker_seed_start = base_seed + (worker_id * 10000)
            
            # Prepare serializable task data (convert numpy arrays to lists)
            worker_task = {
                'worker_id': worker_id,
                'num_sims': worker_sims,
                'seed_start': worker_seed_start,
                'prices': prices.tolist(),
                'acceptance_probs': acceptance_probs.tolist(),
                'edge_weights': scenario_data['edge_weights'].tolist(),
                'num_requesters': scenario_data['num_requesters'],
                'num_taxis': scenario_data['num_taxis'],
                'acceptance_function': acceptance_function
            }
            worker_tasks.append(worker_task)
            current_sim += worker_sims
        
        self.logger.debug(f"{acceptance_function} distributing {num_simulations} simulations across {len(worker_tasks)} workers")
        
        # Run simulations in parallel with proper error handling
        all_revenues = []
        all_matching_rates = []
        all_acceptance_rates = []
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all worker tasks
                future_to_worker = {
                    executor.submit(_run_worker_simulations, task): task['worker_id'] 
                    for task in worker_tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_worker, timeout=900):  # 15 minute total timeout
                    worker_id = future_to_worker[future]
                    try:
                        worker_results = future.result(timeout=900)  # 15 minute timeout per worker
                        
                        # Validate worker results
                        if not isinstance(worker_results, dict):
                            raise ValueError(f"Worker {worker_id} returned invalid result type: {type(worker_results)}")
                        
                        required_keys = ['revenues', 'matching_rates', 'acceptance_rates']
                        for key in required_keys:
                            if key not in worker_results:
                                raise ValueError(f"Worker {worker_id} missing key: {key}")
                        
                        # Aggregate worker results
                        all_revenues.extend(worker_results['revenues'])
                        all_matching_rates.extend(worker_results['matching_rates'])
                        all_acceptance_rates.extend(worker_results['acceptance_rates'])
                        
                        self.logger.debug(f"{acceptance_function} worker {worker_id} completed {len(worker_results['revenues'])} simulations")
                        
                    except FutureTimeoutError:
                        self.logger.error(f"{acceptance_function} worker {worker_id} timed out")
                        raise  # Don't continue if workers are timing out
                    except Exception as e:
                        self.logger.error(f"{acceptance_function} worker {worker_id} failed: {e}")
                        self.logger.error(f"Full traceback: {traceback.format_exc()}")
                        raise  # Don't swallow errors - fail fast
                        
        except Exception as e:
            self.logger.error(f"Parallel simulation failed: {e}, falling back to sequential")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            # Fallback to sequential execution
            return self._run_sequential_simulations(
                prices, acceptance_probs, scenario_data, num_simulations, acceptance_function
            )
        
        # Validate we have the expected number of results
        if len(all_revenues) != num_simulations:
            self.logger.error(f"Expected {num_simulations} results, got {len(all_revenues)}")
            raise ValueError(f"Parallel simulation incomplete: expected {num_simulations}, got {len(all_revenues)}")
        
        return all_revenues, all_matching_rates, all_acceptance_rates
    
    def _run_sequential_simulations(
        self,
        prices: np.ndarray,
        acceptance_probs: np.ndarray,
        scenario_data: Dict[str, Any],
        num_simulations: int,
        acceptance_function: str
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Run simulations sequentially (fallback method).
        
        Returns:
            Tuple of (revenues, matching_rates, acceptance_rates) lists
        """
        revenues = []
        matching_rates = []
        acceptance_rates = []
        
        for sim_idx in range(num_simulations):
            # Simulate acceptance decisions exactly like Hikima
            accepted = np.zeros(len(acceptance_probs))
            for i in range(len(acceptance_probs)):
                tmp = np.random.rand()
                if tmp < acceptance_probs[i]:
                    accepted[i] = 1
            
            # Calculate objective value using Hikima's value_eval function
            opt_value, matched_edges, rewards = self._compute_objective_value_hikima(
                prices, accepted, scenario_data['edge_weights'],
                scenario_data['num_requesters'], scenario_data['num_taxis']
            )
            
            # Calculate metrics like Hikima
            revenue = opt_value  # This is the total objective value
            matching_rate = len(matched_edges) / len(accepted) if len(accepted) > 0 else 0
            acceptance_rate = np.mean(accepted)
            
            revenues.append(revenue)
            matching_rates.append(matching_rate)
            acceptance_rates.append(acceptance_rate)
        
        return revenues, matching_rates, acceptance_rates


def _run_worker_simulations(task: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Run simulations for a single worker (must be at module level for pickling).
    
    Args:
        task: Dictionary with worker configuration and data
        
    Returns:
        Dictionary with simulation results
    """
    import numpy as np
    import networkx as nx
    
    try:
        # Extract task parameters
        worker_id = task['worker_id']
        num_sims = task['num_sims']
        seed_start = task['seed_start']
        prices = np.array(task['prices'])
        acceptance_probs = np.array(task['acceptance_probs'])
        edge_weights = np.array(task['edge_weights'])
        num_requesters = task['num_requesters']
        num_taxis = task['num_taxis']
        acceptance_function = task['acceptance_function']
        
        # Initialize worker-specific random state
        np.random.seed(seed_start)
        
        # Storage for results
        revenues = []
        matching_rates = []
        acceptance_rates = []
        
        # Run simulations for this worker
        for sim_idx in range(num_sims):
            # Use different seed for each simulation to ensure variety
            sim_seed = seed_start + sim_idx
            np.random.seed(sim_seed)
            
            # Simulate acceptance decisions exactly like Hikima
            accepted = np.zeros(len(acceptance_probs))
            for i in range(len(acceptance_probs)):
                tmp = np.random.rand()
                if tmp < acceptance_probs[i]:
                    accepted[i] = 1
            
            # Calculate objective value using Hikima's value_eval function (replicated here)
            opt_value, matched_edges = _compute_objective_value_worker(
                prices, accepted, edge_weights, num_requesters, num_taxis
            )
            
            # Calculate metrics like Hikima
            revenue = opt_value
            matching_rate = len(matched_edges) / len(accepted) if len(accepted) > 0 else 0
            acceptance_rate = np.mean(accepted)
            
            revenues.append(revenue)
            matching_rates.append(matching_rate)
            acceptance_rates.append(acceptance_rate)
        
        return {
            'revenues': revenues,
            'matching_rates': matching_rates,
            'acceptance_rates': acceptance_rates,
            'worker_id': worker_id,
            'num_sims_completed': len(revenues)
        }
        
    except Exception as e:
        # Log error and re-raise so it gets caught by the main process
        import sys
        print(f"Worker {task.get('worker_id', 'unknown')} error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise


def _compute_objective_value_worker(
    prices: np.ndarray,
    acceptance_decisions: np.ndarray,
    edge_weights: np.ndarray,
    num_requesters: int,
    num_taxis: int
) -> Tuple[float, List]:
    """
    Compute objective value for worker (must be at module level for pickling).
    Replicates Hikima's value_eval function.
    """
    import networkx as nx
    
    # Build bipartite graph exactly like Hikima
    group1 = range(num_requesters)
    group2 = range(num_requesters, num_requesters + num_taxis)
    g_post = nx.Graph()
    g_post.add_nodes_from(group1, bipartite=1)
    g_post.add_nodes_from(group2, bipartite=0)
    
    # Add edges for accepted requesters only
    for i in range(num_requesters):
        if acceptance_decisions[i] == 1:
            for j in range(num_taxis):
                val = prices[i] + edge_weights[i, j]  # Hikima's line 182
                g_post.add_edge(i, j + num_requesters, weight=val)
    
    # Find maximum weight matching (Hikima's line 183)
    matched_edges = nx.max_weight_matching(g_post)
    
    # Calculate objective value exactly like Hikima (lines 184-194)
    opt_value = 0.0
    
    for (i, j) in matched_edges:
        # Handle edge order (Hikima's lines 186-192)
        if i > j:
            jtmp = j
            j = i - num_requesters
            i = jtmp
        else:
            j = j - num_requesters
        
        # Calculate value: price + edge_weight (Hikima's line 193)
        opt_value += prices[i] + edge_weights[i, j]
    
    return opt_value, matched_edges 
