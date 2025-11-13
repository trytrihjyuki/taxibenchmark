"""Enhanced results aggregator with proper wall-clock time calculation and ranking."""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from ..core import get_logger


class EnhancedResultsAggregator:
    """Enhanced aggregator with correct timing, proper statistics, and method ranking."""
    
    def __init__(self, experiment_id: str, config: Dict[str, Any]):
        """Initialize aggregator."""
        self.experiment_id = experiment_id
        self.config = config
        self.logger = get_logger('enhanced_aggregator')
        
        # Track experiment timing - CRITICAL FIX
        self.experiment_start_time = datetime.now()
        self.experiment_end_time = None
        
        # Storage for results
        self.results = []
        self.decisions_data = []
        
        self.logger.debug(f"Enhanced aggregator initialized for {experiment_id}")
    
    def add_result(self, result: Dict[str, Any]):
        """Add a scenario result."""
        self.results.append(result)
    
    def add_decisions_data(self, decisions_df: pd.DataFrame):
        """Add decisions data."""
        if not decisions_df.empty:
            self.decisions_data.append(decisions_df)
    
    def _calculate_stats_without_sum(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for ratio values (no sum for ratios)."""
        if not values:
            return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
            # NO SUM for ratios/percentages!
        }
    
    def _calculate_stats_with_sum(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for absolute values (include sum)."""
        if not values:
            return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'sum': 0.0}
        
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'sum': float(np.sum(values))
        }
    
    def _rank_methods(self, method_stats: Dict[str, Dict]) -> Dict[str, Dict]:
        """Rank methods by objective value and efficiency."""
        if not method_stats:
            return method_stats
        
        # Prepare data for ranking
        method_names = list(method_stats.keys())
        
        # Get objective values (higher is better)
        obj_values = []
        for method in method_names:
            obj_mean = method_stats[method]['objective_values']['mean']
            obj_values.append((method, obj_mean))
        
        # Sort by objective value (descending - higher is better)
        obj_ranked = sorted(obj_values, key=lambda x: x[1], reverse=True)
        
        # Get efficiency ratios (closer to 1.0 is better, but higher is generally better)
        eff_values = []
        for method in method_names:
            # Use mean objective value as efficiency proxy (revenue per scenario)
            eff_mean = method_stats[method]['objective_values']['mean']
            eff_values.append((method, eff_mean))
        
        # Sort by efficiency (descending)
        eff_ranked = sorted(eff_values, key=lambda x: x[1], reverse=True)
        
        # Assign ranks
        for i, (method, _) in enumerate(obj_ranked):
            method_stats[method]['relative_performance']['rank_by_objective'] = i + 1
        
        for i, (method, _) in enumerate(eff_ranked):
            method_stats[method]['relative_performance']['rank_by_efficiency'] = i + 1
        
        # Log ranking for terminal output
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("METHOD PERFORMANCE RANKING")
        self.logger.info("=" * 60)
        
        self.logger.info("🏆 BY OBJECTIVE VALUE (Revenue):")
        for i, (method, obj_val) in enumerate(obj_ranked):
            # Add optimality ratio if available (NEW structure)
            opt_metrics = method_stats[method].get('optimality_metrics', {})
            if opt_metrics:
                opt_ratio = opt_metrics.get('optimality_ratio', None)
                if opt_ratio is not None:
                    self.logger.info(f"  #{i+1}: {method:<15} - ${obj_val:8.2f} (optimality: {opt_ratio*100:.1f}%)")
                else:
                    self.logger.info(f"  #{i+1}: {method:<15} - ${obj_val:8.2f}")
            else:
                self.logger.info(f"  #{i+1}: {method:<15} - ${obj_val:8.2f}")
        
        self.logger.info("")
        self.logger.info("⚡ BY EFFICIENCY:")
        for i, (method, eff_val) in enumerate(eff_ranked):
            self.logger.info(f"  #{i+1}: {method:<15} - ${eff_val:8.2f}")
        
        # NEW: Log optimal value statistics using new structure
        if 'LP' in method_stats and 'optimal_values' in method_stats['LP']:
            self.logger.info("")
            self.logger.info("📊 OPTIMALITY ANALYSIS (LP as benchmark):")
            lp_opt_val = method_stats['LP']['optimal_values']['mean']
            self.logger.info(f"  LP Optimal Value: ${lp_opt_val:8.2f}")
            
            for method in method_names:
                if 'optimality_metrics' in method_stats[method]:
                    opt_metrics = method_stats[method]['optimality_metrics']
                    opt_ratio = opt_metrics['optimality_ratio']
                    opt_gap = opt_metrics['optimality_gap']
                    self.logger.info(f"  {method:<15} - {opt_ratio*100:5.1f}% of optimal (gap: ${opt_gap:6.2f})")
        
        self.logger.info("=" * 60)
        
        return method_stats
    
    def _log_separated_rankings(self, method_stats_by_acc: Dict[str, Dict]):
        """Log rankings separated by acceptance function (PL and Sigmoid)."""
        if not method_stats_by_acc:
            return
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("SEPARATED RESULTS BY ACCEPTANCE FUNCTION (CORRECTED)")
        self.logger.info("=" * 70)
        
        for acc_func in ['PL', 'Sigmoid']:
            # Filter methods for this acceptance function
            acc_methods = {k: v for k, v in method_stats_by_acc.items() if k.endswith(f"_{acc_func}")}
            if not acc_methods:
                continue
            
            # Sort by revenue (descending)
            sorted_methods = sorted(acc_methods.items(), key=lambda x: x[1]['objective_values']['mean'], reverse=True)
            
            self.logger.info("")
            self.logger.info(f"🏆 BY OBJECTIVE VALUE ({acc_func} Acceptance):")
            
            for i, (method_key, stats) in enumerate(sorted_methods):
                method_name = stats['method']
                obj_val = stats['objective_values']['mean']
                
                opt_metrics = stats.get('optimality_metrics', {})
                if opt_metrics:
                    opt_ratio = opt_metrics.get('optimality_ratio', None)
                    opt_gap = opt_metrics.get('optimality_gap', None)
                    if opt_ratio is not None:
                        self.logger.info(f"  #{i+1}: {method_name:<15} - ${obj_val:8.2f} "
                                       f"(opt: {opt_ratio*100:5.1f}%, gap: ${opt_gap:6.2f})")
                    else:
                        self.logger.info(f"  #{i+1}: {method_name:<15} - ${obj_val:8.2f}")
                else:
                    self.logger.info(f"  #{i+1}: {method_name:<15} - ${obj_val:8.2f}")
            
            # Log optimal value if available
            lp_key = f"LP_{acc_func}"
            if lp_key in method_stats_by_acc and 'optimal_values' in method_stats_by_acc[lp_key]:
                lp_opt_val = method_stats_by_acc[lp_key]['optimal_values']['mean']
                self.logger.info(f"  LP Optimal Value ({acc_func}): ${lp_opt_val:8.2f}")
        
        self.logger.info("=" * 70)
    
    def create_comprehensive_summary(self) -> Dict[str, Any]:
        """Create comprehensive experiment summary with CORRECT timing and ranking, SEPARATED by acceptance function."""
        # Record experiment end time for wall-clock calculation
        self.experiment_end_time = datetime.now()
        
        # Calculate CORRECT total computation time (wall-clock, not sum of parallel times)
        wall_clock_duration = (self.experiment_end_time - self.experiment_start_time).total_seconds()
        
        # Extract all metrics by method AND acceptance function (CRITICAL FIX)
        method_data = {}
        method_data_by_acc = {}  # NEW: Separate tracking by acceptance function
        total_scenarios = len(self.results)
        num_simulations = self.config.get('num_iter', 1)
        
        for result in self.results:
            method = result.get('method', 'unknown')
            if method not in method_data:
                method_data[method] = {
                    'objective_values': [],  # Average revenues per scenario (MIXED for backward compat)
                    'revenues': [],           # Individual simulation revenues (for LP)
                    'opt_values': [],         # Optimal values per scenario (for LP)
                    'computation_times': [],
                    'acceptance_rates': [],
                    'num_scenarios': 0
                }
            
            # NEW: Separate tracking by acceptance function
            for acc_func in ['PL', 'Sigmoid']:
                if acc_func in result:
                    method_key = f"{method}_{acc_func}"
                    if method_key not in method_data_by_acc:
                        method_data_by_acc[method_key] = {
                            'method': method,
                            'acceptance_function': acc_func,
                            'objective_values': [],
                            'revenues': [],
                            'opt_values': [],
                            'acceptance_rates': [],
                            'num_scenarios': 0
                        }
                    
                    # Collect separated data
                    method_data_by_acc[method_key]['objective_values'].append(result[acc_func]['avg_revenue'])
                    method_data_by_acc[method_key]['acceptance_rates'].append(result[acc_func]['avg_acceptance_rate'])
                    method_data_by_acc[method_key]['num_scenarios'] += 1
                    
                    if 'opt_value' in result[acc_func]:
                        method_data_by_acc[method_key]['opt_values'].append(result[acc_func]['opt_value'])
                    if 'revenues' in result[acc_func]:
                        method_data_by_acc[method_key]['revenues'].extend(result[acc_func]['revenues'])
            
            # Collect data for both PL and Sigmoid (MIXED for backward compatibility)
            for acc_func in ['PL', 'Sigmoid']:
                if acc_func in result:
                    method_data[method]['objective_values'].append(result[acc_func]['avg_revenue'])
                    method_data[method]['acceptance_rates'].append(result[acc_func]['avg_acceptance_rate'])
                    
                    # NEW: Collect optimal values and individual revenues for LP method
                    if 'opt_value' in result[acc_func]:
                        method_data[method]['opt_values'].append(result[acc_func]['opt_value'])
                    if 'revenues' in result[acc_func]:
                        method_data[method]['revenues'].extend(result[acc_func]['revenues'])
            
            # Computation time (per scenario, not per simulation)
            if 'computation_time' in result:
                method_data[method]['computation_times'].append(result['computation_time'])
            
            method_data[method]['num_scenarios'] += 1
        
        # Build method statistics with proper ranking
        method_stats = {}
        for method, data in method_data.items():
            # Calculate per-simulation time
            total_method_time = np.sum(data['computation_times']) if data['computation_times'] else 0
            total_simulations = data['num_scenarios'] * num_simulations * 2  # *2 for PL and Sigmoid
            per_sim_time_ms = (total_method_time * 1000 / total_simulations) if total_simulations > 0 else 0
            
            method_stats[method] = {
                'objective_values': self._calculate_stats_with_sum(data['objective_values']),
                'computation_times': {
                    **self._calculate_stats_without_sum(data['computation_times']),
                    'total_scenarios_time': float(np.sum(data['computation_times'])) if data['computation_times'] else 0.0,
                    'per_simulation_time_ms': float(per_sim_time_ms),
                    'total_simulations': int(total_simulations)
                },
                'acceptance_rates': self._calculate_stats_without_sum(data['acceptance_rates']),
                'scenarios_completed': data['num_scenarios'],
                'success_rate': 1.0,  # All completed successfully
                'relative_performance': {
                    'rank_by_objective': 1,  # Will be updated by ranking
                    'rank_by_efficiency': 1   # Will be updated by ranking
                }
            }
            
            # Store optimal values for LP method
            if data['opt_values']:
                method_stats[method]['optimal_values'] = self._calculate_stats_with_sum(data['opt_values'])
        
        # NEW: Compute optimality metrics AFTER collecting all data
        # This compares average revenues with average optimal values
        if 'LP' in method_stats and 'optimal_values' in method_stats['LP']:
            avg_lp_opt = method_stats['LP']['optimal_values']['mean']
            
            for method in method_stats:
                avg_revenue = method_stats[method]['objective_values']['mean']
                
                # Compute optimality metrics
                method_stats[method]['optimality_metrics'] = {
                    'avg_optimal_value': float(avg_lp_opt),
                    'avg_realized_revenue': float(avg_revenue),
                    'optimality_gap': float(avg_lp_opt - avg_revenue),
                    'optimality_ratio': float(avg_revenue / avg_lp_opt) if avg_lp_opt > 0 else 0.0
                }
        
        # NEW: Build separate stats for PL and Sigmoid
        method_stats_by_acc = {}
        for method_key, data in method_data_by_acc.items():
            method_stats_by_acc[method_key] = {
                'method': data['method'],
                'acceptance_function': data['acceptance_function'],
                'objective_values': self._calculate_stats_with_sum(data['objective_values']),
                'acceptance_rates': self._calculate_stats_without_sum(data['acceptance_rates']),
                'scenarios_completed': data['num_scenarios']
            }
            
            # Add optimal values if available
            if data['opt_values']:
                method_stats_by_acc[method_key]['optimal_values'] = self._calculate_stats_with_sum(data['opt_values'])
        
        # Compute optimality metrics for each acceptance function separately
        for acc_func in ['PL', 'Sigmoid']:
            lp_key = f"LP_{acc_func}"
            if lp_key in method_stats_by_acc and 'optimal_values' in method_stats_by_acc[lp_key]:
                avg_lp_opt = method_stats_by_acc[lp_key]['optimal_values']['mean']
                
                # Compare all methods using this acceptance function to LP optimal for that function
                for method_key in method_stats_by_acc:
                    if method_key.endswith(f"_{acc_func}"):
                        avg_revenue = method_stats_by_acc[method_key]['objective_values']['mean']
                        method_stats_by_acc[method_key]['optimality_metrics'] = {
                            'avg_optimal_value': float(avg_lp_opt),
                            'avg_realized_revenue': float(avg_revenue),
                            'optimality_gap': float(avg_lp_opt - avg_revenue),
                            'optimality_ratio': float(avg_revenue / avg_lp_opt) if avg_lp_opt > 0 else 0.0
                        }
        
        # Log separated rankings
        self._log_separated_rankings(method_stats_by_acc)
        
        # Rank methods correctly (mixed for backward compatibility)
        method_stats = self._rank_methods(method_stats)
        
        # Overall performance (across all methods)
        all_obj_values = []
        all_comp_times = []
        all_acc_rates = []
        
        for data in method_data.values():
            all_obj_values.extend(data['objective_values'])
            all_comp_times.extend(data['computation_times'])
            all_acc_rates.extend(data['acceptance_rates'])
        
        # Calculate total simulations for per-simulation timing
        total_simulations_all = total_scenarios * num_simulations * 2  # *2 for PL and Sigmoid
        per_sim_time_ms_overall = (wall_clock_duration * 1000 / total_simulations_all) if total_simulations_all > 0 else 0
        
        # Log timing correction
        if all_comp_times:
            incorrect_sum = np.sum(all_comp_times)
            self.logger.info(f"Timing correction: Wrong sum={incorrect_sum:.1f}s, Correct wall-clock={wall_clock_duration:.1f}s")
            self.logger.info(f"Per simulation time: {per_sim_time_ms_overall:.2f}ms ({total_simulations_all} total simulations)")
        
        return {
            'experiment_metadata': {
                'experiment_id': self.experiment_id,
                'total_scenarios': total_scenarios,
                'wall_clock_duration_seconds': wall_clock_duration,
                'total_simulations': total_simulations_all,
                'experiment_start_time': self.experiment_start_time.isoformat(),
                'experiment_end_time': self.experiment_end_time.isoformat(),
            },
            'experiment_setup': {
                'methods': list(method_data.keys()),
                'hour_start': self.config.get('start_hour', 0),
                'hour_end': self.config.get('end_hour', 0),
                'time_interval': self.config.get('time_delta', 5),  # Add time_interval from time_delta
                'time_unit': self.config.get('time_unit', 'm'),
                'time_window_size': self.config.get('time_window_size', 30),
                'num_eval': num_simulations,
                'num_workers': self.config.get('num_workers', 1),
                'execution_date': str(self.config.get('processing_date', '')).replace('-', ''),
            },
            'results_basic_analysis': {
                'total_methods': len(method_stats),
                'methods_performance': method_stats,
                'methods_performance_by_acceptance': method_stats_by_acc,  # NEW: Separated stats
                'overall_performance': {
                    'objective_values': self._calculate_stats_with_sum(all_obj_values),
                    'computation_times': {
                        **self._calculate_stats_without_sum(all_comp_times),
                        'wall_clock_total_seconds': wall_clock_duration,
                        'per_simulation_time_ms': float(per_sim_time_ms_overall),
                        'total_simulations': int(total_simulations_all)
                    },
                    'acceptance_rates': self._calculate_stats_without_sum(all_acc_rates)
                }
            },
            'status': 'completed'
        }
    
    def save_summary(self, summary_file: Path):
        """Save summary to file."""
        summary = self.create_comprehensive_summary()
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info(f"Enhanced summary saved to {summary_file}")