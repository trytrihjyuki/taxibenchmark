"""Enhanced experiment runner with S3 storage and improved decision tracking."""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add imports for Hikima alignment
from typing import Tuple

# Add src to path if needed
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from ..core import ExperimentConfig, get_logger
from ..core.types import TimeWindow, Borough, PricingMethod
from ..data import S3DataLoader, S3ResultsUploader, DataProcessor
from ..methods import PricingMethodFactory


def run_single_scenario(args: Tuple[Dict, Dict, int]) -> Dict:
    """
    Run a single scenario (time window + method + borough).
    This function is designed to be pickle-able for multiprocessing.
    
    Args:
        args: Tuple of (scenario_data, config_dict, num_simulations)
    
    Returns:
        Results dictionary with enhanced decision tracking
    """
    try:
        from datetime import datetime
        from ..core import get_logger
        
        scenario_data, config_dict, num_simulations = args
        
        # Get logger for this process
        logger = get_logger('scenario')
        
        method_name = scenario_data['method']
        borough = scenario_data['borough']
        n_req = scenario_data['num_requesters']
        n_tax = scenario_data['num_taxis']
        
        start_time = datetime.now()
        logger.info(f"Starting scenario for {borough} (N={n_req}, M={n_tax}, sims={num_simulations})")
        
        # Create method factory with config
        factory_start = datetime.now()
        factory = PricingMethodFactory(config_dict)
        method = factory.create(PricingMethod(scenario_data['method']))
        factory_time = (datetime.now() - factory_start).total_seconds()
        
        logger.debug(f"[{method_name}] Method creation took {factory_time:.3f}s")
        
        # Run method (returns results for both PL and Sigmoid)
        execution_start = datetime.now()
        results = method.execute(scenario_data, num_simulations)
        execution_time = (datetime.now() - execution_start).total_seconds()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Extract result metrics for logging
        pl_revenue = results.get('PL', {}).get('avg_revenue', 0)
        sigmoid_revenue = results.get('Sigmoid', {}).get('avg_revenue', 0)
        
        # Enhanced logging without redundant prefixes
        logger.info(f"Completed {borough} in {total_time:.2f}s (exec: {execution_time:.2f}s) - PL: ${pl_revenue:.2f}, Sigmoid: ${sigmoid_revenue:.2f}")
        
        # Add metadata including computation time
        results['time_window'] = scenario_data['time_window']
        results['borough'] = scenario_data['borough']
        results['scenario_id'] = scenario_data['scenario_id']
        results['computation_time'] = execution_time
        results['factory_time'] = factory_time
        results['total_time'] = total_time
        
        return results
        
    except Exception as e:
        # Return error result instead of crashing the worker process
        from ..core import get_logger
        logger = get_logger('scenario')
        logger.error(f"Scenario execution failed: {e}", exc_info=True)
        return {
            'method': args[0].get('method', 'unknown') if len(args) > 0 and isinstance(args[0], dict) else 'unknown',
            'borough': args[0].get('borough', 'unknown') if len(args) > 0 and isinstance(args[0], dict) else 'unknown',
            'error': str(e),
            'status': 'failed',
            'PL': {'avg_revenue': 0.0, 'prices': [], 'acceptance_probs': []},
            'Sigmoid': {'avg_revenue': 0.0, 'prices': [], 'acceptance_probs': []}
        }


class ExperimentRunner:
    """Enhanced experiment runner with S3 storage and improved decision tracking."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner."""
        self.config = config
        self.logger = get_logger("experiment.runner")
        
        # Set random seeds for reproducibility (like Hikima's deterministic approach)
        np.random.seed(42)
        
        # Store trip data for scenario creation (will be populated in run method)
        self.all_trip_data = None
        
        # Initialize S3 uploader
        self.s3_uploader = S3ResultsUploader(config)
        
        # Create local output directory (for temporary storage before S3 upload)
        exec_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        proc_date = config.processing_date.strftime('%Y%m%d')
        self.experiment_id = config.get_experiment_id()
        self.output_dir = Path('experiments') / f"run_{exec_date}_{proc_date}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create decisions subdirectory
        self.decisions_dir = self.output_dir / 'decisions'
        self.decisions_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment with S3 storage."""
        start_time = datetime.now()
        
        # Store start time for correct wall-clock calculation
        self.experiment_start_time = start_time
        
        # Load data using experiment-aligned approach  
        self.logger.info("Loading data from S3...")
        try:
            # Use experiment-aligned loader instead of standard S3DataLoader
            from ..data.data_loader import ExperimentDataLoader
            experiment_loader = ExperimentDataLoader(self.config)
            processor = DataProcessor(self.config)
        except Exception as e:
            self.logger.error(f"Failed to initialize data loader: {e}")
            return {'status': 'failed', 'error': f'Init failed: {e}'}
        
        # Load trip data using experiment-aligned approach (pre-filtered for experiment time range)
        data_load_start = datetime.now()
        try:
            all_trip_data = experiment_loader.load_trip_data_hikima_style(
                self.config.processing_date,
                self.config.vehicle_type,
                self.config.boroughs,
                hour_start=self.config.start_hour,
                hour_end=self.config.end_hour
            )
            data_load_time = (datetime.now() - data_load_start).total_seconds()
            
            if all_trip_data.empty:
                self.logger.error("No trip data found after filtering")
                return {'status': 'failed', 'error': 'No data after filtering'}
                
            self.logger.info(f"Data loading completed in {data_load_time:.2f}s - {len(all_trip_data)} trips (pre-filtered)")
            
        except Exception as e:
            self.logger.error(f"Failed to load trip data: {e}", exc_info=True)
            return {'status': 'failed', 'error': f'Data load failed: {e}'}
        
        # Store data for scenario creation with memory monitoring
        self.logger.debug("Storing trip data for scenario creation")
        self.all_trip_data = all_trip_data
        self.logger.debug(f"Stored {len(all_trip_data)} trips in memory")
        
        # Log memory usage after data loading
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.logger.debug(f"Memory after data loading: RSS={memory_info.rss//1024//1024}MB, VMS={memory_info.vms//1024//1024}MB")
        except ImportError:
            pass
        
        # Generate time windows
        self.logger.debug("Generating time windows...")
        time_windows = self._generate_time_windows()
        self.logger.info(f"Processing {len(time_windows)} time windows")
        self.logger.debug(f"Time windows: {[str(tw) for tw in time_windows]}")
        
        # Data is already pre-filtered by loader - no need for additional date filtering
        self.logger.debug(f"Using pre-filtered data: {len(all_trip_data)} trips")
        
        # Process each time window with enhanced error handling
        self.logger.debug("Starting time window processing loop")
        all_results = []
        
        for tw_idx, time_window in enumerate(time_windows):
            tw_start = datetime.now()
            self.logger.info(f"Processing time window {tw_idx+1}/{len(time_windows)}: {time_window}")
            self.logger.debug(f"Window {tw_idx+1}: {time_window.start.strftime('%H:%M:%S')} -> {time_window.end.strftime('%H:%M:%S')} ({self.config.time_window_size}s duration)")
            
            try:
                # Filter pre-filtered data for this specific time window
                self.logger.debug(f"Filtering {len(all_trip_data)} pre-filtered trips for time window {tw_idx+1}")
                filter_start = datetime.now()
                tw_data = processor.filter_by_time_window(all_trip_data, time_window)
                filter_time = (datetime.now() - filter_start).total_seconds()
                
                self.logger.debug(f"Time window filtering took {filter_time:.3f}s, result: {len(tw_data)} trips")
                
                if tw_data.empty:
                    self.logger.warning(f"No data for time window {time_window}")
                    self.logger.debug(f"Skipping time window {tw_idx+1} - no trips found in {self.config.time_window_size}s window")
                    continue
                    
            except Exception as filter_error:
                self.logger.error(f"Error filtering time window {tw_idx+1}: {filter_error}", exc_info=True)
                continue
            
            self.logger.info(f"Filtered to {len(tw_data)} trips in {filter_time:.2f}s for time window {tw_idx+1}")
            
            # Process scenarios for this time window with timeout and monitoring
            self.logger.debug(f"Starting scenario processing for time window {tw_idx+1}")
            process_start = datetime.now()
            
            try:
                # Add timeout mechanism for scenario processing (Unix only)
                import signal
                import platform
                
                timeout_active = False
                if platform.system() != 'Windows':  # SIGALRM not available on Windows
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Scenario processing timed out")
                    
                    # Set 5 minute timeout for scenario processing
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(300)  # 5 minutes
                    timeout_active = True
                    self.logger.debug("Set 5-minute timeout for scenario processing")
                else:
                    self.logger.debug("Timeout not available on Windows, proceeding without timeout")
                
                self.logger.debug(f"Calling _process_time_window with {len(tw_data)} trips")
                tw_results = self._process_time_window(tw_data, time_window, tw_idx)
                
                # Cancel timeout if it was set
                if timeout_active:
                    signal.alarm(0)
                
                process_time = (datetime.now() - process_start).total_seconds()
                
                self.logger.debug(f"_process_time_window returned {len(tw_results)} results")
                self.logger.info(f"Processed {len(tw_results)} scenarios in {process_time:.2f}s for time window {tw_idx+1}")
                all_results.extend(tw_results)
                self.logger.debug(f"Total results so far: {len(all_results)}")
                
            except TimeoutError:
                self.logger.error(f"Scenario processing timed out for time window {tw_idx+1} after 5 minutes")
                continue
            except Exception as process_error:
                self.logger.error(f"Error in scenario processing for time window {tw_idx+1}: {process_error}", exc_info=True)
                continue
            finally:
                # Ensure timeout is cancelled if it was set
                if 'timeout_active' in locals() and timeout_active:
                    signal.alarm(0)
            
            # Save enhanced decisions for this time window
            self.logger.debug(f"Creating decisions dataframe for time window {tw_idx+1}")
            save_start = datetime.now()
            decisions_df = self._create_enhanced_decisions_dataframe(tw_results, tw_idx)
            self.logger.debug(f"Created decisions dataframe with {len(decisions_df)} rows")
            
            if not decisions_df.empty:
                # Save locally first
                local_file = self.decisions_dir / f"time_window_{tw_idx:04d}.parquet"
                self.logger.debug(f"Saving decisions to local file: {local_file}")
                decisions_df.to_parquet(local_file, index=False)
                self.logger.debug(f"Local file saved successfully: {local_file}")
                
                # Upload to S3 immediately
                self.logger.debug(f"Uploading decisions to S3 for time window {tw_idx+1}")
                upload_success = self.s3_uploader.upload_decisions_parquet(decisions_df, self.experiment_id, tw_idx)
                self.logger.debug(f"S3 upload {'succeeded' if upload_success else 'failed'} for time window {tw_idx+1}")
            else:
                self.logger.debug(f"No decisions to save for time window {tw_idx+1}")
                
            save_time = (datetime.now() - save_start).total_seconds()
            
            tw_total_time = (datetime.now() - tw_start).total_seconds()
            
            # Enhanced logging with visual separation and statistics
            self.logger.info("")
            self.logger.info("#" * 80)
            self.logger.info(f"TIME WINDOW {tw_idx+1}/{len(time_windows)} COMPLETED")
            self.logger.info("#" * 80)
            
            # Calculate progress percentage
            progress_pct = ((tw_idx + 1) / len(time_windows)) * 100
            self.logger.info(f"Progress: {progress_pct:.1f}% ({tw_idx+1}/{len(time_windows)} windows)")
            
            # Time window statistics
            if tw_results:
                avg_requesters = np.mean([r.get('num_requesters', 0) for r in tw_results])
                avg_taxis = np.mean([r.get('num_taxis', 0) for r in tw_results])
                
                # Calculate revenues for this window
                pl_revenues = []
                sigmoid_revenues = []
                computation_times = []
                
                for result in tw_results:
                    if 'PL' in result:
                        pl_revenues.append(result['PL'].get('avg_revenue', 0))
                    if 'Sigmoid' in result:
                        sigmoid_revenues.append(result['Sigmoid'].get('avg_revenue', 0))
                    if 'computation_time' in result:
                        computation_times.append(result['computation_time'])
                
                # Log window statistics
                self.logger.info(f"Window Stats:")
                self.logger.info(f"  Scenarios: {len(tw_results)}")
                self.logger.info(f"  Avg Requesters: {avg_requesters:.1f}")
                self.logger.info(f"  Avg Taxis: {avg_taxis:.1f}")
                self.logger.info(f"  Avg PL Revenue: ${np.mean(pl_revenues):.2f}" if pl_revenues else "  No PL Revenue")
                self.logger.info(f"  Avg Sigmoid Revenue: ${np.mean(sigmoid_revenues):.2f}" if sigmoid_revenues else "  No Sigmoid Revenue")
                
                # Simulation timing
                if computation_times:
                    total_sims = len(tw_results) * self.config.num_iter
                    avg_time_per_sim = np.sum(computation_times) / total_sims if total_sims > 0 else 0
                    self.logger.info(f"  Total Simulations: {total_sims}")
                    self.logger.info(f"  Avg Time per Simulation: {avg_time_per_sim*1000:.2f}ms")
                
                self.logger.info(f"  Window Processing Time: {tw_total_time:.2f}s")
                self.logger.info(f"  Data Filtering: {filter_time:.2f}s")
                self.logger.info(f"  Results Saving: {save_time:.2f}s")
            
            self.logger.info("#" * 80)
            self.logger.info("")
            
            # Overall experiment statistics so far
            if len(all_results) > 0:
                # Calculate cumulative statistics
                all_pl_revenues = []
                all_sigmoid_revenues = []
                all_computation_times = []
                total_requesters = 0
                total_taxis = 0
                
                for result in all_results:
                    if 'PL' in result:
                        all_pl_revenues.append(result['PL'].get('avg_revenue', 0))
                    if 'Sigmoid' in result:
                        all_sigmoid_revenues.append(result['Sigmoid'].get('avg_revenue', 0))
                    if 'computation_time' in result:
                        all_computation_times.append(result['computation_time'])
                    
                    total_requesters += result.get('num_requesters', 0)
                    total_taxis += result.get('num_taxis', 0)
                
                # Log cumulative statistics
                self.logger.info(f"CUMULATIVE STATS (through window {tw_idx+1}):")
                self.logger.info(f"  Total Scenarios: {len(all_results)}")
                self.logger.info(f"  Total Requesters: {total_requesters}")
                self.logger.info(f"  Total Taxis: {total_taxis}")
                
                if all_pl_revenues:
                    self.logger.info(f"  Cumulative PL Revenue: ${np.sum(all_pl_revenues):.2f} (avg: ${np.mean(all_pl_revenues):.2f})")
                if all_sigmoid_revenues:
                    self.logger.info(f"  Cumulative Sigmoid Revenue: ${np.sum(all_sigmoid_revenues):.2f} (avg: ${np.mean(all_sigmoid_revenues):.2f})")
                
                if all_computation_times:
                    total_simulations_so_far = len(all_results) * self.config.num_iter
                    cumulative_time = np.sum(all_computation_times)
                    avg_sim_time = cumulative_time / total_simulations_so_far if total_simulations_so_far > 0 else 0
                    
                    self.logger.info(f"  Total Simulations So Far: {total_simulations_so_far}")
                    self.logger.info(f"  Cumulative Computation Time: {cumulative_time:.2f}s")
                    self.logger.info(f"  Average Time per Simulation: {avg_sim_time*1000:.2f}ms")
                
                # Estimated time remaining
                if tw_idx + 1 < len(time_windows) and tw_total_time > 0:
                    remaining_windows = len(time_windows) - (tw_idx + 1)
                    estimated_remaining = remaining_windows * tw_total_time
                    self.logger.info(f"  Estimated Time Remaining: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} min)")
                
                self.logger.info("")
        
        # Create enhanced summary using new aggregator
        self.logger.debug(f"Creating enhanced experiment summary with {len(all_results)} total results")
        from .enhanced_aggregator import EnhancedResultsAggregator
        
        # Create enhanced aggregator with correct start time
        enhanced_aggregator = EnhancedResultsAggregator(self.experiment_id, self.config.to_dict())
        
        # Set the correct experiment start time for wall-clock calculation
        enhanced_aggregator.experiment_start_time = self.experiment_start_time
        
        # Add all results
        for result in all_results:
            enhanced_aggregator.add_result(result)
        
        # Add all decisions data
        for tw_idx in range(len(self._generate_time_windows())):
            decisions_file = self.decisions_dir / f"time_window_{tw_idx:04d}.parquet"
            if decisions_file.exists():
                try:
                    decisions_df = pd.read_parquet(decisions_file)
                    enhanced_aggregator.add_decisions_data(decisions_df)
                except Exception as e:
                    self.logger.warning(f"Could not load decisions file {decisions_file}: {e}")
        
        # Create comprehensive summary
        summary = enhanced_aggregator.create_comprehensive_summary()
        self.logger.debug(f"Enhanced summary created with {summary.get('experiment_metadata', {}).get('total_scenarios', 0)} scenarios")
        
        # Save enhanced summary
        summary_file = self.output_dir / 'experiment_summary.json'
        enhanced_aggregator.save_summary(summary_file)
        self.logger.info(f"Enhanced summary saved to {summary_file}")
        
        # Add config to summary for S3 upload compatibility
        summary['config'] = self.config.to_dict()
        
        # Upload complete results to S3
        self.logger.debug("Starting S3 upload of complete results")
        s3_path = self.s3_uploader.upload_experiment_results(
            self.experiment_id,
            self.output_dir,
            summary['config']
        )
        self.logger.debug(f"S3 upload completed, path: {s3_path}")
        
        if s3_path:
            summary['s3_path'] = s3_path
            self.logger.info(f"Experiment results uploaded to S3: {s3_path}")
        else:
            self.logger.debug("No S3 path returned - upload may have failed")
        
        self.logger.info(f"Experiment completed. Results saved locally to {self.output_dir}")
        self.logger.debug(f"Final summary: {summary}")
        return summary
    
    def _generate_time_windows(self) -> List[TimeWindow]:
        """
        Generate time windows using configurable intervals with configurable duration.
        
        Approach:
        - Creates windows every time_delta minutes (e.g., 30 minutes: 10:00, 10:30, 11:00...)
        - Each window duration is configurable (default 30 seconds)
        - User's --time-delta parameter controls interval between windows
        - User's --time-window-size parameter controls window duration
        """
        # Convert time_delta to minutes if needed
        if self.config.time_unit == 'm':
            interval_minutes = self.config.time_delta
        elif self.config.time_unit == 'h':
            interval_minutes = self.config.time_delta * 60
        else:
            raise ValueError(f"Unknown time unit: {self.config.time_unit}")
        
        self.logger.debug(f"Generating time windows: {self.config.start_hour}:00 to {self.config.end_hour}:00")
        self.logger.debug(f"Window interval: {interval_minutes} minutes, Window duration: {self.config.time_window_size} seconds")
        
        windows = []
        year = self.config.processing_date.year
        month = self.config.processing_date.month
        day = self.config.processing_date.day
        
        # Calculate how many intervals fit in the time range
        total_minutes = (self.config.end_hour - self.config.start_hour) * 60
        num_intervals = total_minutes // interval_minutes
        
        self.logger.debug(f"Total experiment duration: {total_minutes} minutes")
        self.logger.debug(f"Number of {interval_minutes}-minute intervals: {num_intervals}")
        
        # Generate windows with configurable intervals
        for tt_tmp in range(num_intervals):
            tt = tt_tmp * interval_minutes  # Minutes from start
            h, m = divmod(tt, 60)
            hour = self.config.start_hour + h
            minute = m
            second = 0
            
            # Create window start time
            set_time = datetime(year, month, day, hour, minute, second)
            
            # Create window end time using configurable time_window_size parameter
            window_end = set_time + timedelta(seconds=self.config.time_window_size)
            
            # Ensure we don't exceed the experiment end time
            experiment_end = datetime(year, month, day, self.config.end_hour, 0, 0)
            if set_time >= experiment_end:
                break
                
            if window_end > experiment_end:
                window_end = experiment_end
            
            time_window = TimeWindow(start=set_time, end=window_end)
            windows.append(time_window)
            
            self.logger.debug(f"Window {tt_tmp+1}: {set_time.strftime('%H:%M:%S')} -> {window_end.strftime('%H:%M:%S')} ({self.config.time_window_size}s)")
        
        self.logger.info(f"Generated {len(windows)} time windows with {interval_minutes}-minute intervals and {self.config.time_window_size}s duration each")
        return windows
    
    def _process_time_window(
        self,
        tw_data: pd.DataFrame,
        time_window: TimeWindow,
        tw_idx: int
    ) -> List[Dict]:
        """Process all scenarios for a single time window."""
        self.logger.debug(f"_process_time_window called with {len(tw_data)} trips")
        results = []
        
        # Process each borough
        self.logger.debug(f"Processing {len(self.config.boroughs)} boroughs: {[b.value for b in self.config.boroughs]}")
        for borough in self.config.boroughs:
            borough_start = datetime.now()
            self.logger.debug(f"Processing borough: {borough.value}")
            borough_data = tw_data[tw_data['borough'] == borough.value]
            self.logger.debug(f"Borough {borough.value} has {len(borough_data)} trips")
            
            if borough_data.empty:
                self.logger.debug(f"No data for borough {borough.value} in time window {tw_idx+1}")
                continue
            
            # Create scenario data
            self.logger.debug(f"Creating scenario data for borough {borough.value}")
            scenario_start = datetime.now()
            scenario_data = self._create_scenario_from_data(borough, time_window, borough_data)
            scenario_time = (datetime.now() - scenario_start).total_seconds()
            self.logger.debug(f"Scenario creation took {scenario_time:.3f}s")
            
            scenario_data['time_window'] = str(time_window)
            scenario_data['time_window_idx'] = tw_idx
            scenario_data['borough'] = borough.value
            scenario_data['num_workers'] = self.config.num_workers  # Pass num_workers for simulation parallelization
            
            n_req = scenario_data['num_requesters']
            n_tax = scenario_data['num_taxis']
            self.logger.info(f"Borough {borough.value}: {len(borough_data)} trips → {n_req} requesters, {n_tax} taxis (scenario prep: {scenario_time:.2f}s)")
            self.logger.debug(f"Scenario data keys: {list(scenario_data.keys())}")
            
            # Run each method
            self.logger.debug(f"Running {len(self.config.methods)} methods: {[m.value for m in self.config.methods]}")
            for method in self.config.methods:
                method_start = datetime.now()
                self.logger.info(f"Starting {method.value} for {borough.value} (N={n_req}, M={n_tax})")
                self.logger.debug(f"Method processing start for {method.value}")
                
                # Copy scenario data and add method
                self.logger.debug(f"Copying scenario data for method {method.value}")
                method_scenario = scenario_data.copy()
                method_scenario['method'] = method.value
                method_scenario['scenario_id'] = f"tw{tw_idx}_{borough.value}_{method.value}"
                self.logger.debug(f"Created method scenario with ID: {method_scenario['scenario_id']}")
                
                # Determine parallelization strategy
                use_simulation_parallel = (self.config.num_iter > self.config.num_workers and 
                                         self.config.num_workers > 1)
                
                if use_simulation_parallel:
                    # Use simulation-level parallelization - run scenarios directly
                    self.logger.debug(f"Running {method.value} with simulation parallelization ({self.config.num_iter} sims, {self.config.num_workers} workers)")
                    result = run_single_scenario(
                        (method_scenario, self.config.to_dict(), self.config.num_iter)
                    )
                    method_time = (datetime.now() - method_start).total_seconds()
                    self.logger.info(f"Completed {method.value} for {borough.value} in {method_time:.2f}s")
                    self.logger.debug(f"Method result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                    results.append(result)
                elif self.config.num_workers > 1:
                    # Use scenario-level parallelization for low iteration counts
                    self.logger.debug(f"Queuing {method.value} for scenario-level parallel processing (workers: {self.config.num_workers})")
                    results.append(method_scenario)
                else:
                    # Run directly (single-threaded)
                    self.logger.debug(f"Running {method.value} directly (single-threaded)")
                    result = run_single_scenario(
                        (method_scenario, self.config.to_dict(), self.config.num_iter)
                    )
                    method_time = (datetime.now() - method_start).total_seconds()
                    self.logger.info(f"Completed {method.value} for {borough.value} in {method_time:.2f}s")
                    self.logger.debug(f"Method result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                    results.append(result)
            
            borough_time = (datetime.now() - borough_start).total_seconds()
            self.logger.info(f"Borough {borough.value} completed in {borough_time:.2f}s")
        
        # If using scenario-level parallel processing, run all scenarios for this time window
        # Only use scenario-level parallelization if simulation-level isn't already being used
        use_simulation_parallel = (self.config.num_iter > self.config.num_workers and 
                                 self.config.num_workers > 1)
        use_scenario_parallel = (self.config.num_workers > 1 and 
                               not use_simulation_parallel and 
                               len(results) > 0 and 
                               isinstance(results[0], dict) and 'method' in results[0])
        
        if use_scenario_parallel:
            self.logger.debug(f"Starting scenario-level parallel processing with {len(results)} scenarios")
            results = self._run_parallel(results)
            self.logger.debug(f"Scenario-level parallel processing completed with {len(results)} results")
        else:
            if use_simulation_parallel:
                self.logger.debug(f"Using simulation-level parallelization (scenarios already processed)")
            else:
                self.logger.debug(f"Skipping parallel processing: workers={self.config.num_workers}, scenarios={len(results)}, simulation_parallel={use_simulation_parallel}")
        
        self.logger.debug(f"_process_time_window returning {len(results)} results")
        return results
    
    def _run_parallel(self, scenarios: List[Dict]) -> List[Dict]:
        """Run scenarios in parallel with proper exception handling."""
        config_dict = self.config.to_dict()
        num_iter = self.config.num_iter
        
        # Prepare arguments for parallel processing
        args_list = [(s, config_dict, num_iter) for s in scenarios]
        
        results = []
        try:
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                self.logger.info(f"Starting {len(args_list)} scenarios across {self.config.num_workers} workers")
                
                # Use submit/as_completed for better error handling
                futures = {executor.submit(run_single_scenario, args): i for i, args in enumerate(args_list)}
                
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per scenario
                        results.append(result)
                        self.logger.debug(f"Completed scenario {futures[future]+1}/{len(args_list)}")
                    except Exception as e:
                        self.logger.error(f"Scenario {futures[future]+1} failed: {e}")
                        # Continue processing other scenarios
                        
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            self.logger.info("Falling back to sequential processing")
            for args in args_list:
                try:
                    result = run_single_scenario(args)
                    results.append(result)
                except Exception as scenario_error:
                    self.logger.error(f"Sequential scenario failed: {scenario_error}")
        
        self.logger.info(f"Completed {len(results)}/{len(args_list)} scenarios successfully")
        return results
    
    def _create_scenario_from_data(
        self,
        borough: Borough,
        time_window: TimeWindow,
        data: pd.DataFrame,
        max_trips: int = 1000
    ) -> Dict[str, Any]:
        """
        Create scenario from real data using established preprocessing pipeline.
        
        This replicates the data processing to ensure alignment with reference results.
        """
        self.logger.debug(f"_create_scenario_from_data called with {len(data)} trips for {borough.value}")
        
        try:
            import pyproj
            grs80 = pyproj.Geod(ellps='GRS80')
            self.logger.debug("Initialized pyproj.Geod")
            
            # Load area information (like Hikima's line 54)
            # For now, we'll use synthetic coordinates but the structure matches
            self.logger.debug("Loading area information")
            location_data = self._load_area_information()
            self.logger.debug(f"Loaded {len(location_data)} location coordinates")
            
            # Use the passed-in data (already filtered by time window and borough)
            self.logger.debug(f"Input data shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
            self.logger.debug(f"Input data columns: {list(data.columns) if hasattr(data, 'columns') else 'unknown'}")
            
            if data is None or len(data) == 0:
                self.logger.debug("No data provided, returning empty scenario")
                return self._empty_scenario()
        except Exception as e:
            self.logger.error(f"Error in scenario creation initialization: {e}", exc_info=True)
            return self._empty_scenario()
        
        # Apply standard filtering using config parameters
        self.logger.debug(f"Applying data filtering to {len(data)} trips")
        data = data[
            (data['trip_distance'] > self.config.min_trip_distance) & 
            (data['total_amount'] > self.config.min_total_amount) & 
            (data['PULocationID'] < self.config.max_location_id) & 
            (data['DOLocationID'] < self.config.max_location_id)
        ]
        self.logger.debug(f"After filtering: {len(data)} trips remain")
        
        if len(data) == 0:
            self.logger.debug("No trips left after filtering, returning empty scenario")
            return self._empty_scenario()
        
        # Hikima's exact approach: separate pickup vs dropoff data
        
        # Get requesters: trips STARTING in this time window (pickup_datetime)
        requesters_data = data.copy()  # These are new trip requests
        
        # Get taxis: simulate vehicles finishing trips and becoming available
        # In the original setup, this would be trips ending in the time window
        # Since we don't have proper dropoff timing in our simplified data,
        # we simulate this by using a different subset with some randomization
        
        # Simulate realistic taxi availability (usually different from requests)
        np.random.seed(42 + len(data))  # Consistent but different from requesters
        
        if len(data) > 1:
            # Simulate taxi supply: typically 70-130% of request demand
            supply_ratio = np.random.uniform(0.7, 1.3)
            target_taxi_count = max(1, int(len(data) * supply_ratio))
            
            # Sample different indices for taxis (simulating dropoffs in the area)
            if target_taxi_count >= len(data):
                # More taxis than trips - duplicate some with location noise
                taxis_data = data.copy()
                additional_needed = target_taxi_count - len(data)
                if additional_needed > 0:
                    extra_taxis = data.sample(n=min(additional_needed, len(data)), replace=True, random_state=42)
                    taxis_data = pd.concat([taxis_data, extra_taxis], ignore_index=True)
            else:
                # Fewer taxis than requests - sample subset
                taxis_data = data.sample(n=target_taxi_count, replace=False, random_state=42)
        else:
            taxis_data = data.copy()
        
        self.logger.debug(f"Requesters from pickup data: {len(requesters_data)}, Taxis from simulated dropoffs: {len(taxis_data)}")
        self.logger.debug(f"Request/Taxi ratio: {len(requesters_data)/max(1,len(taxis_data)):.2f} (realistic variation)")
        
        # Limit to max_trips if needed
        if len(requesters_data) > max_trips:
            requesters_data = requesters_data.sample(n=max_trips, random_state=42)
        
        n_requesters = len(requesters_data)
        n_taxis = len(taxis_data)
        
        if n_requesters == 0 or n_taxis == 0:
            return self._empty_scenario()
        
        # Sort by distance in ascending order for processing
        requesters_data = requesters_data.sort_values('trip_distance', ascending=True)
        
        # Convert distances to km
        trip_distances = requesters_data['trip_distance'].values * 1.60934  # miles to km
        trip_amounts = requesters_data['total_amount'].values
        
        # Generate locations with noise for realistic positioning
        requester_locations = self._generate_locations_with_noise(
            requesters_data['PULocationID'].values, location_data
        )
        taxi_locations = self._generate_locations_with_noise(
            taxis_data['DOLocationID'].values, location_data  # Taxis at dropoff locations
        )
        
        # Calculate distance matrix for matching optimization
        distance_matrix = self._calculate_distance_matrix(
            requester_locations, taxi_locations, grs80
        )
        
        # Calculate edge weights for optimization
        edge_weights = self._calculate_edge_weights(
            distance_matrix, trip_distances, n_requesters, n_taxis
        )
        
        return {
            'num_requesters': n_requesters,
            'num_taxis': n_taxis,
            'trip_amounts': trip_amounts,
            'trip_distances': trip_distances,
            'edge_weights': edge_weights,
            'pickup_locations': requesters_data['PULocationID'].values,
            'dropoff_locations': requesters_data['DOLocationID'].values,
            'location_ids': requesters_data['PULocationID'].values,  # For MAPS
            'distance_matrix': distance_matrix,  # For reference
        }
    
    def _load_area_information(self) -> Dict[int, Tuple[float, float]]:
        """Load area coordinate information (simplified version of Hikima's area_information.csv)."""
        # This is a simplified version. In full implementation, would load actual CSV
        # For now, create synthetic but consistent coordinates
        import numpy as np
        np.random.seed(42)  # Ensure consistency
        
        location_coords = {}
        for loc_id in range(1, self.config.max_location_id + 1):  # LocationIDs 1-max_location_id
            # Generate consistent coordinates for NYC area
            lat = 40.7 + np.random.normal(0, 0.1)  # Roughly NYC latitude
            lon = -74.0 + np.random.normal(0, 0.1)  # Roughly NYC longitude
            location_coords[loc_id] = (lon, lat)  # (longitude, latitude)
        
        return location_coords
    
    def _generate_locations_with_noise(
        self, 
        location_ids: np.ndarray, 
        location_data: Dict[int, Tuple[float, float]]
    ) -> np.ndarray:
        """Generate locations with Gaussian noise for realistic positioning."""
        import numpy as np
        
        locations = np.zeros((len(location_ids), 2))
        for i, loc_id in enumerate(location_ids):
            if loc_id in location_data:
                base_lon, base_lat = location_data[loc_id]
            else:
                # Default to Manhattan center if location not found
                base_lon, base_lat = -73.9857, 40.7484
            
            # Add realistic positioning noise
            locations[i, 0] = base_lon + np.random.normal(0, 0.00306)  # longitude
            locations[i, 1] = base_lat + np.random.normal(0, 0.000896)  # latitude
        
        return locations
    
    def _calculate_distance_matrix(
        self,
        requester_locations: np.ndarray,
        taxi_locations: np.ndarray,
        grs80
    ) -> np.ndarray:
        """Calculate distance matrix between requesters and taxis."""
        n_requesters = len(requester_locations)
        n_taxis = len(taxi_locations)
        distance_matrix = np.zeros((n_requesters, n_taxis))
        
        for i in range(n_requesters):
            for j in range(n_taxis):
                azimuth, bkw_azimuth, distance = grs80.inv(
                    requester_locations[i, 0], requester_locations[i, 1],
                    taxi_locations[j, 0], taxi_locations[j, 1]
                )
                distance_matrix[i, j] = distance * 0.001  # Convert to km
        
        return distance_matrix
    
    def _calculate_edge_weights(
        self,
        distance_matrix: np.ndarray,
        trip_distances: np.ndarray,
        n_requesters: int,
        n_taxis: int
    ) -> np.ndarray:
        """Calculate edge weights for optimization matching."""
        edge_weights = np.zeros((n_requesters, n_taxis))
        
        for i in range(n_requesters):
            for j in range(n_taxis):
                # Cost formula: W[i,j] = -(distance_ij[i,j] + trip_distance[i]) / s_taxi * alpha
                edge_weights[i, j] = -(distance_matrix[i, j] + trip_distances[i]) / self.config.s_taxi * self.config.alpha
        
        return edge_weights
    
    def _empty_scenario(self) -> Dict[str, Any]:
        """Return an empty scenario dictionary."""
        return {
            'num_requesters': 0,
            'num_taxis': 0,
            'trip_amounts': [],
            'trip_distances': [],
            'edge_weights': [],
            'pickup_locations': [],
            'dropoff_locations': [],
            'location_ids': [],
            'distance_matrix': [],
        }
    
    def _create_enhanced_decisions_dataframe(self, results: List[Dict], tw_idx: int) -> pd.DataFrame:
        """Create enhanced decisions DataFrame with sampled decisions, profit, and compute_time."""
        if not results:
            return pd.DataFrame()
        
        # Collect all decisions with enhanced information
        decisions = []
        
        for result in results:
            if 'prices' not in result:
                continue
                
            prices = result['prices']
            n_requesters = len(prices)
            method = result.get('method', '')
            borough = result.get('borough', '')
            computation_time = result.get('computation_time', 0)
            
            # Extract matching results for both acceptance functions
            for accept_func in ['PL', 'Sigmoid']:
                if accept_func not in result:
                    continue
                
                func_results = result[accept_func]
                acceptance_probs = func_results.get('acceptance_probs', [])
                matching_results = func_results.get('matching_results', [])
                
                # Simulate actual decisions (sampling) for each requester
                np.random.seed(42 + tw_idx)  # Deterministic but varied by time window
                
                for i in range(n_requesters):
                    price = prices[i] if i < len(prices) else 0
                    accept_prob = acceptance_probs[i] if i < len(acceptance_probs) else 0
                    
                    # Sample decision based on acceptance probability
                    sampled_decision = 1 if np.random.random() < accept_prob else 0
                    
                    # Calculate profit - revenue if matched, 0 if not
                    # In a real scenario, this would depend on actual matching results
                    profit = price * sampled_decision if sampled_decision else 0
                    
                    # Check if this requester was actually matched in the optimization
                    was_matched = 0
                    if matching_results and i < len(matching_results):
                        was_matched = 1 if matching_results[i] >= 0 else 0
                    
                    decision = {
                        'time_window_idx': tw_idx,
                        'time_window': result.get('time_window', ''),
                        'borough': borough,
                        'method': method,
                        'acceptance_function': accept_func,
                        'requester_id': i,
                        'price': price,
                        'acceptance_prob': accept_prob,
                        'sampled_decision': sampled_decision,  # New: actual sampled acceptance
                        'was_matched': was_matched,           # New: whether actually matched in optimization
                        'profit': profit,                     # New: profit from this decision
                        'compute_time': computation_time,     # New: computation time for this method
                    }
                    decisions.append(decision)
        
        # Create DataFrame
        if decisions:
            df = pd.DataFrame(decisions)
            return df
        else:
            return pd.DataFrame()
    

    
 