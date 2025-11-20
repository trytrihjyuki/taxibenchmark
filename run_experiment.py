#!/usr/bin/env python
"""
Simplified experiment runner for taxi pricing benchmark.

Usage:
    python run_experiment.py --processing-date 2019-10-06 [options]
"""

import sys
import argparse
import signal
from datetime import date
from pathlib import Path

# Add both parent and src to path for robust module resolution
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

# Debug path information
print(f"DEBUG: Current working directory: {Path.cwd()}")
print(f"DEBUG: Script directory: {current_dir}")
print(f"DEBUG: Python path: {sys.path[:3]}")  # Show first 3 entries
print(f"DEBUG: src/__init__.py exists: {(current_dir / 'src' / '__init__.py').exists()}")

try:
    from src.core import ExperimentConfig, setup_logger, get_logger
    from src.core.types import VehicleType, Borough, PricingMethod
    from src.experiments import ExperimentRunner
    print("DEBUG: All imports successful")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print(f"DEBUG: Available files in src/: {list((current_dir / 'src').glob('*')) if (current_dir / 'src').exists() else 'src directory not found'}")
    raise


def signal_handler(signum, frame):
    """Handle termination signals and log them."""
    logger = get_logger()
    logger.error(f"Received signal {signum} ({signal.Signals(signum).name}), terminating...")
    logger.error(f"Frame info: {frame}")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run taxi pricing experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--processing-date',
        type=str,
        required=True,
        help='Date to process (YYYY-MM-DD)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--vehicle-type',
        type=str,
        default='green',
        choices=['green', 'yellow'],
        help='Vehicle type'
    )
    
    parser.add_argument(
        '--boroughs',
        nargs='+',
        default=['Manhattan'],
        help='Boroughs to process'
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['MinMaxCostFlow', 'MAPS'],
        help='Pricing methods to use'
    )
    
    parser.add_argument(
        '--num-iter',
        type=int,
        default=100,
        help='Number of Monte Carlo iterations'
    )
    
    parser.add_argument(
        '--start-hour',
        type=int,
        default=10,
        help='Start hour (0-23)'
    )
    
    parser.add_argument(
        '--end-hour',
        type=int,
        default=11,
        help='End hour (0-23)'
    )
    
    parser.add_argument(
        '--time-delta',
        type=int,
        default=5,
        help='Interval between time window starts in minutes (Hikima uses 5)'
    )
    
    parser.add_argument(
        '--time-window-size',
        type=int,
        default=30,
        help='Time window duration in seconds (Hikima default: 30s)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='DEBUG',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    # LP-specific parameters
    parser.add_argument(
        '--lp-price-grid-size',
        type=int,
        default=70,
        help='Number of discrete price levels (linear spacing from 0 to 2×trip_amount, default: 5)'
    )
    
    parser.add_argument(
        '--lp-solver',
        type=str,
        default='cbc',
        choices=['cbc', 'highs', 'gurobi', 'cplex'],
        help='LP solver to use (default: cbc, highs is faster if available)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42, like reference code)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger(level=args.log_level)
    
    # Register signal handlers to catch termination
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGKILL'):
        try:
            signal.signal(signal.SIGKILL, signal_handler)
        except OSError:
            pass  # SIGKILL cannot be caught
    if hasattr(signal, 'SIGQUIT'):
        signal.signal(signal.SIGQUIT, signal_handler)
    
    logger.debug("Signal handlers registered")
    
    logger.info("=" * 80)
    logger.info("TAXI PRICING BENCHMARK")
    logger.info("=" * 80)
    
    try:
        # Parse date
        try:
            processing_date = date.fromisoformat(args.processing_date)
        except ValueError:
            logger.error(f"Invalid date format: {args.processing_date}")
            return 1
        
        # Parse vehicle type
        vehicle_type = VehicleType.from_string(args.vehicle_type)
        
        # Parse boroughs
        boroughs = []
        for b in args.boroughs:
            try:
                boroughs.append(Borough.from_string(b))
            except ValueError:
                logger.error(f"Invalid borough: {b}")
                return 1
        
        # Parse methods
        methods = []
        for m in args.methods:
            try:
                methods.append(PricingMethod(m))
            except ValueError:
                logger.error(f"Invalid method: {m}")
                logger.info(f"Available methods: {[pm.value for pm in PricingMethod]}")
                return 1
        
        # Create configuration
        config = ExperimentConfig(
            processing_date=processing_date,
            vehicle_type=vehicle_type,
            boroughs=boroughs,
            methods=methods,
            start_hour=args.start_hour,
            end_hour=args.end_hour,
            time_delta=args.time_delta,
            time_window_size=args.time_window_size,
            num_iter=args.num_iter,
            num_workers=args.num_workers,
            lp_price_grid_size=args.lp_price_grid_size,
            lp_solver=args.lp_solver,
            random_seed=args.seed
        )
        
        # Log configuration
        logger.info("Configuration:")
        logger.info(f"  Date: {config.processing_date}")
        logger.info(f"  Vehicle: {config.vehicle_type.value}")
        logger.info(f"  Boroughs: {[b.value for b in config.boroughs]}")
        logger.info(f"  Methods: {[m.value for m in config.methods]}")
        logger.info(f"  Time: {config.start_hour:02d}:00 - {config.end_hour:02d}:00")
        logger.info(f"  Window Interval: {config.time_delta} minutes (Hikima: 5min)")
        logger.info(f"  Window Duration: {config.time_window_size} seconds (Hikima: 30s)")
        logger.info(f"  Iterations: {config.num_iter}")
        logger.info(f"  Workers: {config.num_workers}")
        logger.info("")
        
        # Create and run experiment
        logger.info("Starting experiment...")
        runner = ExperimentRunner(config)
        summary = runner.run()
        
        # Log results
        if summary.get('status') == 'completed':
            logger.info("=" * 80)
            logger.info("EXPERIMENT COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Duration: {summary.get('duration_seconds', 0):.1f} seconds")
            
            if 'metrics' in summary:
                metrics = summary['metrics']
                logger.info(f"Scenarios: {metrics.get('total_scenarios', 0)}")
                logger.info(f"PL Revenue: ${metrics.get('pl_avg_revenue', 0):.2f}")
                logger.info(f"Sigmoid Revenue: ${metrics.get('sigmoid_avg_revenue', 0):.2f}")
            
            logger.info(f"Results saved to: experiments/{summary.get('experiment_id', 'unknown')}")
            return 0
        else:
            logger.error("Experiment failed")
            return 1
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())