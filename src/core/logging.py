"""Logging configuration for taxi benchmark framework."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorlog


def setup_logger(
    name: str = "taxi_benchmark",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    colorize: bool = True
) -> logging.Logger:
    """
    Setup a logger with colored console output and optional file logging.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        colorize: Whether to use colored output for console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with colors
    if colorize:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(f"taxi_benchmark.{name}")


class ExperimentLogger:
    """Context manager for experiment logging with progress tracking."""
    
    def __init__(self, experiment_id: str, total_scenarios: int):
        self.experiment_id = experiment_id
        self.total_scenarios = total_scenarios
        self.completed_scenarios = 0
        self.start_time = None
        self.logger = get_logger(f"experiment.{experiment_id}")
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting experiment {self.experiment_id}")
        self.logger.info(f"Total scenarios to process: {self.total_scenarios}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(
                f"Experiment {self.experiment_id} completed successfully in {duration:.2f}s"
            )
        else:
            self.logger.error(
                f"Experiment {self.experiment_id} failed after {duration:.2f}s: {exc_val}"
            )
        return False
    
    def update(self, scenarios_completed: int = 1):
        """Update progress."""
        self.completed_scenarios += scenarios_completed
        progress = (self.completed_scenarios / self.total_scenarios) * 100
        
        if self.completed_scenarios % 10 == 0 or self.completed_scenarios == self.total_scenarios:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.completed_scenarios / elapsed if elapsed > 0 else 0
            eta = (self.total_scenarios - self.completed_scenarios) / rate if rate > 0 else 0
            
            self.logger.info(
                f"Progress: {self.completed_scenarios}/{self.total_scenarios} "
                f"({progress:.1f}%) - Rate: {rate:.2f} scenarios/s - ETA: {eta:.0f}s"
            )
    
    def log_scenario_result(self, scenario_id: str, success: bool, metrics: Optional[dict] = None):
        """Log result of a single scenario."""
        if success:
            msg = f"Scenario {scenario_id} completed"
            if metrics:
                msg += f" - Profit: ${metrics.get('profit', 0):.2f}"
                msg += f" - Matches: {metrics.get('num_matched', 0)}"
            self.logger.debug(msg)
        else:
            self.logger.warning(f"Scenario {scenario_id} failed")


class PerformanceLogger:
    """Logger for tracking performance metrics."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"performance.{name}")
        self.timings = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.timings[operation] = datetime.now()
    
    def end_timer(self, operation: str, log_message: bool = True):
        """End timing an operation and optionally log the duration."""
        if operation not in self.timings:
            self.logger.warning(f"Timer for {operation} was not started")
            return None
        
        duration = (datetime.now() - self.timings[operation]).total_seconds()
        
        if log_message:
            self.logger.debug(f"{operation} took {duration:.3f}s")
        
        del self.timings[operation]
        return duration
    
    def log_memory_usage(self):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.logger.debug(f"Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            pass  # psutil not installed 