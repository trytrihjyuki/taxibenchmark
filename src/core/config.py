"""Configuration management for taxi benchmark experiments."""

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json

from .types import (
    VehicleType, Borough, PricingMethod, TimeWindow
)


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment run."""
    
    # Core parameters
    processing_date: date
    vehicle_type: VehicleType
    boroughs: List[Borough]
    methods: List[PricingMethod]
    
    # Time window configuration
    start_hour: int = 0
    end_hour: int = 23
    time_delta: int = 5  # minutes - interval between window starts (Hikima uses 5)
    time_unit: str = 'm'  # 'm' for minutes, 'h' for hours, 's' for seconds
    time_window_size: int = 30  # seconds - actual window duration (Hikima default: 30s)
    
    # Monte Carlo parameters
    num_iter: int = 100  # Number of Monte Carlo iterations per time window
    
    # Parallel processing
    num_workers: int = 4
    
    # S3 configuration
    s3_base: str = field(default_factory=lambda: os.getenv("S3_BASE", "magisterka"))
    s3_results_bucket: str = "taxi-benchmark"
    
    # Algorithm parameters (from original paper)
    alpha: float = 18.0  # Cost parameter
    s_taxi: float = 25.0  # Taxi speed (km/h)
    
    # Acceptance function parameters
    # For PL (ReLU)
    pl_alpha: float = 1.5  # Upper bound multiplier
    
    # For Sigmoid
    sigmoid_beta: float = 1.3
    sigmoid_gamma: float = field(default_factory=lambda: (0.3 * (3**0.5) / 3.141592653589793))  # Exact match to Hikima's line 49
    
    # LinUCB parameters
    ucb_alpha: float = 0.5
    base_price: float = 5.875
    price_multipliers: List[float] = field(
        default_factory=lambda: [0.6, 0.8, 1.0, 1.2, 1.4]
    )
    
    # MAPS parameters
    maps_s0_rate: float = 1.5
    maps_price_delta: float = 0.05
    maps_matching_radius: float = 2.0  # km
    
    # LP parameters
    lp_price_grid_size: int = 50  # Number of price points in discretization (was 80, reduced for performance)
    lp_solver: str = 'highs'  # LP solver: 'cbc' (default), 'highs' (faster if available), 'gurobi', 'cplex'
    lp_price_min_mult: float = 0.5  # Minimum price = valuation * this multiplier
    lp_price_max_mult: float = 2.0  # Maximum price = valuation * this multiplier
    
    # Data filtering
    min_trip_distance: float = 0.001  # km
    min_total_amount: float = 0.001  # dollars
    max_location_id: int = 264
    
    # Output configuration
    save_intermediate: bool = True
    batch_size: int = 100  # Number of scenarios per batch
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.start_hour > self.end_hour:
            raise ValueError(f"start_hour ({self.start_hour}) must be <= end_hour ({self.end_hour})")
        
        if self.time_delta <= 0:
            raise ValueError(f"time_delta must be positive, got {self.time_delta}")
        
        if self.num_iter <= 0:
            raise ValueError(f"num_iter must be positive, got {self.num_iter}")
        
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")
    
    def get_time_windows(self) -> List[TimeWindow]:
        """Generate all time windows for the experiment."""
        windows = []
        
        if self.time_unit == 'm':
            delta_minutes = self.time_delta
        elif self.time_unit == 'h':
            delta_minutes = self.time_delta * 60
        else:
            raise ValueError(f"Unknown time unit: {self.time_unit}")
        
        current_minutes = self.start_hour * 60
        end_minutes = self.end_hour * 60
        
        while current_minutes < end_minutes:
            start_h = current_minutes // 60
            start_m = current_minutes % 60
            
            end_time_minutes = min(current_minutes + delta_minutes, end_minutes)
            end_h = end_time_minutes // 60
            end_m = end_time_minutes % 60
            
            windows.append(TimeWindow(start_h, start_m, end_h, end_m))
            current_minutes += delta_minutes
        
        return windows
    
    def get_total_scenarios(self) -> int:
        """Calculate total number of scenarios."""
        num_windows = len(self.get_time_windows())
        return num_windows * len(self.boroughs) * len(self.methods) * self.num_iter
    
    def get_s3_data_path(self) -> str:
        """Get S3 path for input data."""
        year = self.processing_date.year
        month = self.processing_date.month
        
        return (f"s3://{self.s3_base}/datasets/{self.vehicle_type.value}/"
                f"year={year}/month={month:02d}/"
                f"{self.vehicle_type.value}_tripdata_{year}-{month:02d}.parquet")
    
    def get_s3_results_path(self) -> str:
        """Get S3 path for results."""
        exp_id = self.get_experiment_id()
        return f"s3://{self.s3_results_bucket}/experiments/{exp_id}/"
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = self.processing_date.strftime("%Y%m%d")
        borough_str = "_".join([b.value[:3] for b in self.boroughs])
        method_str = "_".join([m.value for m in self.methods])
        
        return f"{date_str}_{self.vehicle_type.value}_{borough_str}_{method_str}_{timestamp}"
    
    def get_linucb_model_path(self, borough: Borough) -> str:
        """Get S3 path for LinUCB pre-trained model."""
        year = self.processing_date.year
        month = self.processing_date.month
        
        return (f"s3://{self.s3_base}/models/linucb/"
                f"{self.vehicle_type.value}_{borough.value}_{year}{month:02d}/"
                "trained_model.pkl")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'processing_date': self.processing_date.isoformat(),
            'vehicle_type': self.vehicle_type.value,
            'boroughs': [b.value for b in self.boroughs],
            'methods': [m.value for m in self.methods],
            'start_hour': self.start_hour,
            'end_hour': self.end_hour,
            'time_delta': self.time_delta,
            'time_unit': self.time_unit,
            'time_window_size': self.time_window_size,
            'num_iter': self.num_iter,
            'num_workers': self.num_workers,
            's3_base': self.s3_base,
            's3_results_bucket': self.s3_results_bucket,
            'alpha': self.alpha,
            's_taxi': self.s_taxi,
            'pl_alpha': self.pl_alpha,
            'sigmoid_beta': self.sigmoid_beta,
            'sigmoid_gamma': self.sigmoid_gamma,
            'ucb_alpha': self.ucb_alpha,
            'base_price': self.base_price,
            'price_multipliers': self.price_multipliers,
            'maps_s0_rate': self.maps_s0_rate,
            'maps_price_delta': self.maps_price_delta,
            'maps_matching_radius': self.maps_matching_radius,
            'experiment_id': self.get_experiment_id()
        }
    
    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        # Convert string dates to date objects
        data['processing_date'] = date.fromisoformat(data['processing_date'])
        
        # Convert string enums
        data['vehicle_type'] = VehicleType(data['vehicle_type'])
        data['boroughs'] = [Borough(b) for b in data['boroughs']]
        data['methods'] = [PricingMethod(m) for m in data['methods']]
        # Note: AcceptanceFunction handling can be added if needed in the future
        
        # Remove fields that aren't part of the dataclass
        data.pop('experiment_id', None)
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario within an experiment."""
    experiment_config: ExperimentConfig
    time_window: TimeWindow
    borough: Borough
    method: PricingMethod
    iteration: int
    
    @property
    def scenario_id(self) -> str:
        """Generate unique scenario ID."""
        exp_id = self.experiment_config.get_experiment_id()
        tw_str = f"{self.time_window.start_hour:02d}{self.time_window.start_minute:02d}"
        return f"{exp_id}_{self.borough.value[:3]}_{self.method.value}_{tw_str}_{self.iteration:04d}"
    
    def get_data_filters(self) -> Dict[str, Any]:
        """Get filters for loading data for this scenario."""
        start_dt, end_dt = self.time_window.to_datetime_range(
            self.experiment_config.processing_date
        )
        
        return {
            'start_datetime': start_dt,
            'end_datetime': end_dt,
            'borough': self.borough.value,
            'min_trip_distance': self.experiment_config.min_trip_distance,
            'min_total_amount': self.experiment_config.min_total_amount,
            'max_location_id': self.experiment_config.max_location_id
        } 