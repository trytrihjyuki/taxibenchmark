"""Type definitions for the taxi benchmark framework."""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime, date, time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd


class VehicleType(Enum):
    """Types of vehicles in the taxi system."""
    GREEN = "green"
    YELLOW = "yellow"
    FHV = "fhv"
    FHVHV = "fhvhv"
    
    @classmethod
    def from_string(cls, value: str) -> 'VehicleType':
        """Create VehicleType from string value."""
        value_lower = value.lower()
        for vehicle_type in cls:
            if vehicle_type.value == value_lower:
                return vehicle_type
        raise ValueError(f"Invalid vehicle type: {value}")
    
    def get_pickup_column(self) -> str:
        """Get the pickup datetime column name for this vehicle type."""
        if self == VehicleType.YELLOW:
            return 'tpep_pickup_datetime'
        elif self == VehicleType.GREEN:
            return 'lpep_pickup_datetime'
        else:
            return 'pickup_datetime'
    
    def get_dropoff_column(self) -> str:
        """Get the dropoff datetime column name for this vehicle type."""
        if self == VehicleType.YELLOW:
            return 'tpep_dropoff_datetime'
        elif self == VehicleType.GREEN:
            return 'lpep_dropoff_datetime'
        else:
            return 'dropOff_datetime'


class Borough(Enum):
    """NYC Boroughs."""
    MANHATTAN = "Manhattan"
    BROOKLYN = "Brooklyn"
    QUEENS = "Queens"
    BRONX = "Bronx"
    STATEN_ISLAND = "Staten Island"
    EWR = "EWR"  # Newark Airport
    
    @classmethod
    def from_string(cls, value: str) -> 'Borough':
        """Create Borough from string value."""
        value_upper = value.upper().replace(' ', '_')
        for borough in cls:
            if borough.name == value_upper:
                return borough
        raise ValueError(f"Unknown borough: {value}")


class PricingMethod(Enum):
    """Available pricing methods."""
    LP = "LP"  # Linear Programming (NEW)
    MIN_MAX_COST_FLOW = "MinMaxCostFlow"
    MAPS = "MAPS"
    LIN_UCB = "LinUCB"
    
    def is_learning_based(self) -> bool:
        """Check if method requires historical data for learning."""
        return self == PricingMethod.LIN_UCB


class AcceptanceFunction(Enum):
    """Acceptance probability functions."""
    PL = "PL"  # Piecewise Linear (ReLU)
    SIGMOID = "Sigmoid"
    
    def compute_probability(self, price: float, valuation: float, 
                           beta: float = 0.5, gamma: float = 0.1) -> float:
        """
        Compute acceptance probability given price and valuation.
        
        Args:
            price: Offered price
            valuation: Customer's valuation
            beta: Beta parameter (for sigmoid)
            gamma: Gamma parameter (for sigmoid)
            
        Returns:
            Acceptance probability
        """
        if self == AcceptanceFunction.PL:
            # Piecewise linear (ReLU-style)
            if price <= valuation:
                return 1.0
            elif price <= 1.5 * valuation:
                return 2.0 - price / valuation
            else:
                return 0.0
        else:  # Sigmoid
            # Sigmoid acceptance function
            import numpy as np
            exponent = (valuation - price) / (gamma * abs(valuation) if valuation != 0 else 1.0)
            return 1.0 / (1.0 + np.exp(-exponent))


@dataclass
class TimeWindow:
    """Represents a time window for experiments."""
    start: datetime
    end: datetime
    
    def __post_init__(self):
        """Validate time window."""
        if self.start >= self.end:
            raise ValueError("Start time must be before end time")
    
    def duration_minutes(self) -> int:
        """Get duration in minutes."""
        return int((self.end - self.start).total_seconds() / 60)


@dataclass
class ScenarioResult:
    """Results from a single scenario execution."""
    scenario_id: str
    date: date
    time_window: TimeWindow
    borough: Borough
    vehicle_type: VehicleType
    method: PricingMethod
    
    # Metrics
    num_requests: int
    num_taxis: int
    num_matched: int
    total_revenue: float
    total_cost: float
    profit: float
    
    # Performance
    computation_time: float
    acceptance_rate: float
    matching_rate: float
    
    # Additional data
    prices: Optional[np.ndarray] = None
    matches: Optional[List[Tuple[int, int]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'scenario_id': self.scenario_id,
            'date': self.date.isoformat(),
            'time_window_start': f"{self.time_window.start_hour:02d}:{self.time_window.start_minute:02d}",
            'time_window_end': f"{self.time_window.end_hour:02d}:{self.time_window.end_minute:02d}",
            'borough': self.borough.value,
            'vehicle_type': self.vehicle_type.value,
            'method': self.method.value,
            'num_requests': self.num_requests,
            'num_taxis': self.num_taxis,
            'num_matched': self.num_matched,
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'profit': self.profit,
            'computation_time': self.computation_time,
            'acceptance_rate': self.acceptance_rate,
            'matching_rate': self.matching_rate,
            'metadata': self.metadata or {}
        } 