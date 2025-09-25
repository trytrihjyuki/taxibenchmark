"""Experiment-aligned data loader."""

from datetime import date, datetime
from typing import List, Dict, Any
import pandas as pd

from ..core import get_logger
from ..core.types import VehicleType, Borough
from .s3_loader import S3DataLoader


class ExperimentDataLoader:
    """
    Experiment-aligned data loader that wraps S3DataLoader.
    
    This provides a clean interface for the experiment runner while maintaining
    the same data loading logic that was previously in HikimaAlignedLoader.
    """
    
    def __init__(self, config=None):
        """Initialize loader."""
        self.config = config
        self.logger = get_logger('data.experiment_loader')
        
        # Initialize the underlying S3 loader
        self.s3_loader = S3DataLoader(config)
        
        self.logger.debug("Experiment data loader initialized")
    
    def load_trip_data_hikima_style(
        self,
        processing_date: date,
        vehicle_type: VehicleType,
        boroughs: List[Borough],
        hour_start: int = 10,
        hour_end: int = 20
    ) -> pd.DataFrame:
        """
        Load trip data using experiment-aligned approach.
        
        This method replicates the data loading to ensure alignment with reference implementation.
        
        Args:
            processing_date: Date to process
            vehicle_type: Type of vehicle 
            boroughs: List of boroughs
            hour_start: Start hour for experiment (configurable)
            hour_end: End hour for experiment (configurable)
        
        Returns:
            DataFrame with trip data (pre-filtered)
        """
        self.logger.info(f"Loading trip data for {processing_date}")
        self.logger.info(f"Experiment window: {hour_start}:00 to {hour_end}:00 (buffered for taxi positioning)")
        
        # Calculate time range with buffer for taxi positioning
        year = processing_date.year
        month = processing_date.month
        day = processing_date.day
        
        # Add buffer before/after for taxi positioning
        if hour_start > 0:
            day_start_time = datetime(year, month, day, hour_start-1, 55, 0)  # 5min before
        else:
            day_start_time = datetime(year, month, day, 0, 0, 0)
        
        day_end_time = datetime(year, month, day, hour_end, 5, 0)  # 5min after
        
        self.logger.info(f"Time range with buffer: {day_start_time} to {day_end_time}")
        
        # Load trip data using the robust S3 loader
        try:
            trip_data = self.s3_loader.load_trip_data(
                processing_date, vehicle_type, boroughs
            )
            
            if trip_data.empty:
                self.logger.warning("No trip data loaded from S3")
                return trip_data
            
            self.logger.info(f"S3 data loaded: {len(trip_data)} trips")
            
            # Apply time range filter (with buffer)
            if 'pickup_datetime' in trip_data.columns:
                initial_count = len(trip_data)
                
                # Filter to buffered time range
                time_mask = (
                    (trip_data['pickup_datetime'] >= day_start_time) &
                    (trip_data['pickup_datetime'] < day_end_time)
                )
                trip_data = trip_data.loc[time_mask].copy()
                
                self.logger.info(f"Time filtering: {initial_count} -> {len(trip_data)} trips")
            
            # Additional filters are already applied in S3DataLoader
            self.logger.info(f"Final filtered dataset: {len(trip_data)} trips")
            
            return trip_data
            
        except Exception as e:
            self.logger.error(f"Failed to load trip data: {e}")
            return pd.DataFrame()
    
    def load_area_info(self) -> pd.DataFrame:
        """Load area information."""
        return self.s3_loader.load_area_info()