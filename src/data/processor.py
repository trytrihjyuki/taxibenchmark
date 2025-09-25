"""Simplified data processor."""

import pandas as pd
from datetime import datetime
from typing import Optional

from ..core import get_logger
from ..core.types import TimeWindow


class DataProcessor:
    """Simple data processor."""
    
    def __init__(self, config=None):
        """Initialize processor."""
        self.logger = get_logger('data.processor')
        self.config = config
    
    def filter_by_time_window(
        self,
        df: pd.DataFrame,
        time_window: TimeWindow
    ) -> pd.DataFrame:
        """
        Filter dataframe by time window.
        
        Args:
            df: DataFrame with pickup_datetime column
            time_window: Time window to filter by
        
        Returns:
            Filtered DataFrame
        """
        self.logger.debug(f"filter_by_time_window called with {len(df)} trips")
        self.logger.debug(f"Time window: {time_window.start} to {time_window.end}")
        
        if df.empty:
            self.logger.debug("DataFrame is empty, returning empty DataFrame")
            return df
        
        # Ensure datetime column exists
        if 'pickup_datetime' not in df.columns:
            self.logger.warning("No pickup_datetime column found")
            self.logger.debug(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        self.logger.debug(f"pickup_datetime column found with dtype: {df['pickup_datetime'].dtype}")
        self.logger.debug(f"Sample pickup times: {df['pickup_datetime'].head().tolist()}")
        
        # Create time window filter with optimized operations
        try:
            # Use query for better performance on large DataFrames
            start_str = time_window.start.strftime('%Y-%m-%d %H:%M:%S')
            end_str = time_window.end.strftime('%Y-%m-%d %H:%M:%S')
            
            self.logger.debug(f"Creating optimized filter for time range: {start_str} to {end_str}")
            
            # Use vectorized operations with explicit filtering
            mask = (
                (df['pickup_datetime'] >= time_window.start) &
                (df['pickup_datetime'] < time_window.end)
            )
            
            mask_count = mask.sum()
            self.logger.debug(f"Filtering mask created, {mask_count} trips match time window")
            
            if mask_count == 0:
                self.logger.debug("No trips match time window, returning empty DataFrame")
                return pd.DataFrame()
            
            # Use .loc for efficient filtering
            self.logger.debug("Applying filter to DataFrame...")
            filtered = df.loc[mask].copy()
            
            # Log memory usage after filtering
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                self.logger.debug(f"Memory after filtering: RSS={memory_info.rss//1024//1024}MB")
            except ImportError:
                pass
            
            self.logger.debug(f"Filtered {len(df)} trips to {len(filtered)} for {time_window}")
            
            return filtered
            
        except Exception as filter_error:
            self.logger.error(f"Error during time window filtering: {filter_error}", exc_info=True)
            return pd.DataFrame() 