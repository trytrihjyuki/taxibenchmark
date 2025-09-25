"""Data loading and processing modules."""

from .s3_loader import S3DataLoader
from .s3_uploader import S3ResultsUploader
from .processor import DataProcessor
from .data_loader import ExperimentDataLoader

# Enhanced data manager with S3 integration
class DataManager:
    """Enhanced data manager with full S3 integration."""
    
    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config
        self.s3_loader = S3DataLoader(config)
        self.s3_uploader = S3ResultsUploader(config)
        self.processor = DataProcessor(config)
        
        # Simple in-memory cache
        self._cache = {}
    
    def get_trips(self, vehicle_type, date, borough, start_time, end_time):
        """Get trips for a specific time window."""
        # Simple implementation - in full version would use proper caching
        try:
            # Load day's data
            from ..core.types import VehicleType, Borough
            
            vehicle_type_enum = VehicleType(vehicle_type) if isinstance(vehicle_type, str) else vehicle_type
            borough_enum = Borough(borough) if isinstance(borough, str) else borough
            
            day_data = self.s3_loader.load_trip_data(
                date, vehicle_type_enum, [borough_enum]
            )
            
            if day_data.empty:
                return None
            
            # Filter by time window
            from ..core.types import TimeWindow
            time_window = TimeWindow(start=start_time, end=end_time)
            return self.processor.filter_by_time_window(day_data, time_window)
            
        except Exception as e:
            return None

__all__ = [
    'S3DataLoader',
    'S3ResultsUploader', 
    'DataProcessor',
    'DataManager',
    'ExperimentDataLoader'
] 