"""LinUCB pricing method implementation (Baseline from Chu et al.).

This implements the Linear Upper Confidence Bound (LinUCB) algorithm
used as a baseline in Hikima et al.'s experiments.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from .base import BasePricingMethod


class LinUCBMethod(BasePricingMethod):
    """
    LinUCB (Linear Upper Confidence Bound) pricing method.
    
    This is a contextual multi-armed bandit approach that learns
    optimal pricing strategies over time.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LinUCB pricing method."""
        super().__init__(config)
        
        # LinUCB-specific parameters
        self.ucb_alpha = config.get('ucb_alpha', 0.5)
        self.base_price = config.get('base_price', 5.875)
        self.price_multipliers = config.get('price_multipliers', [0.6, 0.8, 1.0, 1.2, 1.4])
        self.num_arms = len(self.price_multipliers)
        
        # Feature dimensions (will be set based on data)
        self.feature_dim = None
        
        # Initialize parameters for each arm
        self.reset_parameters()
        
        # Try to load pre-trained parameters if available
        self.load_pretrained = config.get('linucb_load_pretrained', False)
        if self.load_pretrained:
            self._load_pretrained_parameters(config)
    
    def reset_parameters(self):
        """Reset LinUCB parameters."""
        # Default feature dimension
        if self.feature_dim is None:
            # Estimate: hour (10) + locations (100) + trip features (2) 
            self.feature_dim = 112
        
        # A matrices for each arm (feature covariance)
        self.A = [np.eye(self.feature_dim) for _ in range(self.num_arms)]
        
        # b vectors for each arm (reward weighted features)
        self.b = [np.zeros(self.feature_dim) for _ in range(self.num_arms)]
        
        # Theta parameters (will be computed from A and b)
        self.theta = [np.zeros(self.feature_dim) for _ in range(self.num_arms)]
    
    def get_method_name(self) -> str:
        """Get method name."""
        return "LinUCB"
    
    def compute_prices(
        self,
        scenario_data: Dict[str, Any],
        acceptance_function: str
    ) -> np.ndarray:
        """
        Compute prices using the LinUCB algorithm optimized for specific acceptance function.
        
        Args:
            scenario_data: Dictionary with scenario data
            acceptance_function: 'PL' or 'Sigmoid' - which function to optimize for
            
        Returns:
            Array of prices for each requester
        """
        n_requesters = scenario_data['num_requesters']
        n_taxis = scenario_data['num_taxis']
        trip_amounts = scenario_data['trip_amounts']
        trip_distances = scenario_data.get('trip_distances', np.ones(n_requesters))
        
        # Handle edge cases
        if n_requesters == 0:
            return np.array([])
        if n_taxis == 0:
            return np.zeros(n_requesters)
        
        # Extract features for each requester
        features = self._extract_features(scenario_data)
        
        # Update feature dimension if needed
        if features.shape[1] != self.feature_dim:
            self.feature_dim = features.shape[1]
            self.reset_parameters()
        
        # Compute theta for each arm
        for k in range(self.num_arms):
            try:
                self.theta[k] = np.linalg.solve(self.A[k], self.b[k])
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                self.theta[k] = np.linalg.pinv(self.A[k]) @ self.b[k]
        
        # Select arms (prices) for each requester
        prices = np.zeros(n_requesters)
        selected_arms = np.zeros(n_requesters, dtype=int)
        
        for i in range(n_requesters):
            x = features[i]
            
            # Compute UCB for each arm
            ucb_values = np.zeros(self.num_arms)
            for k in range(self.num_arms):
                # Mean reward estimate
                mean_reward = np.dot(self.theta[k], x)
                
                # Confidence width
                try:
                    confidence = self.ucb_alpha * np.sqrt(
                        np.dot(x, np.linalg.solve(self.A[k], x))
                    )
                except np.linalg.LinAlgError:
                    # If singular, use pseudo-inverse
                    confidence = self.ucb_alpha * np.sqrt(
                        np.dot(x, np.linalg.pinv(self.A[k]) @ x)
                    )
                
                ucb_values[k] = mean_reward + confidence
            
            # Select arm with highest UCB
            selected_arm = np.argmax(ucb_values)
            selected_arms[i] = selected_arm
            
            # Compute price exactly like Hikima (line 790 in both experiments)
            price_multiplier = self.price_multipliers[selected_arm]
            prices[i] = self.base_price * price_multiplier * trip_distances[i]
        
        # Store selected arms for potential update
        self.last_selected_arms = selected_arms
        self.last_features = features
        
        return prices
    
    def update_parameters(
        self,
        features: np.ndarray,
        selected_arms: np.ndarray,
        rewards: np.ndarray
    ):
        """
        Update LinUCB parameters based on observed rewards.
        
        Args:
            features: Feature matrix for requesters
            selected_arms: Selected arms for each requester
            rewards: Observed rewards for each requester
        """
        for i in range(len(selected_arms)):
            k = selected_arms[i]
            x = features[i]
            r = rewards[i]
            
            # Update A and b for the selected arm
            self.A[k] += np.outer(x, x)
            self.b[k] += x * r
    
    def _extract_features(self, scenario_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features for LinUCB from scenario data.
        
        Args:
            scenario_data: Dictionary with scenario data
            
        Returns:
            Feature matrix (n_requesters x feature_dim)
        """
        n_requesters = scenario_data['num_requesters']
        
        # Extract available features
        features_list = []
        
        # Time features (hour of day)
        hour = scenario_data.get('hour', 12)
        hour_features = self._create_hour_features(hour)
        
        # Location features
        pickup_locations = scenario_data.get('pickup_locations', np.zeros(n_requesters))
        dropoff_locations = scenario_data.get('dropoff_locations', np.zeros(n_requesters))
        location_features = self._create_location_features(
            pickup_locations, dropoff_locations, n_requesters
        )
        
        # Trip features
        trip_distances = scenario_data.get('trip_distances', np.ones(n_requesters))
        trip_durations = scenario_data.get('trip_durations', np.ones(n_requesters) * 600)
        
        # Combine features
        features = np.zeros((n_requesters, len(hour_features) + location_features.shape[1] + 2))
        
        for i in range(n_requesters):
            feature_vec = np.concatenate([
                hour_features,
                location_features[i],
                [trip_distances[i], trip_durations[i] / 3600.0]  # Duration in hours
            ])
            features[i] = feature_vec
        
        return features
    
    def _create_hour_features(self, hour: int, num_hours: int = 10) -> np.ndarray:
        """Create one-hot encoded hour features."""
        # Assume hours 10-19 (10 AM to 7 PM)
        hour_features = np.zeros(num_hours)
        if 10 <= hour < 20:
            hour_features[hour - 10] = 1
        return hour_features
    
    def _create_location_features(
        self,
        pickup_locations: np.ndarray,
        dropoff_locations: np.ndarray,
        n_requesters: int,
        max_locations: int = 50
    ) -> np.ndarray:
        """Create location-based features."""
        # Get unique locations
        all_locations = np.concatenate([pickup_locations, dropoff_locations])
        unique_locations = np.unique(all_locations)
        
        # Limit to max_locations most common
        if len(unique_locations) > max_locations:
            location_counts = np.bincount(all_locations.astype(int))
            top_locations = np.argsort(location_counts)[-max_locations:]
            location_map = {loc: i for i, loc in enumerate(top_locations)}
        else:
            location_map = {loc: i for i, loc in enumerate(unique_locations)}
        
        # Create features
        n_location_features = min(len(unique_locations), max_locations)
        features = np.zeros((n_requesters, n_location_features * 2))
        
        for i in range(n_requesters):
            # Pickup location features
            if pickup_locations[i] in location_map:
                features[i, location_map[pickup_locations[i]]] = 1
            
            # Dropoff location features
            if dropoff_locations[i] in location_map:
                features[i, n_location_features + location_map[dropoff_locations[i]]] = 1
        
        return features
    
    def _load_pretrained_parameters(self, config: Dict[str, Any]):
        """Load pre-trained LinUCB parameters from S3 or local storage."""
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        
        # Check if we should load from S3
        use_s3 = config.get('linucb_use_s3', True)
        
        if use_s3:
            # S3 configuration
            s3_bucket = config.get('s3_bucket', 'taxi-benchmark')
            borough = config.get('borough', 'Queens')
            month = config.get('month', '201909')
            s3_prefix = f"models/work/learned_matrix_PL/{month}_{borough}"
            
            try:
                # Initialize S3 client
                s3 = boto3.client('s3')
                
                # Load A matrices and b vectors for each arm
                for k in range(self.num_arms):
                    # Load A matrix
                    a_key = f"{s3_prefix}/A_{k}_09"
                    try:
                        response = s3.get_object(Bucket=s3_bucket, Key=a_key)
                        self.A[k] = pickle.loads(response['Body'].read())
                        self.logger.debug(f"Loaded A_{k} from S3: {a_key}")
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'NoSuchKey':
                            self.logger.warning(f"A matrix not found in S3: {a_key}")
                        else:
                            raise
                    
                    # Load b vector
                    b_key = f"{s3_prefix}/b_{k}_09"
                    try:
                        response = s3.get_object(Bucket=s3_bucket, Key=b_key)
                        self.b[k] = pickle.loads(response['Body'].read())
                        self.logger.debug(f"Loaded b_{k} from S3: {b_key}")
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'NoSuchKey':
                            self.logger.warning(f"b vector not found in S3: {b_key}")
                        else:
                            raise
                
                # Also try to load from multiple months (following Hikima pattern)
                months_to_try = ['201907', '201908', '201909']
                combined_A = [np.zeros_like(self.A[k]) for k in range(self.num_arms)]
                combined_b = [np.zeros_like(self.b[k]) for k in range(self.num_arms)]
                
                for month_str in months_to_try:
                    month_suffix = month_str[-2:]  # Get last 2 digits (07, 08, 09)
                    s3_prefix_month = f"models/work/learned_matrix_PL/{month_str}_{borough}"
                    
                    for k in range(self.num_arms):
                        try:
                            # Load A matrix for this month
                            a_key = f"{s3_prefix_month}/A_{k}_{month_suffix}"
                            response = s3.get_object(Bucket=s3_bucket, Key=a_key)
                            matrix_data = pickle.loads(response['Body'].read())
                            if matrix_data.shape == self.A[k].shape:
                                combined_A[k] += matrix_data
                            
                            # Load b vector for this month
                            b_key = f"{s3_prefix_month}/b_{k}_{month_suffix}"
                            response = s3.get_object(Bucket=s3_bucket, Key=b_key)
                            vector_data = pickle.loads(response['Body'].read())
                            if vector_data.shape == self.b[k].shape:
                                combined_b[k] += vector_data
                                
                        except Exception:
                            pass  # Skip if not found
                
                # Use combined matrices if we got data
                for k in range(self.num_arms):
                    if not np.allclose(combined_A[k], 0):
                        self.A[k] = combined_A[k] + np.eye(self.A[k].shape[0])
                        self.b[k] = combined_b[k]
                
                self.logger.info(f"Loaded pretrained LinUCB parameters from S3: {s3_bucket}/{s3_prefix}")
                
            except NoCredentialsError:
                self.logger.warning("No AWS credentials found, falling back to local loading")
                self._load_from_local(config)
            except Exception as e:
                self.logger.warning(f"Failed to load from S3: {e}, falling back to local")
                self._load_from_local(config)
        else:
            self._load_from_local(config)
    
    def _load_from_local(self, config: Dict[str, Any]):
        """Load pre-trained parameters from local filesystem."""
        pretrained_dir = config.get('linucb_pretrained_dir', 'data/pretrained/linucb')
        pretrained_path = Path(pretrained_dir)
        
        if not pretrained_path.exists():
            self.logger.warning(f"Pretrained directory {pretrained_dir} not found")
            return
        
        try:
            # Load A matrices
            for k in range(self.num_arms):
                a_file = pretrained_path / f'A_{k}.pkl'
                if a_file.exists():
                    with open(a_file, 'rb') as f:
                        self.A[k] = pickle.load(f)
                
                # Load b vectors
                b_file = pretrained_path / f'b_{k}.pkl'
                if b_file.exists():
                    with open(b_file, 'rb') as f:
                        self.b[k] = pickle.load(f)
            
            self.logger.info(f"Loaded pretrained LinUCB parameters from {pretrained_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load pretrained parameters: {e}")
    
    def save_parameters(self, save_dir: str):
        """
        Save LinUCB parameters to disk.
        
        Args:
            save_dir: Directory to save parameters
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for k in range(self.num_arms):
            # Save A matrix
            with open(save_path / f'A_{k}.pkl', 'wb') as f:
                pickle.dump(self.A[k], f)
            
            # Save b vector
            with open(save_path / f'b_{k}.pkl', 'wb') as f:
                pickle.dump(self.b[k], f)
        
        self.logger.info(f"Saved LinUCB parameters to {save_dir}")
    
    def execute(
        self,
        scenario_data: Dict[str, Any],
        num_simulations: int = 1
    ) -> Dict[str, Any]:
        """
        Execute LinUCB with online learning.
        
        Overrides base execute to include parameter updates.
        """
        # Run base execution (which now returns results for both PL and Sigmoid)
        results = super().execute(scenario_data, num_simulations)
        
        # If we have results and selected arms, update parameters
        # Use PL results for updating (as per original Hikima implementation)
        if hasattr(self, 'last_selected_arms') and 'PL' in results:
            # Simulate rewards based on matching results from PL
            pl_results = results['PL']
            avg_reward = pl_results.get('avg_revenue', 0) / scenario_data['num_requesters'] \
                        if scenario_data['num_requesters'] > 0 else 0
            rewards = np.ones(len(self.last_selected_arms)) * avg_reward
            
            # Update parameters
            self.update_parameters(
                self.last_features,
                self.last_selected_arms,
                rewards
            )
        
        return results 