"""S3 uploader for experiment results."""

import os
import json
import io
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError

from ..core import get_logger


class S3ResultsUploader:
    """Handles uploading experiment results to S3."""
    
    def __init__(self, config=None):
        """Initialize uploader."""
        self.logger = get_logger('data.uploader')
        
        # S3 configuration
        bucket_env = os.getenv('S3_BUCKET', 'magisterka')
        self.bucket_name = str(bucket_env) if bucket_env is not None else 'magisterka'
        
        region_env = os.getenv('AWS_REGION', 'eu-north-1') 
        self.aws_region = str(region_env) if region_env is not None else 'eu-north-1'
        
        # Results bucket (separate from data bucket)
        self.results_bucket = config.s3_results_bucket if config else "taxi-benchmark"
        
        self.logger.info(f"S3 Results Uploader initialized - results bucket: {self.results_bucket}")
    
    def _get_s3_client(self):
        """Create S3 client on demand."""
        return boto3.client('s3', region_name=self.aws_region)
    
    def upload_experiment_results(
        self, 
        experiment_id: str, 
        local_dir: Path,
        config: Dict[str, Any]
    ) -> str:
        """
        Upload complete experiment results to S3.
        
        Args:
            experiment_id: Unique experiment identifier
            local_dir: Local directory containing results
            config: Experiment configuration
        
        Returns:
            S3 path where results were uploaded
        """
        s3_base_path = f"experiments/{experiment_id}"
        
        try:
            s3_client = self._get_s3_client()
            
            # Upload all parquet files from decisions directory
            decisions_dir = local_dir / 'decisions'
            if decisions_dir.exists():
                self.logger.info(f"Uploading decision files from {decisions_dir}")
                for parquet_file in decisions_dir.glob("*.parquet"):
                    s3_key = f"{s3_base_path}/decisions/{parquet_file.name}"
                    s3_client.upload_file(
                        str(parquet_file), 
                        self.results_bucket, 
                        s3_key
                    )
                    self.logger.debug(f"Uploaded {parquet_file.name} to s3://{self.results_bucket}/{s3_key}")
            
            # Upload summary file if it exists
            summary_file = local_dir / 'experiment_summary.json'
            if summary_file.exists():
                s3_key = f"{s3_base_path}/experiment_summary.json"
                s3_client.upload_file(
                    str(summary_file), 
                    self.results_bucket, 
                    s3_key
                )
                self.logger.info(f"Uploaded summary to s3://{self.results_bucket}/{s3_key}")
            
            # Upload configuration
            config_s3_key = f"{s3_base_path}/experiment_config.json"
            s3_client.put_object(
                Bucket=self.results_bucket,
                Key=config_s3_key,
                Body=json.dumps(config, indent=2, default=str),
                ContentType='application/json'
            )
            
            full_s3_path = f"s3://{self.results_bucket}/{s3_base_path}/"
            self.logger.info(f"Experiment results uploaded to {full_s3_path}")
            return full_s3_path
            
        except NoCredentialsError:
            self.logger.error("AWS credentials not found. Cannot upload to S3.")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to upload experiment results: {e}")
            return ""
    
    def upload_decisions_parquet(
        self, 
        decisions_df: pd.DataFrame, 
        experiment_id: str, 
        time_window_idx: int
    ) -> bool:
        """
        Upload a single decision parquet file to S3.
        
        Args:
            decisions_df: DataFrame with decision data
            experiment_id: Unique experiment identifier
            time_window_idx: Time window index
        
        Returns:
            True if successful, False otherwise
        """
        try:
            s3_client = self._get_s3_client()
            
            # Convert DataFrame to parquet bytes
            buffer = io.BytesIO()
            decisions_df.to_parquet(buffer, index=False)
            buffer.seek(0)
            
            # Upload to S3
            s3_key = f"experiments/{experiment_id}/decisions/time_window_{time_window_idx:04d}.parquet"
            s3_client.put_object(
                Bucket=self.results_bucket,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            self.logger.debug(f"Uploaded decisions for time window {time_window_idx} to s3://{self.results_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload decisions parquet: {e}")
            return False
    
    def list_experiment_results(self, prefix: str = "") -> List[str]:
        """
        List available experiment results in S3.
        
        Args:
            prefix: Optional prefix to filter results
        
        Returns:
            List of experiment IDs
        """
        try:
            s3_client = self._get_s3_client()
            
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.results_bucket,
                Prefix=f"experiments/{prefix}",
                Delimiter='/'
            )
            
            experiment_ids = []
            for page in pages:
                if 'CommonPrefixes' in page:
                    for obj in page['CommonPrefixes']:
                        prefix_path = obj['Prefix']
                        # Extract experiment ID from path: experiments/exp_id/
                        exp_id = prefix_path.split('/')[-2]
                        experiment_ids.append(exp_id)
            
            return sorted(experiment_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to list experiment results: {e}")
            return []
    
    def download_experiment_decisions(
        self, 
        experiment_id: str, 
        local_dir: Path = None
    ) -> pd.DataFrame:
        """
        Download and combine all decision files for an experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            local_dir: Optional local directory to save files
        
        Returns:
            Combined DataFrame with all decisions
        """
        try:
            s3_client = self._get_s3_client()
            
            # List all decision files for this experiment
            prefix = f"experiments/{experiment_id}/decisions/"
            response = s3_client.list_objects_v2(
                Bucket=self.results_bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                self.logger.warning(f"No decision files found for experiment {experiment_id}")
                return pd.DataFrame()
            
            all_decisions = []
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                if s3_key.endswith('.parquet'):
                    # Download parquet file
                    response = s3_client.get_object(Bucket=self.results_bucket, Key=s3_key)
                    df = pd.read_parquet(io.BytesIO(response['Body'].read()))
                    all_decisions.append(df)
                    
                    # Optionally save to local directory
                    if local_dir:
                        local_file = local_dir / Path(s3_key).name
                        df.to_parquet(local_file, index=False)
            
            if all_decisions:
                combined_df = pd.concat(all_decisions, ignore_index=True)
                self.logger.info(f"Downloaded {len(combined_df)} decisions for experiment {experiment_id}")
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to download experiment decisions: {e}")
            return pd.DataFrame() 