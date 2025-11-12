"""S3 data loader for taxi trip data and supporting datasets."""

import os
from datetime import date
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import boto3
import io
import pickle
from pathlib import Path

from ..core import get_logger
from ..core.types import VehicleType, Borough


class S3DataLoader:
    """S3 data loader for taxi benchmark datasets."""
    
    def __init__(self, config=None):
        """Initialize loader."""
        self.logger = get_logger('data.loader')
        
        # Ensure bucket name is always a string
        bucket_env = os.getenv('S3_BUCKET', 'magisterka')
        self.bucket_name = str(bucket_env) if bucket_env is not None else 'magisterka'
        
        region_env = os.getenv('AWS_REGION', 'eu-north-1') 
        self.aws_region = str(region_env) if region_env is not None else 'eu-north-1'
        
        # Don't store S3 client as instance variable to avoid pickle issues
        # Create client when needed in methods
        
        self.logger.info(f"S3 Loader initialized - bucket: {self.bucket_name} (type: {type(self.bucket_name)})")
    
    def _get_s3_client(self):
        """Create S3 client on demand."""
        return boto3.client('s3', region_name=self.aws_region)
    
    def load_trip_data(
        self,
        processing_date: date,
        vehicle_type: VehicleType,
        boroughs: List[Borough]
    ) -> pd.DataFrame:
        """
        Load trip data from S3 using Hikima's exact approach.
        
        This replicates the data loading from experiment_PL.py and experiment_Sigmoid.py
        to ensure 1:1 alignment with original implementation.
        
        Args:
            processing_date: Date to process
            vehicle_type: Type of vehicle
            boroughs: List of boroughs
        
        Returns:
            DataFrame with trip data (pre-filtered like Hikima)
        """
        # Build S3 key with correct format: datasets/green/year=2019/month=10/green_tripdata_2019-10.parquet
        year = processing_date.strftime('%Y')
        month = processing_date.strftime('%m')
        year_month = processing_date.strftime('%Y-%m')
        filename = f"{vehicle_type.value}_tripdata_{year_month}.parquet"
        s3_key = f"datasets/{vehicle_type.value}/year={year}/month={month}/{filename}"
        
        self.logger.info(f"Loading trip data from s3://{self.bucket_name}/{s3_key}")
        
        # Log pandas and pyarrow versions for debugging
        try:
            import pandas as pd_version_check
            import pyarrow as pa_version_check
            self.logger.debug(f"pandas version: {pd_version_check.__version__}")
            self.logger.debug(f"pyarrow version: {pa_version_check.__version__}")
        except ImportError as e:
            self.logger.debug(f"Version check failed: {e}")
        
        try:
            # Create client when needed
            self.logger.debug("Creating S3 client")
            s3_client = self._get_s3_client()
            self.logger.debug("S3 client created successfully")
            
            # Download parquet file
            self.logger.debug(f"Downloading parquet file from S3: {s3_key}")
            response = s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            self.logger.debug(f"S3 response received, content length: {response.get('ContentLength', 'unknown')}")
            
            # Read response body with detailed logging
            self.logger.debug("Starting to read S3 response body...")
            import sys
            sys.stdout.flush()  # Force flush logs before potential crash
            
            try:
                body_data = response['Body'].read()
                self.logger.debug(f"S3 body read successfully, {len(body_data)} bytes")
                sys.stdout.flush()
                
                self.logger.debug("Creating BytesIO buffer...")
                bytes_buffer = io.BytesIO(body_data)
                self.logger.debug(f"BytesIO buffer created, size: {len(body_data)} bytes")
                sys.stdout.flush()
                
                # Check if we can seek to validate the buffer
                self.logger.debug("Validating BytesIO buffer...")
                initial_pos = bytes_buffer.tell()
                bytes_buffer.seek(0, 2)  # Seek to end
                buffer_size = bytes_buffer.tell()
                bytes_buffer.seek(initial_pos)  # Seek back to start
                self.logger.debug(f"Buffer validation successful, size: {buffer_size} bytes")
                sys.stdout.flush()
                
                # Add memory info if psutil is available
                try:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    self.logger.debug(f"Memory before parquet parsing: RSS={memory_info.rss//1024//1024}MB, VMS={memory_info.vms//1024//1024}MB")
                except ImportError:
                    self.logger.debug("psutil not available for memory monitoring")
                sys.stdout.flush()
                
                # Try to peek at parquet file metadata first
                self.logger.debug("Checking parquet file metadata...")
                parquet_file = None
                total_rows = 0
                try:
                    import pyarrow.parquet as pq
                    bytes_buffer.seek(0)  # Reset position
                    parquet_file = pq.ParquetFile(bytes_buffer)
                    total_rows = parquet_file.metadata.num_rows
                    self.logger.debug(f"Parquet metadata: {parquet_file.num_row_groups} row groups, {total_rows} total rows")
                    self.logger.debug(f"Parquet schema: {[col.name for col in parquet_file.schema]}")
                except Exception as meta_error:
                    self.logger.warning(f"Could not read parquet metadata: {meta_error}")
                    total_rows = 0
                
                # Reset buffer position for actual reading
                bytes_buffer.seek(0)
                
                # Track whether date filtering was done during parsing
                date_filtering_done = False
                
                if total_rows > 5_000_000:  # Large file - use chunked reading
                    self.logger.debug(f"Large file detected ({total_rows:,} rows), using chunked reading with date filtering...")
                    df = self._read_large_parquet_chunked(bytes_buffer, total_rows, processing_date)
                    date_filtering_done = True  # Date filtering was done during chunked reading
                else:
                    self.logger.debug("Starting parquet parsing with pandas...")
                    sys.stdout.flush()
                    df = self._read_parquet_with_fallback(bytes_buffer)
                
                self.logger.debug(f"Parquet file loaded successfully, shape: {df.shape}")
                sys.stdout.flush()
                
            except Exception as parquet_error:
                self.logger.error(f"Failed to parse parquet file: {parquet_error}", exc_info=True)
                
                # Try alternative parsing methods
                self.logger.debug("Attempting alternative parquet parsing with pyarrow...")
                try:
                    import pyarrow.parquet as pq
                    bytes_buffer = io.BytesIO(body_data)
                    table = pq.read_table(bytes_buffer)
                    df = table.to_pandas()
                    self.logger.debug(f"Alternative parsing succeeded, shape: {df.shape}")
                except Exception as alt_error:
                    self.logger.error(f"Alternative parquet parsing also failed: {alt_error}", exc_info=True)
                    
                    # Final emergency fallback - read just a sample of the data
                    self.logger.debug("Attempting emergency fallback with data sampling...")
                    try:
                        import pyarrow.parquet as pq
                        bytes_buffer = io.BytesIO(body_data)
                        
                        # Read only first 100,000 rows as emergency fallback
                        parquet_file = pq.ParquetFile(bytes_buffer)
                        first_batch = next(parquet_file.iter_batches(batch_size=100000))
                        df = first_batch.to_pandas()
                        
                        self.logger.warning(f"Emergency fallback successful - using SAMPLE DATA ONLY: {df.shape}")
                        self.logger.warning("WARNING: Only using first 100,000 rows due to parsing failures!")
                        
                    except Exception as emergency_error:
                        self.logger.error(f"Emergency fallback also failed: {emergency_error}", exc_info=True)
                        raise parquet_error  # Re-raise original error
            
            # Convert pickup datetime to datetime if it's string
            self.logger.debug(f"Available columns: {list(df.columns)}")
            if 'lpep_pickup_datetime' in df.columns:
                self.logger.debug("Found lpep_pickup_datetime, converting to pickup_datetime")
                df['pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
            elif 'tpep_pickup_datetime' in df.columns:
                self.logger.debug("Found tpep_pickup_datetime, converting to pickup_datetime")
                df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
            else:
                self.logger.warning("No recognized pickup datetime column found")
            
            if 'pickup_datetime' in df.columns:
                self.logger.debug(f"pickup_datetime dtype: {df['pickup_datetime'].dtype}")
                self.logger.debug(f"Sample pickup_datetime values: {df['pickup_datetime'].head().tolist()}")
            
            # Filter by date with enhanced error handling and memory optimization
            # Skip if date filtering was already done during chunked reading
            if not date_filtering_done:
                self.logger.debug(f"Filtering by date: {processing_date}")
                initial_count = len(df)
                
                try:
                    # Log memory before date filtering
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        self.logger.debug(f"Memory before date filtering: RSS={memory_info.rss//1024//1024}MB")
                    except ImportError:
                        pass
                
                    # Check if we have the pickup_datetime column
                    if 'pickup_datetime' not in df.columns:
                        self.logger.error("pickup_datetime column missing for date filtering")
                        return pd.DataFrame()
                    
                    # Use more memory-efficient date filtering
                    self.logger.debug(f"Applying date filter to {len(df)} rows...")
                    sys.stdout.flush()
                    
                    # Convert target date to datetime range for faster comparison
                    target_date_start = pd.Timestamp(processing_date)
                    target_date_end = target_date_start + pd.Timedelta(days=1)
                    
                    # Use datetime range instead of .dt.date conversion (more memory efficient)
                    date_mask = (
                        (df['pickup_datetime'] >= target_date_start) &
                        (df['pickup_datetime'] < target_date_end)
                    )
                    
                    self.logger.debug(f"Date mask created, {date_mask.sum()} trips match")
                    sys.stdout.flush()
                    
                    # Apply the filter
                    df = df.loc[date_mask].copy()
                    
                    self.logger.debug(f"Date filtering: {initial_count} -> {len(df)} trips")
                    sys.stdout.flush()
                    
                except Exception as date_filter_error:
                    self.logger.error(f"Date filtering failed: {date_filter_error}", exc_info=True)
                    
                    # Fallback: try with smaller chunks if the full operation failed
                    try:
                        self.logger.debug("Attempting fallback chunked date filtering...")
                        filtered_chunks = []
                        chunk_size = 100_000
                        
                        for start_idx in range(0, len(df), chunk_size):
                            end_idx = min(start_idx + chunk_size, len(df))
                            chunk = df.iloc[start_idx:end_idx]
                            
                            # Apply date filter to chunk
                            target_date_start = pd.Timestamp(processing_date)
                            target_date_end = target_date_start + pd.Timedelta(days=1)
                            
                            chunk_mask = (
                                (chunk['pickup_datetime'] >= target_date_start) &
                                (chunk['pickup_datetime'] < target_date_end)
                            )
                            
                            filtered_chunk = chunk.loc[chunk_mask]
                            if not filtered_chunk.empty:
                                filtered_chunks.append(filtered_chunk)
                            
                            self.logger.debug(f"Processed chunk {start_idx//chunk_size + 1}: {len(filtered_chunk)} trips")
                        
                        if filtered_chunks:
                            df = pd.concat(filtered_chunks, ignore_index=True)
                            self.logger.debug(f"Fallback chunked filtering succeeded: {len(df)} trips")
                        else:
                            self.logger.warning("No trips found for the specified date")
                            df = pd.DataFrame()
                            
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback date filtering also failed: {fallback_error}", exc_info=True)
                        return pd.DataFrame()
            else:
                self.logger.debug(f"Date filtering already done during chunked reading, current dataset: {len(df)} trips")
            
            # Filter by boroughs if area mapping is available
            if boroughs:
                # Load area mapping
                area_info = self.load_area_info()
                if area_info is not None:
                    borough_location_ids = set()
                    for borough in boroughs:
                        borough_ids = area_info[
                            area_info['borough'] == borough.value
                        ]['LocationID'].unique()
                        borough_location_ids.update(borough_ids)
                    
                    if borough_location_ids:
                        df = df[df['PULocationID'].isin(borough_location_ids)]
                        
                        # Create mapping without using LocationID as index (since it can repeat)
                        # Use merge instead of map to handle duplicate LocationIDs properly
                        df = df.merge(
                            area_info[['LocationID', 'borough']],
                            left_on='PULocationID',
                            right_on='LocationID',
                            how='left'
                        ).drop(columns=['LocationID'])
                    else:
                        # Fallback - assign borough randomly for demo
                        df['borough'] = [b.value for b in boroughs][0]
                else:
                    # No area info - assign first borough
                    df['borough'] = [b.value for b in boroughs][0]
            
            # Clean data - align with Hikima's filtering (line 144, 147)
            df = df[df['trip_distance'] > 10**(-3)]  # Min distance (align with Hikima)
            df = df[df['total_amount'] > 10**(-3)]   # Min amount (align with Hikima)
            df = df.dropna(subset=['PULocationID', 'DOLocationID', 'trip_distance', 'total_amount'])
            
            # Filter LocationIDs < 264 (align with Hikima line 144, 147)
            df = df[(df['PULocationID'] < 264) & (df['DOLocationID'] < 264)]
            
            self.logger.info(f"Loaded {len(df)} trips for {processing_date} after filtering")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load trip data: {e}")
            return pd.DataFrame()
    
    def load_area_info(self) -> Optional[pd.DataFrame]:
        """Load taxi zone area information."""
        try:
            s3_client = self._get_s3_client()
            
            # Try to load area info with correct path
            s3_key = 'datasets/area_information.csv'
            response = s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            
            # Rename columns for consistency
            df = df.rename(columns={'Borough': 'borough'})
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Could not load area info: {e}")
            return None
    
    def load_linucb_matrices(
        self, 
        months: List[str], 
        acceptance_function: str, 
        borough: str = 'Manhattan'
    ) -> Dict[str, Any]:
        """
        Load pre-trained LinUCB matrices from S3 exactly like Hikima's setup.
        
        This loads all A_0 through A_4 and b_0 through b_4 matrices for each acceptance function.
        
        Args:
            months: List of months to load (e.g., ['201907', '201908', '201909'])
            acceptance_function: 'PL' or 'Sigmoid' 
            borough: Borough name (default: 'Manhattan')
        
        Returns:
            Dictionary with matrices for each arm: {'A_0': matrix, 'A_1': matrix, ..., 'b_0': vector, ...}
        """
        matrices = {}
        
        # Initialize matrices for each arm (0-4)
        for arm in range(5):
            matrices[f'A_{arm}'] = None
            matrices[f'b_{arm}'] = None
        
        try:
            s3_client = self._get_s3_client()
            
            # Load matrices for each month and combine them (just like in Hikima's code)
            for month in months:
                self.logger.info(f"Loading LinUCB matrices for {month} {acceptance_function} {borough}")
                
                # Load A matrices for each arm
                for arm in range(5):
                    # A matrices: models/work/learned_matrix_PL/201908_Manhattan/A_0_08
                    s3_key = f'models/work/learned_matrix_{acceptance_function}/{month}_{borough}/A_{arm}_{month[-2:]}'
                    
                    try:
                        response = s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                        arm_matrix = pickle.loads(response['Body'].read())
                        
                        if matrices[f'A_{arm}'] is None:
                            matrices[f'A_{arm}'] = arm_matrix
                        else:
                            # Add matrices like in Hikima's code (lines 70-74, 79-83, etc.)
                            matrices[f'A_{arm}'] += arm_matrix
                        
                        self.logger.debug(f"Loaded A_{arm} for {month}")
                        
                    except Exception as e:
                        self.logger.warning(f"Could not load A_{arm} for {month}: {e}")
                
                # Load b vectors for each arm
                for arm in range(5):
                    # b vectors: models/work/learned_matrix_PL/201908_Manhattan/b_0_08
                    s3_key = f'models/work/learned_matrix_{acceptance_function}/{month}_{borough}/b_{arm}_{month[-2:]}'
                    
                    try:
                        response = s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                        arm_vector = pickle.loads(response['Body'].read())
                        
                        if matrices[f'b_{arm}'] is None:
                            matrices[f'b_{arm}'] = arm_vector
                        else:
                            # Add vectors like in Hikima's code (lines 105-108, 110-113, etc.)
                            matrices[f'b_{arm}'] += arm_vector
                        
                        self.logger.debug(f"Loaded b_{arm} for {month}")
                        
                    except Exception as e:
                        self.logger.warning(f"Could not load b_{arm} for {month}: {e}")
            
            # Add identity matrix to A matrices like in Hikima's code (line 74, 81, 88, 95, 102)
            for arm in range(5):
                if matrices[f'A_{arm}'] is not None:
                    matrices[f'A_{arm}'] += np.eye(matrices[f'A_{arm}'].shape[0])
                    self.logger.debug(f"Added identity to A_{arm}")
            
            self.logger.info(f"Successfully loaded LinUCB matrices for {acceptance_function} {borough}")
            
        except Exception as e:
            self.logger.error(f"Failed to load LinUCB matrices: {e}")
        
        return matrices 
    
    def _read_parquet_with_fallback(self, bytes_buffer: io.BytesIO) -> pd.DataFrame:
        """Read parquet with multiple engine fallbacks."""
        import sys
        
        # Strategy 1: pandas with pyarrow engine (default)
        try:
            self.logger.debug("Attempting pandas.read_parquet with pyarrow engine...")
            sys.stdout.flush()
            bytes_buffer.seek(0)
            df = pd.read_parquet(bytes_buffer, engine='pyarrow')
            self.logger.debug("Successfully parsed with pandas + pyarrow engine")
            return df
        except Exception as e1:
            self.logger.warning(f"pandas + pyarrow failed: {e1}")
        
        # Strategy 2: Direct pyarrow
        try:
            self.logger.debug("Attempting direct pyarrow parsing...")
            sys.stdout.flush()
            import pyarrow.parquet as pq
            bytes_buffer.seek(0)
            table = pq.read_table(bytes_buffer)
            df = table.to_pandas()
            self.logger.debug("Successfully parsed with direct pyarrow")
            return df
        except Exception as e2:
            self.logger.warning(f"Direct pyarrow failed: {e2}")
        
        # Strategy 3: pandas with fastparquet engine (if available)
        try:
            self.logger.debug("Attempting pandas.read_parquet with fastparquet engine...")
            sys.stdout.flush()
            bytes_buffer.seek(0)
            df = pd.read_parquet(bytes_buffer, engine='fastparquet')
            self.logger.debug("Successfully parsed with pandas + fastparquet engine")
            return df
        except Exception as e3:
            self.logger.warning(f"pandas + fastparquet failed: {e3}")
        
        # All strategies failed  
        raise RuntimeError("All parsing strategies failed. Check previous warning messages for specific errors.")
    
    def _read_large_parquet_chunked(self, bytes_buffer: io.BytesIO, total_rows: int, processing_date=None) -> pd.DataFrame:
        """Read large parquet file in chunks to avoid memory issues."""
        import sys
        import pyarrow.parquet as pq
        
        try:
            self.logger.debug(f"Reading {total_rows:,} rows in chunks...")
            bytes_buffer.seek(0)
            parquet_file = pq.ParquetFile(bytes_buffer)
            
            # Determine chunk size based on total rows (conservative for memory management)
            if total_rows > 10_000_000:
                chunk_size = 250_000  # 250K rows per chunk for very large files
            elif total_rows > 5_000_000:
                chunk_size = 500_000  # 500K rows per chunk for large files  
            elif total_rows > 1_000_000:
                chunk_size = 750_000  # 750K rows per chunk for medium files
            else:
                chunk_size = total_rows  # Read all at once for small files
                
            self.logger.debug(f"Using chunk size: {chunk_size:,} rows")
            
            # Define essential columns for the experiment (reduces memory usage)
            essential_columns = [
                'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                'lpep_pickup_datetime', 'lpep_dropoff_datetime', 
                'PULocationID', 'DOLocationID', 
                'trip_distance', 'total_amount',
                'passenger_count', 'fare_amount'
            ]
            
            # Check which columns actually exist in the file
            available_columns = [col.name for col in parquet_file.schema]
            columns_to_read = [col for col in essential_columns if col in available_columns]
            
            if len(columns_to_read) < len(essential_columns):
                missing_cols = set(essential_columns) - set(columns_to_read)
                self.logger.debug(f"Missing columns (will read all): {missing_cols}")
                columns_to_read = None  # Read all columns if essential ones are missing
            else:
                self.logger.debug(f"Reading optimized column set: {columns_to_read}")
            
            chunks = []
            rows_read = 0
            
            # Check initial memory
            try:
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info()
                self.logger.debug(f"Initial memory: RSS={initial_memory.rss//1024//1024}MB, VMS={initial_memory.vms//1024//1024}MB")
            except ImportError:
                self.logger.debug("Starting chunked processing (psutil not available for memory monitoring)")
            
            # Prepare date filtering parameters if provided
            target_date_start = None
            target_date_end = None
            if processing_date:
                target_date_start = pd.Timestamp(processing_date)
                target_date_end = target_date_start + pd.Timedelta(days=1)
                self.logger.debug(f"Will apply date filtering during chunked reading: {target_date_start} to {target_date_end}")
            
            # Wrap chunked reading in comprehensive error handling
            try:
                for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns_to_read):
                    try:
                        self.logger.debug(f"Processing chunk: rows {rows_read:,} to {rows_read + len(batch):,}")
                        sys.stdout.flush()
                        
                        chunk_df = batch.to_pandas()
                        initial_chunk_size = len(chunk_df)
                        
                        # Apply date filtering to chunk if processing_date provided
                        if processing_date:
                            # Convert datetime column if needed and ensure pickup_datetime exists
                            if 'pickup_datetime' not in chunk_df.columns:
                                if 'tpep_pickup_datetime' in chunk_df.columns:
                                    chunk_df['pickup_datetime'] = pd.to_datetime(chunk_df['tpep_pickup_datetime'])
                                elif 'lpep_pickup_datetime' in chunk_df.columns:
                                    chunk_df['pickup_datetime'] = pd.to_datetime(chunk_df['lpep_pickup_datetime'])
                            
                            # Apply date filter to chunk if pickup_datetime is available
                            if 'pickup_datetime' in chunk_df.columns:
                                date_mask = (
                                    (chunk_df['pickup_datetime'] >= target_date_start) &
                                    (chunk_df['pickup_datetime'] < target_date_end)
                                )
                                chunk_df = chunk_df.loc[date_mask]
                                self.logger.debug(f"Chunk date filtering: {initial_chunk_size} -> {len(chunk_df)} trips")
                        
                        # Only keep chunks that have data after filtering
                        if not chunk_df.empty:
                            chunks.append(chunk_df)
                            
                            # Memory management: if we've accumulated too many chunks, combine them
                            if len(chunks) >= 10:  # Combine every 10 chunks to manage memory
                                self.logger.debug(f"Combining {len(chunks)} chunks to manage memory...")
                                combined_chunk = pd.concat(chunks, ignore_index=True)
                                chunks = [combined_chunk]  # Replace with single combined chunk
                        
                        rows_read += len(batch)
                        
                        # Log memory usage periodically  
                        if rows_read % (chunk_size * 5) == 0:  # Every 5 chunks
                            try:
                                import psutil
                                process = psutil.Process()
                                memory_info = process.memory_info()
                                self.logger.debug(f"Memory after processing {rows_read:,} rows: RSS={memory_info.rss//1024//1024}MB, active chunks: {len(chunks)}")
                            except ImportError:
                                self.logger.debug(f"Processed {rows_read:,} rows, active chunks: {len(chunks)}")
                                
                    except Exception as chunk_error:
                        self.logger.error(f"Error processing chunk at rows {rows_read:,}: {chunk_error}")
                        # Continue with next chunk rather than failing completely
                        rows_read += len(batch) if 'batch' in locals() else chunk_size
                        continue
                    
            except MemoryError as mem_error:
                self.logger.error(f"Memory exhausted during chunked reading at row {rows_read:,}: {mem_error}")
                self.logger.error(f"Processed {len(chunks)} chunks before memory failure")
                raise mem_error
            except KeyboardInterrupt:
                self.logger.warning(f"Chunked reading interrupted by user at row {rows_read:,}")
                raise
            except Exception as read_error:
                self.logger.error(f"Chunked reading loop failed at row {rows_read:,}: {read_error}")
                self.logger.error(f"Processed {len(chunks)} chunks before failure")
                raise read_error
            
            # Final combination step
            try:
                self.logger.debug(f"Combining {len(chunks)} chunks into single DataFrame...")
                sys.stdout.flush()
                
                if not chunks:
                    self.logger.warning("No valid chunks found after processing - returning empty DataFrame")
                    return pd.DataFrame()
                
                df = pd.concat(chunks, ignore_index=True)
                
                # Log final memory usage
                try:
                    import psutil
                    process = psutil.Process()
                    final_memory = process.memory_info()
                    self.logger.debug(f"Final memory: RSS={final_memory.rss//1024//1024}MB, VMS={final_memory.vms//1024//1024}MB")
                except ImportError:
                    pass
                
                self.logger.debug(f"Chunked reading completed: {len(df):,} rows, {len(df.columns)} columns")
                return df
                
            except MemoryError as concat_error:
                self.logger.error(f"Memory exhausted during final concatenation of {len(chunks)} chunks: {concat_error}")
                raise concat_error
            
        except Exception as e:
            self.logger.error(f"Chunked reading failed: {e}")
            # Fallback to regular parsing with reduced memory
            self.logger.debug("Attempting fallback with memory-optimized settings...")
            bytes_buffer.seek(0)
            
            try:
                # Try with specific columns only (reduce memory)
                essential_columns = [
                    'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                    'PULocationID', 'DOLocationID', 
                    'trip_distance', 'total_amount'
                ]
                
                table = pq.read_table(bytes_buffer, columns=essential_columns)
                df = table.to_pandas()
                self.logger.debug(f"Fallback successful with reduced columns: {df.shape}")
                return df
            except Exception as fallback_error:
                self.logger.error(f"All chunked reading strategies failed: {fallback_error}")
                raise e 