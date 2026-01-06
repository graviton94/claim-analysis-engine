"""
Storage Module for Claim Data Intelligence System
Handles Parquet database operations and queries with partitioning support
Phase 2: Parquet Storage Optimization
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import glob
from datetime import datetime


class ParquetStorage:
    """High-performance Parquet database storage manager with partitioning"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize storage manager
        
        Args:
            data_dir: Directory containing Parquet files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.partitioned_dir = self.data_dir / "partitioned"
        self.partitioned_dir.mkdir(parents=True, exist_ok=True)
    
    def save_partitioned(
        self,
        df: pd.DataFrame,
        partition_cols: List[str] = None,
        compression: str = 'snappy',
        basename_template: str = 'part-{i}.parquet'
    ) -> str:
        """
        Save DataFrame to partitioned Parquet with optimized schema
        
        Phase 2.1: Schema Design with partitioning strategy
        - Primary: claim_date (monthly partitions)
        - Secondary: claim_type
        - Tertiary: claim_status
        
        Args:
            df: DataFrame to save
            partition_cols: Columns to partition by (default: ['claim_date_month', 'claim_type', 'claim_status'])
            compression: Compression algorithm (snappy, gzip, brotli, zstd)
            basename_template: Template for partition file names
            
        Returns:
            Path to partitioned dataset
        """
        if df.empty:
            return str(self.partitioned_dir)
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Default partition columns
        if partition_cols is None:
            partition_cols = []
            
            # Add monthly partition if claim_date exists
            if 'claim_date' in df.columns:
                df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
                df['claim_date_month'] = df['claim_date'].dt.to_period('M').astype(str)
                partition_cols.append('claim_date_month')
            
            # Add claim_type if exists
            if 'claim_type' in df.columns:
                partition_cols.append('claim_type')
            
            # Add claim_status if exists
            if 'claim_status' in df.columns:
                partition_cols.append('claim_status')
        
        # Convert to PyArrow Table (let PyArrow infer schema with UTF-8 support)
        table = pa.Table.from_pandas(df, preserve_index=False)
        
        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=str(self.partitioned_dir),
            partition_cols=partition_cols if partition_cols else None,
            compression=compression,
            basename_template=basename_template,
            existing_data_behavior='overwrite_or_ignore',
            use_dictionary=True  # Enable dictionary encoding for categorical columns
        )
        
        print(f"✓ Saved partitioned data to {self.partitioned_dir}")
        if partition_cols:
            print(f"  Partitioned by: {', '.join(partition_cols)}")
        print(f"  Compression: {compression.upper()}")
        
        return str(self.partitioned_dir)
    
    def _create_optimized_schema(self, df: pd.DataFrame, partition_cols: List[str]) -> pa.Schema:
        """
        Create optimized PyArrow schema with appropriate compression strategies
        
        Phase 2.1: Column compression strategy
        - SNAPPY for general fields
        - DICTIONARY encoding for categorical fields
        - RLE for status/boolean fields
        
        Args:
            df: DataFrame to create schema for
            partition_cols: Columns used for partitioning
            
        Returns:
            PyArrow schema
        """
        fields = []
        
        for col in df.columns:
            if col in partition_cols:
                continue  # Skip partition columns from main schema
            
            dtype = df[col].dtype
            
            # Map pandas dtypes to PyArrow types with Korean text support (UTF-8)
            if pd.api.types.is_integer_dtype(dtype):
                pa_type = pa.int64()
            elif pd.api.types.is_float_dtype(dtype):
                pa_type = pa.float64()
            elif pd.api.types.is_bool_dtype(dtype):
                pa_type = pa.bool_()
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                pa_type = pa.timestamp('ns')
            elif pd.api.types.is_categorical_dtype(dtype):
                # Use dictionary encoding for categorical
                pa_type = pa.dictionary(pa.int32(), pa.string())
            else:
                # String type with UTF-8 encoding (supports Korean)
                pa_type = pa.string()
            
            fields.append(pa.field(col, pa_type))
        
        return pa.schema(fields)
    
    def load_partitioned(
        self,
        filters: Optional[List[Tuple]] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data from partitioned Parquet dataset with optimized filtering
        
        Phase 2.3: Query Optimization with filter predicates
        
        Args:
            filters: List of tuples for partition filtering
                     e.g., [('claim_type', '=', 'Medical'), ('claim_date_month', '=', '2024-01')]
            columns: Specific columns to load (None loads all)
            
        Returns:
            DataFrame with loaded data
        """
        try:
            # Read partitioned dataset
            table = pq.read_table(
                str(self.partitioned_dir),
                filters=filters,
                columns=columns
            )
            df = table.to_pandas()
            
            # Remove partition columns if they were added
            if 'claim_date_month' in df.columns and 'claim_date' in df.columns:
                df = df.drop(columns=['claim_date_month'])
            
            return df
        except Exception as e:
            print(f"Warning: Could not load partitioned data: {str(e)}")
            return pd.DataFrame()
    
    def load_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load data from Parquet file(s)
        
        Args:
            filename: Specific file to load (None loads all parquet files)
            
        Returns:
            DataFrame containing the data
        """
        if filename:
            file_path = self.data_dir / filename
            if not file_path.suffix:
                file_path = file_path.with_suffix('.parquet')
            return pd.read_parquet(file_path)
        else:
            # Load all parquet files and concatenate
            parquet_files = list(self.data_dir.glob("*.parquet"))
            if not parquet_files:
                return pd.DataFrame()
            
            dfs = [pd.read_parquet(f) for f in parquet_files]
            return pd.concat(dfs, ignore_index=True)
    
    def query_data(
        self,
        filename: str = None,
        filters: Optional[Dict] = None,
        date_range: Optional[tuple] = None,
        date_column: str = "claim_date"
    ) -> pd.DataFrame:
        """
        Query data with filters
        
        Args:
            filename: Specific file to query
            filters: Dictionary of column:value pairs for filtering
            date_range: Tuple of (start_date, end_date) for date filtering
            date_column: Name of the date column to filter on
            
        Returns:
            Filtered DataFrame
        """
        df = self.load_data(filename)
        
        if df.empty:
            return df
        
        # Apply filters
        if filters:
            for column, value in filters.items():
                if column in df.columns:
                    if isinstance(value, list):
                        df = df[df[column].isin(value)]
                    else:
                        df = df[df[column] == value]
        
        # Apply date range filter
        if date_range and date_column in df.columns:
            start_date, end_date = date_range
            df[date_column] = pd.to_datetime(df[date_column])
            df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
        
        return df
    
    def get_statistics(self, filename: str = None) -> Dict:
        """
        Get summary statistics for the data
        
        Args:
            filename: Specific file to analyze
            
        Returns:
            Dictionary containing statistics
        """
        df = self.load_data(filename)
        
        if df.empty:
            return {}
        
        stats = {
            "total_records": len(df),
            "columns": list(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        # Categorical column statistics
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            stats["categorical_counts"] = {
                col: df[col].value_counts().to_dict()
                for col in object_cols[:5]  # Limit to first 5 categorical columns
            }
        
        return stats
    
    def save_metadata(self, metadata: Dict, filename: str = "metadata.json"):
        """
        Save metadata about the dataset
        
        Phase 2.2: Metadata storage
        
        Args:
            metadata: Dictionary containing metadata information
            filename: Name of metadata file
        """
        import json
        
        metadata_path = self.data_dir / filename
        
        # Add timestamp
        metadata['last_updated'] = datetime.now().isoformat()
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved metadata to {metadata_path}")
    
    def load_metadata(self, filename: str = "metadata.json") -> Dict:
        """
        Load metadata about the dataset
        
        Returns:
            Dictionary containing metadata
        """
        import json
        
        metadata_path = self.data_dir / filename
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def incremental_update(
        self,
        new_df: pd.DataFrame,
        key_column: str = 'claim_id',
        mode: str = 'append'
    ) -> str:
        """
        Perform incremental update to partitioned dataset
        
        Phase 2.2: Implement incremental updates
        
        Args:
            new_df: New data to add/update
            key_column: Column to use for identifying duplicates
            mode: Update mode ('append', 'upsert', 'replace')
            
        Returns:
            Path to updated dataset
        """
        if mode == 'append':
            # Simply append new data
            return self.save_partitioned(new_df)
        
        elif mode == 'upsert':
            # Load existing data
            existing_df = self.load_partitioned()
            
            if existing_df.empty:
                return self.save_partitioned(new_df)
            
            # Merge with new data (keep latest)
            if key_column in existing_df.columns and key_column in new_df.columns:
                # Remove duplicates from existing, keeping new data
                existing_df = existing_df[~existing_df[key_column].isin(new_df[key_column])]
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            return self.save_partitioned(combined_df)
        
        elif mode == 'replace':
            # Replace entire dataset
            return self.save_partitioned(new_df)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'append', 'upsert', or 'replace'")
    
    def get_partition_info(self) -> Dict:
        """
        Get information about partitions
        
        Phase 2.3: Query optimization metadata
        
        Returns:
            Dictionary with partition statistics
        """
        import os
        
        partition_info = {
            'total_partitions': 0,
            'total_size_mb': 0,
            'partitions': []
        }
        
        if not self.partitioned_dir.exists():
            return partition_info
        
        # Walk through partition structure
        for root, dirs, files in os.walk(self.partitioned_dir):
            for file in files:
                if file.endswith('.parquet'):
                    file_path = Path(root) / file
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    
                    # Extract partition info from path
                    rel_path = file_path.relative_to(self.partitioned_dir)
                    partition_keys = str(rel_path.parent).split(os.sep) if rel_path.parent != Path('.') else []
                    
                    partition_info['partitions'].append({
                        'path': str(rel_path),
                        'size_mb': round(size_mb, 2),
                        'partition_keys': partition_keys
                    })
                    
                    partition_info['total_partitions'] += 1
                    partition_info['total_size_mb'] += size_mb
        
        partition_info['total_size_mb'] = round(partition_info['total_size_mb'], 2)
        
        return partition_info
    
    def list_files(self) -> List[str]:
        """
        List all available Parquet files
        
        Returns:
            List of file names
        """
        return [f.name for f in self.data_dir.glob("*.parquet")]
