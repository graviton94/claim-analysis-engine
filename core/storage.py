"""
Storage Module for Claim Data Intelligence System
Handles Parquet database operations and queries
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import glob


class ParquetStorage:
    """High-performance Parquet database storage manager"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize storage manager
        
        Args:
            data_dir: Directory containing Parquet files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def list_files(self) -> List[str]:
        """
        List all available Parquet files
        
        Returns:
            List of file names
        """
        return [f.name for f in self.data_dir.glob("*.parquet")]
