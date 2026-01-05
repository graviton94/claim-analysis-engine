"""
ETL Module for Claim Data Intelligence System
Handles ingestion and standardization of CSV/XLSX/JSON data into Parquet format
"""

import pandas as pd
import json
import os
from typing import Union, List
from pathlib import Path


class ETLProcessor:
    """Extract, Transform, Load processor for claim data"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize ETL processor
        
        Args:
            data_dir: Directory where Parquet files will be stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        Read data from CSV, XLSX, or JSON file
        
        Args:
            file_path: Path to the input file
            
        Returns:
            DataFrame containing the data
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize claim data schema
        
        Expected columns:
        - claim_id: Unique claim identifier
        - claim_date: Date of claim
        - claim_amount: Amount of claim
        - claim_type: Type/category of claim
        - status: Claim status (pending, approved, rejected)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Standardized DataFrame
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Convert date columns to datetime
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure numeric columns are numeric
        amount_columns = [col for col in df.columns if 'amount' in col.lower()]
        for col in amount_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df = df.fillna({
            col: 0 if df[col].dtype in ['float64', 'int64'] else 'Unknown'
            for col in df.columns
        })
        
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, name: str) -> str:
        """
        Save DataFrame to Parquet format
        
        Args:
            df: DataFrame to save
            name: Name of the parquet file (without extension)
            
        Returns:
            Path to saved file
        """
        output_path = self.data_dir / f"{name}.parquet"
        df.to_parquet(output_path, index=False, compression='snappy')
        return str(output_path)
    
    def process_file(self, file_path: str, name: str = None) -> str:
        """
        Complete ETL pipeline: read, standardize, and save to Parquet
        
        Args:
            file_path: Path to input file
            name: Optional name for output file (defaults to input filename)
            
        Returns:
            Path to saved Parquet file
        """
        if name is None:
            name = Path(file_path).stem
        
        # Extract
        df = self.read_file(file_path)
        
        # Transform
        df = self.standardize_schema(df)
        
        # Load
        output_path = self.save_to_parquet(df, name)
        
        return output_path
    
    def process_multiple_files(self, file_paths: List[str], output_name: str = "claims_consolidated") -> str:
        """
        Process multiple files and consolidate into single Parquet file
        
        Args:
            file_paths: List of input file paths
            output_name: Name for consolidated output file
            
        Returns:
            Path to consolidated Parquet file
        """
        dfs = []
        for file_path in file_paths:
            df = self.read_file(file_path)
            df = self.standardize_schema(df)
            dfs.append(df)
        
        # Consolidate all dataframes
        consolidated_df = pd.concat(dfs, ignore_index=True)
        
        # Save consolidated data
        output_path = self.save_to_parquet(consolidated_df, output_name)
        
        return output_path
