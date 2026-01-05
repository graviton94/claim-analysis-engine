"""
ETL Module for Claim Data Intelligence System
Handles ingestion and standardization of CSV/XLSX/JSON data into Parquet format
"""

import pandas as pd
import json
import os
from typing import Union, List, Dict, Tuple
from pathlib import Path
from datetime import datetime


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
        self.standard_schema = None
        self.field_mappings = None
    
    def read_input_header(self, header_file: str = "input_header.csv") -> List[str]:
        """
        Read and parse the 80-field header from input_header.csv
        
        Args:
            header_file: Path to the header CSV file
            
        Returns:
            List of 80 field names
        """
        try:
            with open(header_file, 'r') as f:
                header_line = f.readline().strip()
                fields = [field.strip() for field in header_line.split(',')]
            
            print(f"✓ Loaded {len(fields)} fields from {header_file}")
            return fields
        except Exception as e:
            raise ValueError(f"Error reading input header: {str(e)}")
    
    def create_standard_schema(self, fields: List[str] = None) -> Dict[str, Dict]:
        """
        Convert 80-field CSV header to standardized schema with type definitions
        
        Args:
            fields: List of field names (if None, reads from input_header.csv)
            
        Returns:
            Dictionary mapping field names to their types and processing rules
        """
        if fields is None:
            fields = self.read_input_header()
        
        # Define schema with field types and processing rules
        schema = {}
        
        # Categorize fields by type
        date_fields = [f for f in fields if 'date' in f.lower()]
        amount_fields = [f for f in fields if 'amount' in f.lower() or 'premium' in f.lower() 
                        or 'value' in f.lower() or 'deductible' in f.lower() or 'limit' in f.lower()]
        id_fields = [f for f in fields if f.endswith('_id') or f.endswith('_number')]
        text_fields = [f for f in fields if 'name' in f.lower() or 'description' in f.lower() 
                      or 'address' in f.lower() or 'notes' in f.lower()]
        code_fields = [f for f in fields if 'code' in f.lower() or 'status' in f.lower() 
                      or 'type' in f.lower() or 'category' in f.lower()]
        contact_fields = [f for f in fields if 'email' in f.lower() or 'phone' in f.lower()]
        location_fields = [f for f in fields if any(x in f.lower() for x in ['city', 'state', 'zip', 'country'])]
        score_fields = [f for f in fields if 'score' in f.lower() or 'indicator' in f.lower()]
        
        # Build schema with type definitions
        for field in fields:
            if field in date_fields:
                schema[field] = {'type': 'datetime64[ns]', 'nullable': True, 'fill_strategy': 'null'}
            elif field in amount_fields:
                schema[field] = {'type': 'float64', 'nullable': True, 'fill_strategy': 0.0}
            elif field in id_fields:
                schema[field] = {'type': 'str', 'nullable': False, 'fill_strategy': 'UNKNOWN'}
            elif field in score_fields:
                schema[field] = {'type': 'float64', 'nullable': True, 'fill_strategy': 0.0}
            elif field in contact_fields:
                schema[field] = {'type': 'str', 'nullable': True, 'fill_strategy': ''}
            elif field in location_fields:
                schema[field] = {'type': 'str', 'nullable': True, 'fill_strategy': 'Unknown'}
            elif field in code_fields:
                schema[field] = {'type': 'category', 'nullable': True, 'fill_strategy': 'Unknown'}
            elif field in text_fields:
                schema[field] = {'type': 'str', 'nullable': True, 'fill_strategy': ''}
            else:
                schema[field] = {'type': 'str', 'nullable': True, 'fill_strategy': 'Unknown'}
        
        self.standard_schema = schema
        print(f"✓ Created standard schema for {len(schema)} fields")
        print(f"  - Date fields: {len(date_fields)}")
        print(f"  - Amount fields: {len(amount_fields)}")
        print(f"  - ID fields: {len(id_fields)}")
        print(f"  - Code/Category fields: {len(code_fields)}")
        
        return schema
    
    def apply_standard_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the standard schema to a DataFrame with 80 fields
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized types and filled missing values
        """
        if self.standard_schema is None:
            # Auto-detect from DataFrame columns
            self.create_standard_schema(list(df.columns))
        
        df = df.copy()
        
        # Apply type conversions and fill strategies
        for field, spec in self.standard_schema.items():
            if field not in df.columns:
                continue
                
            field_type = spec['type']
            fill_value = spec['fill_strategy']
            
            try:
                if field_type == 'datetime64[ns]':
                    df[field] = pd.to_datetime(df[field], errors='coerce')
                elif field_type == 'float64':
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                    if fill_value is not None:
                        df[field] = df[field].fillna(fill_value)
                elif field_type == 'category':
                    df[field] = df[field].astype('str').replace('nan', fill_value)
                    df[field] = df[field].astype('category')
                elif field_type == 'str':
                    df[field] = df[field].astype('str').replace('nan', fill_value)
            except Exception as e:
                print(f"Warning: Error processing field '{field}': {str(e)}")
        
        return df
    
    def validate_80_field_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has the expected 80 fields and correct types
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        messages = []
        is_valid = True
        
        # Check field count
        expected_fields = self.read_input_header()
        actual_fields = set(df.columns)
        expected_set = set(expected_fields)
        
        missing_fields = expected_set - actual_fields
        extra_fields = actual_fields - expected_set
        
        if len(missing_fields) > 0:
            is_valid = False
            messages.append(f"Missing {len(missing_fields)} fields: {list(missing_fields)[:5]}...")
        
        if len(extra_fields) > 0:
            messages.append(f"Extra {len(extra_fields)} fields: {list(extra_fields)[:5]}...")
        
        if len(df) == 0:
            is_valid = False
            messages.append("DataFrame is empty")
        
        # Type validation
        if self.standard_schema:
            date_cols = [f for f, s in self.standard_schema.items() if s['type'] == 'datetime64[ns]' and f in df.columns]
            invalid_dates = sum(1 for col in date_cols if not pd.api.types.is_datetime64_any_dtype(df[col]))
            if invalid_dates > 0:
                messages.append(f"{invalid_dates} date columns have incorrect types")
        
        if is_valid:
            messages.append(f"✓ Validation passed: {len(df)} records with {len(df.columns)} fields")
        
        return is_valid, messages
        
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
