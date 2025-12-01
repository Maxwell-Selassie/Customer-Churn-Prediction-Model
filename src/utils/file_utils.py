
"""
File Handler Module for Data I/O Operations
Handles reading, writing, and data operations with validation
"""

import pandas as pd
import json
import sqlite3
from pathlib import Path
from typing import Union, Dict, Any, Optional
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Unified file I/O handler with validation and error handling"""
    
    SUPPORTED_FORMATS = {'.csv', '.json', '.parquet', '.xlsx'}
    
    @staticmethod
    def validate_file_path(file_path: str) -> Path:
        """Validate and return Path object"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            return path
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            raise
    
    @staticmethod
    def read_csv(
        file_path: str,
        encoding: str = 'utf-8',
        dtype_spec: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Read CSV file with validation and error handling"""
        try:
            path = FileHandler.validate_file_path(file_path)
            logger.info(f"Reading CSV: {file_path}")
            
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                dtype=dtype_spec,
                **kwargs
            )
            
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")
            
            logger.info(
                f"Successfully loaded {len(df)} rows and {len(df.columns)} columns"
            )
            return df
            
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty")
            raise
        except UnicodeDecodeError:
            logger.error(f"Encoding error with {encoding}. Trying utf-8-sig...")
            return FileHandler.read_csv(
                file_path, encoding='utf-8-sig', dtype_spec=dtype_spec, **kwargs
            )
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise
    
    @staticmethod
    def save_csv(
        df: pd.DataFrame,
        output_path: str,
        index: bool = False,
        **kwargs
    ) -> str:
        """Save DataFrame to CSV with error handling"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=index, **kwargs)
            logger.info(f"CSV saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            raise
    
    @staticmethod
    def save_json(
        data: Union[Dict, list],
        output_path: str,
        indent: int = 2,
        **kwargs
    ) -> str:
        """Save data to JSON with error handling"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Handle numpy types for JSON serialization
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    try:
                        import numpy as np
                        if isinstance(obj, (np.integer, np.floating)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                    except:
                        pass
                    return super().default(obj)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=indent, cls=NumpyEncoder, **kwargs)
            
            logger.info(f"JSON saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            raise
    
    @staticmethod
    def save_to_sqlite(
        data: Dict[str, pd.DataFrame],
        db_path: str,
        if_exists: str = 'append'
    ) -> str:
        """Save multiple DataFrames to SQLite database"""
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            for table_name, df in data.items():
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            conn.close()
            
            logger.info(f"SQLite database saved: {db_path}")
            return db_path
        except Exception as e:
            logger.error(f"Error saving to SQLite: {e}")
            raise


class DataOperations:
    """Generic data operations with validation"""
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        try:
            return {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': len(df[df.duplicated()]),
            }
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            raise
    
    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> list:
        """Get numeric columns safely"""
        try:
            return df.select_dtypes(include=['number']).columns.tolist()
        except Exception as e:
            logger.error(f"Error extracting numeric columns: {e}")
            raise
    
    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> list:
        """Get categorical columns safely"""
        try:
            return df.select_dtypes(exclude=['number']).columns.tolist()
        except Exception as e:
            logger.error(f"Error extracting categorical columns: {e}")
            raise
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = 10) -> bool:
        """Validate DataFrame integrity"""
        try:
            if df is None or not isinstance(df, pd.DataFrame):
                raise ValueError("Invalid DataFrame object")
            if df.empty:
                raise ValueError("DataFrame is empty")
            if len(df) < min_rows:
                raise ValueError(f"DataFrame has fewer than {min_rows} rows")
            return True
        except Exception as e:
            logger.error(f"DataFrame validation failed: {e}")
            raise
    
    @staticmethod
    def create_timestamped_filename(base_name: str, extension: str) -> str:
        """Generate timestamped filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}{extension}"