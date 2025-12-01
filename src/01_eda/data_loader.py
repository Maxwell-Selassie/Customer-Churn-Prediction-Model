# src/core/data_loader.py
"""
Data Loader Module
Handles data ingestion, preprocessing, and type conversions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from src.utils.logger import get_logger
from utils.file_handler import FileHandler, DataOperations
from utils.timer import ExecutionTimer

logger = get_logger(__name__)


class DataLoader:
    """Handles data loading and initial preprocessing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['data']
        self.df = None
        self.original_df = None
    
    def load_data(self) -> pd.DataFrame:
        """Load data from configured source"""
        with ExecutionTimer(name="Data Loading", verbose=True):
            file_path = self.config.get('file_path')
            encoding = self.config.get('encoding', 'utf-8')
            dtype_spec = self.config.get('dtype_spec')
            
            self.df = FileHandler.read_csv(
                file_path=file_path,
                encoding=encoding,
                dtype_spec=dtype_spec
            )
            
            # Backup original for comparison
            self.original_df = self.df.copy()
            
            logger.info(f"Data shape: {self.df.shape}")
            return self.df
    
    def auto_type_convert(self) -> pd.DataFrame:
        """Automatically convert column types with validation"""
        with ExecutionTimer(name="Type Conversion", verbose=True):
            try:
                # Convert datetime columns
                date_candidates = self.df.select_dtypes(include=['object']).columns
                for col in date_candidates:
                    if any(date_keyword in col.lower() for date_keyword in 
                           ['date', 'time', 'created', 'updated', 'login']):
                        try:
                            self.df[col] = pd.to_datetime(self.df[col])
                            logger.info(f"Converted {col} to datetime")
                        except Exception as e:
                            logger.warning(f"Could not convert {col} to datetime: {e}")
                
                # Round currency columns
                currency_keywords = ['cost', 'revenue', 'profit', 'price', 'amount', 'fee']
                for col in self.df.columns:
                    if any(kw in col.lower() for kw in currency_keywords):
                        if self.df[col].dtype in ['float64', 'float32']:
                            self.df[col] = self.df[col].round(2)
                            logger.info(f"Rounded {col} to 2 decimal places")
                
                logger.info("Type conversion completed")
                return self.df
                
            except Exception as e:
                logger.error(f"Error during type conversion: {e}")
                raise
    
    def handle_infinite_values(self) -> pd.DataFrame:
        """Replace infinite values with NaN"""
        with ExecutionTimer(name="Infinite Value Handling", verbose=True):
            try:
                numeric_cols = DataOperations.get_numeric_columns(self.df)
                
                inf_count = 0
                for col in numeric_cols:
                    inf_mask = np.isinf(self.df[col])
                    if inf_mask.any():
                        inf_count += inf_mask.sum()
                        self.df.loc[inf_mask, col] = np.nan
                        logger.warning(f"Found {inf_mask.sum()} infinite values in {col}")
                
                if inf_count > 0:
                    logger.info(f"Replaced {inf_count} infinite values with NaN")
                
                return self.df
                
            except Exception as e:
                logger.error(f"Error handling infinite values: {e}")
                raise
    
    def get_data_snapshot(self) -> Dict[str, Any]:
        """Return snapshot of loaded data"""
        try:
            return {
                'shape': self.df.shape,
                'columns': self.df.columns.tolist(),
                'dtypes': self.df.dtypes.to_dict(),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
                'head': self.df.head().to_dict('records')
            }
        except Exception as e:
            logger.error(f"Error generating data snapshot: {e}")
            raise