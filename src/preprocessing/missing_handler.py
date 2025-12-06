
import pandas as pd
import numpy as np
from utils.logger import Logger
from utils.timer import Timer

class MissingHandler:
    """Handle missing values"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['missing_values']
        self.drop_columns = self.config['drop_columns']
    
    @Timer.measure
    def handle_missing(self, df, fit=True):
        """
        Handle missing values by dropping rows with null Customer ID
        This takes care of all other missing values as per EDA finding
        """
        try:
            self.logger.info(f"Handling missing values - Before: {len(df)} rows")
            
            initial_rows = len(df)
            
            # Drop rows with Customer ID null
            for col in self.drop_columns:
                if col in df.columns:
                    df = df.dropna(subset=[col])
                    self.logger.info(f"Dropped rows with null {col}: {initial_rows - len(df)} rows removed")
            
            self.logger.info(f"Handling missing values - After: {len(df)} rows")
            
            missing_summary = df.isnull().sum()
            if missing_summary.sum() > 0:
                self.logger.warning(f"Remaining missing values:\n{missing_summary[missing_summary > 0]}")
            else:
                self.logger.info("No missing values remaining")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error handling missing values: {e}")
            raise
