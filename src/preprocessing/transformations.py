import pandas as pd
import numpy as np
from utils.logger import Logger
from utils.timer import Timer

class FeatureTransformer:
    """Apply mathematical transformations"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['transformations']
    
    @Timer.measure
    def transform_features(self, df, fit=True):
        """Apply log transformation to reduce skewness"""
        try:
            self.logger.info("Applying feature transformations...")
            
            log_cols = self.config['log_columns']
            
            for col in log_cols:
                if col not in df.columns:
                    self.logger.warning(f"Column {col} not found for log transformation")
                    continue
                
                df[f'{col}_log'] = np.log1p(df[col])
                self.logger.debug(f"Log transformed {col}")
            
            self.logger.info("Feature transformations completed")
            return df
        
        except Exception as e:
            self.logger.error(f"Error in feature transformation: {e}")
            raise

