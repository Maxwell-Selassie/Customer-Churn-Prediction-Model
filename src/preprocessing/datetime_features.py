import pandas as pd
import numpy as np
from utils.logger import Logger
from utils.timer import Timer

class DatetimeFeatureExtractor:
    """Extract datetime features with cyclical encoding"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['datetime']
    
    @Timer.measure
    def extract_features(self, df, fit=True):
        """Extract datetime features with cyclical encoding"""
        try:
            self.logger.info("Extracting datetime features...")
            
            datetime_cols = self.config['datetime_columns']
            
            for col in datetime_cols:
                if col not in df.columns:
                    self.logger.warning(f"Column {col} not found")
                    continue
                
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Basic temporal components
                df[f'Year_{col}'] = df[col].dt.year
                df[f'Month_{col}'] = df[col].dt.month
                df[f'Day_{col}'] = df[col].dt.day
                df[f'Quarter_{col}'] = df[col].dt.quarter
                df[f'Hour_{col}'] = df[col].dt.hour
                df[f'Minute_{col}'] = df[col].dt.minute
                df[f'Seconds_{col}'] = df[col].dt.second
                df[f'DayOfWeek_{col}'] = df[col].dt.dayofweek
                df[f'WeekOfYear_{col}'] = df[col].dt.isocalendar().week
                
                # Binary features
                df[f'Is_weekend_{col}'] = (df[col].dt.dayofweek > 4).astype(int)
                df[f'Is_night_{col}'] = (df[col].dt.hour > 17).astype(int)
                
                # Cyclical encoding
                if self.config['cyclical_encoding']:
                    df = self._add_cyclical_features(df, col)
            
            self.logger.info("Datetime feature extraction completed")
            return df
        
        except Exception as e:
            self.logger.error(f"Error extracting datetime features: {e}")
            raise
    
    def _add_cyclical_features(self, df, col):
        """Add sin/cos cyclical encoding for circular features"""
        cyclical_config = self.config['cyclical_columns']
        
        # Month cyclical (12 months)
        df[f'Month_{col}_sin'] = np.sin(2 * np.pi * df[f'Month_{col}'] / cyclical_config['month'])
        df[f'Month_{col}_cos'] = np.cos(2 * np.pi * df[f'Month_{col}'] / cyclical_config['month'])
        
        # Day of week cyclical (7 days)
        df[f'DayOfWeek_{col}_sin'] = np.sin(2 * np.pi * df[f'DayOfWeek_{col}'] / cyclical_config['day_of_week'])
        df[f'DayOfWeek_{col}_cos'] = np.cos(2 * np.pi * df[f'DayOfWeek_{col}'] / cyclical_config['day_of_week'])
        
        # Hour cyclical (24 hours)
        df[f'Hour_{col}_sin'] = np.sin(2 * np.pi * df[f'Hour_{col}'] / cyclical_config['hour'])
        df[f'Hour_{col}_cos'] = np.cos(2 * np.pi * df[f'Hour_{col}'] / cyclical_config['hour'])
        
        return df