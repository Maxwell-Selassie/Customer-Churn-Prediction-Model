import pandas as pd
from utils.logger import Logger
from utils.timer import Timer

class FeatureEncoder:
    """Encode categorical features"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['encoding']
        self.encoding_cache = {}  # Cache frequency/target mappings
    
    @Timer.measure
    def encode_features(self, df, fit=True):
        """
        Apply encoding strategies:
        - One-hot for low cardinality
        - Frequency for high cardinality
        """
        try:
            self.logger.info("Encoding categorical features...")
            
            # One-hot encoding
            df = self._one_hot_encode(df, fit)
            
            # Frequency encoding
            df = self._frequency_encode(df, fit)
            
            self.logger.info("Feature encoding completed")
            return df
        
        except Exception as e:
            self.logger.error(f"Error encoding features: {e}")
            raise
    
    def _one_hot_encode(self, df, fit=True):
        """One-hot encode low cardinality features"""
        try:
            columns = self.config['one_hot_columns']
            
            for col in columns:
                if col not in df.columns:
                    self.logger.warning(f"Column {col} not found for one-hot encoding")
                    continue
                
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                
                self.logger.debug(f"One-hot encoded {col} into {len(dummies.columns)} features")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error in one-hot encoding: {e}")
            raise
    
    def _frequency_encode(self, df, fit=True):
        """Frequency encode high cardinality features"""
        try:
            columns = self.config['frequency_columns']
            
            for col in columns:
                if col not in df.columns:
                    self.logger.warning(f"Column {col} not found for frequency encoding")
                    continue
                
                if fit:
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    self.encoding_cache[f"{col}_freq"] = freq_map
                
                freq_map = self.encoding_cache.get(f"{col}_freq", {})
                feature_name = f"{col}_frequency"
                df[feature_name] = df[col].map(freq_map)
                
                self.logger.debug(f"Frequency encoded {col}")
                df = df.drop(columns=[col])
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error in frequency encoding: {e}")
            raise
