import pandas as pd
from utils.logger import Logger
from utils.timer import Timer

class FeatureEngineer:
    """Create derived features via aggregations"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['feature_engineering']
        self.aggregation_cache = {}  # Cache computed aggregations from training
    
    @Timer.measure
    def engineer_features(self, df, fit=True):
        """
        Create aggregation features
        Compute on training set, lookup on dev/test to prevent leakage
        """
        try:
            self.logger.info("Feature engineering...")
            
            for agg in self.config['aggregations']:
                if agg['type'] == 'groupby':
                    df = self._groupby_aggregation(df, agg, fit)
                elif agg['type'] == 'count':
                    df = self._count_aggregation(df, agg, fit)
            
            self.logger.info("Feature engineering completed")
            return df
        
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            raise
    
    def _groupby_aggregation(self, df, agg_config, fit=True):
        """Create groupby aggregation features"""
        try:
            col = agg_config['column']
            agg_col = agg_config['agg_col']
            operations = agg_config['operations']
            
            cache_key = f"{col}_{agg_col}"
            
            if fit:
                # Compute aggregations from training data
                agg_dict = {}
                for op in operations:
                    feature_name = f"{op}_{col}_{agg_col}"
                    agg_dict[op] = (agg_col, op)
                    
                    result = df.groupby(col)[agg_col].agg(op).to_dict()
                    self.aggregation_cache[f"{cache_key}_{op}"] = result
                    
                    df[feature_name] = df[col].map(result)
                    self.logger.debug(f"Created feature: {feature_name}")
            else:
                # Apply cached aggregations to dev/test
                for op in operations:
                    feature_name = f"{op}_{col}_{agg_col}"
                    cache_key_op = f"{cache_key}_{op}"
                    
                    if cache_key_op in self.aggregation_cache:
                        df[feature_name] = df[col].map(self.aggregation_cache[cache_key_op])
                    else:
                        self.logger.warning(f"Cache miss for {cache_key_op}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error in groupby aggregation: {e}")
            raise
    
    def _count_aggregation(self, df, agg_config, fit=True):
        """Create count aggregation features"""
        try:
            col = agg_config['column']
            agg_col = agg_config['agg_col']
            feature_name = f"{col}_count_{agg_col}"
            
            cache_key = feature_name
            
            if fit:
                result = df.groupby(col).size().to_dict()
                self.aggregation_cache[cache_key] = result
                df[feature_name] = df[col].map(result)
                self.logger.debug(f"Created feature: {feature_name}")
            else:
                if cache_key in self.aggregation_cache:
                    df[feature_name] = df[col].map(self.aggregation_cache[cache_key])
                else:
                    self.logger.warning(f"Cache miss for {cache_key}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error in count aggregation: {e}")
            raise

