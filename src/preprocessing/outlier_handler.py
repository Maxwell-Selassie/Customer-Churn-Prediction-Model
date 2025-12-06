import pandas as pd
import numpy as np
from utils.logger import Logger
from utils.timer import Timer

class OutlierHandler:
    """Flag outliers without removing them"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['outliers']
        self.outlier_bounds = {}  # Store bounds from training set
    
    @Timer.measure
    def handle_outliers(self, df, fit=True):
        """
        Flag outliers using IQR method
        Compute bounds on training set, apply to all sets
        """
        try:
            self.logger.info(f"Processing outliers...")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if fit:
                self.logger.info("Computing outlier bounds from training data...")
                self.outlier_bounds = self._compute_bounds(df, numeric_cols)
            
            df = self._flag_outliers(df, numeric_cols)
            
            outlier_count = (df['is_outlier'] == 1).sum()
            self.logger.info(f"Rows flagged as outliers: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error handling outliers: {e}")
            raise
    
    def _compute_bounds(self, df, numeric_cols):
        """Compute IQR bounds for each numeric column"""
        bounds = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.config['multiplier'] * IQR
            upper_bound = Q3 + self.config['multiplier'] * IQR
            
            bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
        
        self.logger.debug(f"Computed outlier bounds for {len(numeric_cols)} columns")
        return bounds
    
    def _flag_outliers(self, df, numeric_cols):
        """Flag rows as outliers based on training bounds"""
        is_outlier = pd.Series(0, index=df.index)
        
        for col in numeric_cols:
            if col not in self.outlier_bounds:
                continue
            
            lower = self.outlier_bounds[col]['lower']
            upper = self.outlier_bounds[col]['upper']
            
            col_outliers = (df[col] < lower) | (df[col] > upper)
            is_outlier = is_outlier | col_outliers.astype(int)
        
        df['is_outlier'] = is_outlier
        return df