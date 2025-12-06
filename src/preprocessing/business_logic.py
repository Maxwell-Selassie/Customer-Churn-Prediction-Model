import pandas as pd
import numpy as np
from utils.logger import Logger
from utils.timer import Timer

class BusinessLogicHandler:
    """Handle business logic errors"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['business_logic']
    
    @Timer.measure
    def handle_business_logic(self, df, fit=True):
        """
        Handle business logic errors:
        Drop rows where Quantity != 0 but Price or Revenue = 0
        """
        try:
            self.logger.info(f"Handling business logic errors - Before: {len(df)} rows")
            
            if self.config['drop_zero_outliers']:
                quantity_col = self.config['zero_logic_columns']['quantity']
                price_col = self.config['zero_logic_columns']['price']
                revenue_col = self.config['zero_logic_columns']['revenue']
                
                # Condition: Quantity != 0 but Price or Revenue = 0
                invalid_rows = (
                    (df[quantity_col] != 0) & 
                    ((df[price_col] == 0) | (df[revenue_col] == 0))
                )
                
                dropped = invalid_rows.sum()
                df = df[~invalid_rows]
                
                self.logger.info(f"Dropped {dropped} rows with business logic errors")
            
            self.logger.info(f"Handling business logic errors - After: {len(df)} rows")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error handling business logic: {e}")
            raise