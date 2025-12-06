import pandas as pd
from utils.logger import Logger
from utils.timer import Timer

class DuplicateHandler:
    """Handle duplicate rows"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['duplicates']
    
    @Timer.measure
    def handle_duplicates(self, df, fit=True):
        """Detect and remove exact duplicates"""
        try:
            self.logger.info(f"Checking for duplicates - Before: {len(df)} rows")
            
            if self.config['check_duplicates']:
                duplicates_count = df.duplicated().sum()
                self.logger.info(f"Exact duplicates found: {duplicates_count}")
                
                if duplicates_count > 0:
                    df = df.drop_duplicates()
                    self.logger.info(f"Duplicates removed - After: {len(df)} rows")
                else:
                    self.logger.info("No duplicates detected")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error handling duplicates: {e}")
            raise