import pandas as pd
from pathlib import Path
import json
from utils.logger import Logger
from utils.file_utils import IOHandler
from utils.timer import Timer

class ReportGenerator:
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config
        self.io_handler = IOHandler(config['output']['results_dir'])
    
    @Timer.measure

    
    def _save_csv_reports(self, quality, univariate, bivariate):
        """Save results as CSV files"""
        try:
            # Missing values
            if quality['missing_values']['count'] > 0:
                missing_df = pd.DataFrame(quality['missing_values']['details']).T
                self.io_handler.save_csv(missing_df, 'missing_values')
            
            # Outliers
            outliers_df = pd.DataFrame(quality['outliers']).T
            self.io_handler.save_csv(outliers_df, 'outliers')
            
            # Normality tests
            normality_df = pd.DataFrame(univariate['normality_tests']).T
            self.io_handler.save_csv(normality_df, 'normality_tests')
            
            # Confidence intervals
            ci_df = pd.DataFrame(univariate['confidence_intervals']).T
            self.io_handler.save_csv(ci_df, 'confidence_intervals')
            
            self.logger.info("CSV reports saved")
        except Exception as e:
            self.logger.error(f"Error saving CSV reports: {e}")
    
    def _save_sqlite_reports(self, quality, univariate, bivariate):
        """Save results to SQLite"""
        try:
            # Missing values
            if quality['missing_values']['count'] > 0:
                missing_df = pd.DataFrame(quality['missing_values']['details']).T.reset_index()
                self.io_handler.save_to_sqlite(missing_df, 'missing_values')
            
            # Outliers
            outliers_df = pd.DataFrame(quality['outliers']).T.reset_index()
            self.io_handler.save_to_sqlite(outliers_df, 'outliers')
            
            # Normality tests
            normality_df = pd.DataFrame(univariate['normality_tests']).T.reset_index()
            self.io_handler.save_to_sqlite(normality_df, 'normality_tests')
            
            self.logger.info("SQLite reports saved")
        except Exception as e:
            self.logger.error(f"Error saving SQLite reports: {e}")