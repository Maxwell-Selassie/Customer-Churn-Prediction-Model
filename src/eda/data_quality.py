
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, Path(str(__file__)).parent.parent)

from utils import Logger, Timer


class DataQuality:
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['quality_checks']
    
    @Timer.measure
    def run_quality_checks(self, df):
        """Execute all quality checks"""
        results = {}
        
        self.logger.info("Starting data quality checks...")
        results['shape'] = {'observations': len(df), 'features': len(df.columns)}
        results['missing_values'] = self._check_missing_values(df)
        results['duplicates'] = self._check_duplicates(df)
        results['outliers'] = self._check_outliers(df)
        results['data_types'] = self._get_data_types(df)
        
        self.logger.info("Data quality checks completed")
        return results
    
    def _check_missing_values(self, df):
        """Check for missing values"""
        try:
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            
            if len(missing) == 0:
                self.logger.info("No missing values detected")
                return {'count': 0, 'details': {}}
            
            missing_pct = (missing / len(df)) * 100
            max_missing = self.config['max_missing_pct']
            
            summary = pd.DataFrame({
                'missing': missing,
                'missing_pct': missing_pct.round(2)
            })
            
            problematic = summary[summary['missing_pct'] > max_missing]
            if len(problematic) > 0:
                self.logger.warning(f"Columns with >{max_missing}% missing: {problematic.index.tolist()}")
            
            return {
                'count': len(missing),
                'details': summary.to_dict('index')
            }
        except Exception as e:
            self.logger.error(f"Error checking missing values: {e}")
            return {'count': 0, 'details': {}, 'error': str(e)}
    
    def _check_duplicates(self, df):
        """Check for duplicate rows"""
        try:
            duplicates = df.duplicated().sum()
            self.logger.info(f"Duplicate rows detected: {duplicates}")
            return {'count': duplicates}
        except Exception as e:
            self.logger.error(f"Error checking duplicates: {e}")
            return {'count': 0, 'error': str(e)}
    
    def _check_outliers(self, df):
        """Detect outliers using IQR method"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            outlier_summary = {}
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                outlier_summary[col] = {
                    'count': len(outliers),
                    'pct': round((len(outliers) / len(df)) * 100, 2),
                    'range': f"({round(lower_bound, 2)} - {round(upper_bound, 2)})"
                }
            
            return outlier_summary
        except Exception as e:
            self.logger.error(f"Error checking outliers: {e}")
            return {}
    
    def _get_data_types(self, df):
        """Get data type summary"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            numeric_summary = {}
            for col in numeric_cols:
                numeric_summary[col] = {
                    'type': str(df[col].dtype),
                    'min': round(float(df[col].min()), 4) if pd.notna(df[col].min()) else None,
                    'max': round(float(df[col].max()), 4) if pd.notna(df[col].max()) else None
                }
            
            categorical_summary = {}
            for col in categorical_cols:
                categorical_summary[col] = {
                    'type': str(df[col].dtype),
                    'unique': df[col].nunique(),
                    'examples': df[col].unique()[:3].tolist()
                }
            
            return {
                'numeric': numeric_summary,
                'categorical': categorical_summary
            }
        except Exception as e:
            self.logger.error(f"Error getting data types: {e}")
            return {}