import pandas as pd
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
import sys
from pathlib import Path

sys.path.insert(0, Path(str(__file__)).parent.parent)

from utils import Logger, Timer

class UnivariateAnalysis:
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config
        self.alpha = config['statistical_tests']['alpha']
        self.confidence = config['statistical_tests']['confidence_level']
    
    @Timer.measure
    def run_analysis(self, df):
        """Execute univariate analysis"""
        results = {}
        
        self.logger.info("Starting univariate analysis...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results['normality_tests'] = self._test_normality(df, numeric_cols)
        results['confidence_intervals'] = self._calculate_ci(df, numeric_cols)
        results['descriptive_stats'] = self._descriptive_stats(df, numeric_cols)
        
        self.logger.info("Univariate analysis completed")
        return results
    
    def _test_normality(self, df, numeric_cols):
        """Shapiro-Wilk normality test"""
        try:
            sample_size = self.config['statistical_tests']['normality_sample_size']
            
            def test_col(col):
                try:
                    series = df[col].dropna()
                    if len(series) < 3:
                        return {col: {'error': 'Insufficient data'}}
                    
                    stat, p_val = stats.shapiro(
                        series.sample(min(sample_size, len(series)), random_state=42)
                    )
                    
                    return {
                        col: {
                            'statistic': round(float(stat), 4),
                            'p_value': round(float(p_val), 6),
                            'is_normal': p_val >= self.alpha
                        }
                    }
                except Exception as e:
                    return {col: {'error': str(e)}}
            
            results = Parallel(
                n_jobs=self.config['performance']['n_jobs'],
                backend=self.config['performance']['backend']
            )(delayed(test_col)(col) for col in numeric_cols)
            
            combined = {}
            for r in results:
                combined.update(r)
            
            self.logger.info(f"Normality tests completed for {len(numeric_cols)} columns")
            return combined
        except Exception as e:
            self.logger.error(f"Error in normality tests: {e}")
            return {}
    
    def _calculate_ci(self, df, numeric_cols):
        """Calculate confidence intervals"""
        try:
            def calc_ci(col):
                try:
                    clean = df[col].dropna()
                    if len(clean) < 3:
                        return {col: {'error': 'Insufficient data'}}
                    
                    mean = float(np.mean(clean))
                    std_err = float(stats.sem(clean))
                    margin = std_err * stats.t.ppf(
                        (1 + self.confidence) / 2, len(clean) - 1
                    )
                    
                    return {
                        col: {
                            'mean': round(mean, 4),
                            'lower_bound': round(mean - margin, 4),
                            'upper_bound': round(mean + margin, 4),
                            'margin_of_error': round(margin, 4),
                            'confidence': self.confidence
                        }
                    }
                except Exception as e:
                    return {col: {'error': str(e)}}
            
            results = Parallel(
                n_jobs=self.config['performance']['n_jobs'],
                backend=self.config['performance']['backend']
            )(delayed(calc_ci)(col) for col in numeric_cols)
            
            combined = {}
            for r in results:
                combined.update(r)
            
            self.logger.info(f"Confidence intervals calculated for {len(numeric_cols)} columns")
            return combined
        except Exception as e:
            self.logger.error(f"Error calculating CI: {e}")
            return {}
    
    def _descriptive_stats(self, df, numeric_cols):
        """Calculate descriptive statistics"""
        try:
            stats_df = df[numeric_cols].describe().T
            stats_dict = stats_df.round(4).to_dict('index')
            self.logger.info(f"Descriptive statistics calculated for {len(numeric_cols)} columns")
            return stats_dict
        except Exception as e:
            self.logger.error(f"Error calculating descriptive stats: {e}")
            return {}
