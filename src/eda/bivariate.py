import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency
from joblib import Parallel, delayed
import sys
from pathlib import Path
from utils.logger import Logger
from utils.timer import Timer


class BivariateAnalysis:
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config
        self.alpha = config['statistical_tests']['alpha']
    
    @Timer.measure
    def run_analysis(self, df, target_col=None):
        """Execute bivariate analysis"""
        results = {}
        
        self.logger.info("Starting bivariate analysis...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if target_col and target_col in df.columns:
            results['numeric_vs_target'] = self._numeric_tests(df, numeric_cols, target_col)
            results['categorical_vs_target'] = self._categorical_tests(df, categorical_cols, target_col)
            results['correlation'] = self._correlation_analysis(df, numeric_cols)
            results['target_distribution'] = self._target_distribution(df, target_col)
        else:
            results['correlation'] = self._correlation_analysis(df, numeric_cols)
        
        self.logger.info("Bivariate analysis completed")
        return results
    
    def _numeric_tests(self, df, numeric_cols, target_col):
        """Mann-Whitney U and t-tests"""
        try:
            if target_col not in df.columns or df[target_col].nunique() != 2:
                self.logger.warning(f"Target {target_col} not binary or missing")
                return {}
            
            def test_numeric(col):
                try:
                    group1 = df[df[target_col] == df[target_col].unique()[0]][col].dropna()
                    group2 = df[df[target_col] == df[target_col].unique()[1]][col].dropna()
                    
                    if len(group1) < 3 or len(group2) < 3:
                        return {col: {'error': 'Insufficient data'}}
                    
                    sample_size = min(5000, len(group1), len(group2))
                    _, p1 = stats.shapiro(
                        group1.sample(min(sample_size, len(group1)), random_state=42)
                    )
                    _, p2 = stats.shapiro(
                        group2.sample(min(sample_size, len(group2)), random_state=42)
                    )
                    
                    if p1 < self.alpha or p2 < self.alpha:
                        stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                        test_type = 'Mann-Whitney U'
                    else:
                        stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                        test_type = "Welch's t-test"
                    
                    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / len(df))
                    cohens_d = abs(group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    effect = 'negligible' if cohens_d < 0.2 else 'small' if cohens_d < 0.5 else 'medium' if cohens_d < 0.8 else 'large'
                    
                    return {
                        col: {
                            'test_type': test_type,
                            'group1_mean': round(float(group1.mean()), 4),
                            'group2_mean': round(float(group2.mean()), 4),
                            'p_value': round(float(p_val), 6),
                            'significant': p_val < self.alpha,
                            'cohens_d': round(float(cohens_d), 4),
                            'effect': effect
                        }
                    }
                except Exception as e:
                    return {col: {'error': str(e)}}
            
            results = Parallel(
                n_jobs=self.config['performance']['n_jobs'],
                backend=self.config['performance']['backend']
            )(delayed(test_numeric)(col) for col in numeric_cols if col != target_col)
            
            combined = {}
            for r in results:
                combined.update(r)
            
            return combined
        except Exception as e:
            self.logger.error(f"Error in numeric tests: {e}")
            return {}
    
    def _categorical_tests(self, df, categorical_cols, target_col):
        """Chi-square tests"""
        try:
            def test_categorical(col):
                try:
                    contingency = pd.crosstab(df[col], df[target_col])
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    
                    n = contingency.sum().sum()
                    min_dim = min(contingency.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                    
                    effect = 'negligible' if cramers_v < 0.1 else 'small' if cramers_v < 0.3 else 'medium' if cramers_v < 0.5 else 'high'
                    
                    return {
                        col: {
                            'chi2': round(float(chi2), 4),
                            'p_value': round(float(p_val), 6),
                            'cramers_v': round(float(cramers_v), 4),
                            'effect': effect,
                            'significant': p_val < self.alpha
                        }
                    }
                except Exception as e:
                    return {col: {'error': str(e)}}
            
            results = Parallel(
                n_jobs=self.config['performance']['n_jobs'],
                backend=self.config['performance']['backend']
            )(delayed(test_categorical)(col) for col in categorical_cols if col != target_col)
            
            combined = {}
            for r in results:
                combined.update(r)
            
            return combined
        except Exception as e:
            self.logger.error(f"Error in categorical tests: {e}")
            return {}
    
    def _correlation_analysis(self, df, numeric_cols):
        """Spearman correlation analysis"""
        try:
            corr_matrix = df[numeric_cols].corr(method='spearman', numeric_only=True)
            
            return {
                'correlation_matrix': corr_matrix.round(4).to_dict(),
                'shape': corr_matrix.shape
            }
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            return {}
    
    def _target_distribution(self, df, target_col):
        """Analyze target variable distribution"""
        try:
            dist = df[target_col].value_counts()
            dist_pct = df[target_col].value_counts(normalize=True) * 100
            
            return {
                'distribution': dist.to_dict(),
                'distribution_pct': dist_pct.round(2).to_dict(),
                'class_imbalance_ratio': round(dist.iloc[0] / dist.iloc[1], 2) if len(dist) > 1 else None
            }
        except Exception as e:
            self.logger.error(f"Error analyzing target distribution: {e}")
            return {}