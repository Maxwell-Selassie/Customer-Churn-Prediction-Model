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
    # def generate_all_reports(self, quality_results, univariate_results, bivariate_results, df):
    #     """Generate all report formats"""
    #     try:
    #         output_formats = self.config['output']['formats']
    #         results_summary = {
    #             'quality': quality_results,
    #             'univariate': univariate_results,
    #             'bivariate': bivariate_results
    #         }
            
    #         if 'html' in output_formats:
    #             self._generate_html(quality_results, univariate_results, bivariate_results)
            
    #         if 'json' in output_formats:
    #             self.io_handler.save_json(results_summary, 'eda_results')
            
    #         if 'csv' in output_formats:
    #             self._save_csv_reports(quality_results, univariate_results, bivariate_results)
            
    #         if 'sqlite' in output_formats:
    #             self._save_sqlite_reports(quality_results, univariate_results, bivariate_results)
            
    #         self.logger.info("All reports generated successfully")
    #     except Exception as e:
    #         self.logger.error(f"Error generating reports: {e}")
    #         raise
    
    # @Timer.measure
    # def _generate_html(self, quality, univariate, bivariate):
    #     """Generate comprehensive HTML report"""
    #     try:
    #         html_content = self._build_html(quality, univariate, bivariate)
    #         report_path = self.io_handler.results_dir / f"{self.config['output']['report_name']}.html"
            
    #         with open(report_path, 'w') as f:
    #             f.write(html_content)
            
    #         self.logger.info(f"HTML report generated: {report_path}")
    #     except Exception as e:
    #         self.logger.error(f"Error generating HTML report: {e}")
    #         raise
    
    # def _build_html(self, quality, univariate, bivariate):
    #     """Build HTML structure"""
    #     html = """
    #     <!DOCTYPE html>
    #     <html lang="en">
    #     <head>
    #         <meta charset="UTF-8">
    #         <meta name="viewport" content="width=device-width, initial-scale=1.0">
    #         <title>EDA Report</title>
    #         <style>
    #             body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; color: #333; line-height: 1.6; }
    #             header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    #             header h1 { font-size: 2.5em; margin-bottom: 10px; }
    #             header p { font-size: 1.1em; opacity: 0.9; }
    #             .container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
    #             section { background: white; margin: 20px 0; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    #             h2 { color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; margin: 20px 0 15px; }
    #             h3 { color: #555; margin-top: 15px; margin-bottom: 10px; }
    #             table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.95em; }
    #             th { background-color: #667eea; color: white; padding: 12px; text-align: left; font-weight: 600; }
    #             td { padding: 10px 12px; border-bottom: 1px solid #e0e0e0; }
    #             tr:hover { background-color: #f8f9fa; }
    #             .metric { display: inline-block; background: #f8f9fa; padding: 15px 20px; margin: 10px 10px 10px 0; border-radius: 5px; border-left: 4px solid #667eea; }
    #             .metric-label { font-weight: 600; color: #667eea; }
    #             .metric-value { font-size: 1.3em; color: #333; font-weight: bold; }
    #             .warning { background-color: #fff3cd; border-left-color: #ffc107; }
    #             .success { background-color: #d4edda; border-left-color: #28a745; }
    #             .error { background-color: #f8d7da; border-left-color: #dc3545; }
    #             body { margin: 0; padding: 0; box-sizing: border-box; }
    #             footer { text-align: center; padding: 20px; color: #666; font-size: 0.9em; }
    #         </style>
    #     </head>
    #     <body>
    #         <header>
    #             <h1>📊 Exploratory Data Analysis Report</h1>
    #             <p>Comprehensive statistical analysis and data quality assessment</p>
    #         </header>
            
    #         <div class="container">
    #             <section>
    #                 <h2>1. Data Overview</h2>
    #                 <div class="metric success">
    #                     <div class="metric-label">Observations</div>
    #                     <div class="metric-value">{:,}</div>
    #                 </div>
    #                 <div class="metric success">
    #                     <div class="metric-label">Features</div>
    #                     <div class="metric-value">{:,}</div>
    #                 </div>
    #                 <div class="metric warning">
    #                     <div class="metric-label">Missing Values</div>
    #                     <div class="metric-value">{}</div>
    #                 </div>
    #                 <div class="metric warning">
    #                     <div class="metric-label">Duplicate Rows</div>
    #                     <div class="metric-value">{}</div>
    #                 </div>
    #             </section>
    #     """.format(
    #         quality['shape']['observations'],
    #         quality['shape']['features'],
    #         quality['missing_values']['count'],
    #         quality['duplicates']['count']
    #     )
        
    #     # Data Types Section
    #     if 'data_types' in quality:
    #         html += self._format_data_types_html(quality['data_types'])
        
    #     # Missing Values Details
    #     if quality['missing_values']['count'] > 0:
    #         html += self._format_missing_values_html(quality['missing_values']['details'])
        
    #     # Outliers Section
    #     html += self._format_outliers_html(quality['outliers'])
        
    #     # Univariate Analysis
    #     html += self._format_univariate_html(univariate)
        
    #     # Bivariate Analysis
    #     html += self._format_bivariate_html(bivariate)
        
    #     html += """
    #         </div>
    #         <footer>
    #             <p>Generated by Production-Level EDA Framework | All analyses completed successfully</p>
    #         </footer>
    #     </body>
    #     </html>
    #     """
    #     return html
    
    # def _format_data_types_html(self, data_types):
    #     html = '<section><h2>2. Data Types & Statistics</h2>'
        
    #     if 'numeric' in data_types and data_types['numeric']:
    #         html += '<h3>Numeric Columns</h3><table><tr><th>Column</th><th>Type</th><th>Min</th><th>Max</th></tr>'
    #         for col, info in data_types['numeric'].items():
    #             html += f"<tr><td>{col}</td><td>{info['type']}</td><td>{info['min']}</td><td>{info['max']}</td></tr>"
    #         html += '</table>'
        
    #     if 'categorical' in data_types and data_types['categorical']:
    #         html += '<h3>Categorical Columns</h3><table><tr><th>Column</th><th>Type</th><th>Unique</th></tr>'
    #         for col, info in data_types['categorical'].items():
    #             html += f"<tr><td>{col}</td><td>{info['type']}</td><td>{info['unique']}</td></tr>"
    #         html += '</table>'
        
    #     html += '</section>'
    #     return html
    
    # def _format_missing_values_html(self, missing_details):
    #     html = '<section><h2>3. Missing Values Detail</h2><table><tr><th>Column</th><th>Count</th><th>Percentage</th></tr>'
    #     for col, info in missing_details.items():
    #         html += f"<tr><td>{col}</td><td>{info['missing']}</td><td>{info['missing_pct']}%</td></tr>"
    #     html += '</table></section>'
    #     return html
    
    # def _format_outliers_html(self, outliers):
    #     html = '<section><h2>4. Outlier Detection (IQR Method)</h2><table><tr><th>Column</th><th>Count</th><th>Percentage</th><th>Valid Range</th></tr>'
    #     for col, info in outliers.items():
    #         html += f"<tr><td>{col}</td><td>{info['count']}</td><td>{info['pct']}%</td><td>{info['range']}</td></tr>"
    #     html += '</table></section>'
    #     return html
    
    # def _format_univariate_html(self, univariate):
    #     html = '<section><h2>5. Univariate Analysis</h2>'
        
    #     if 'normality_tests' in univariate:
    #         html += '<h3>Normality Tests (Shapiro-Wilk)</h3><table><tr><th>Column</th><th>Statistic</th><th>P-Value</th><th>Normal</th></tr>'
    #         for col, info in univariate['normality_tests'].items():
    #             if 'error' not in info:
    #                 is_normal = '✓ Yes' if info['is_normal'] else '✗ No'
    #                 html += f"<tr><td>{col}</td><td>{info['statistic']}</td><td>{info['p_value']}</td><td>{is_normal}</td></tr>"
    #         html += '</table>'
        
    #     if 'confidence_intervals' in univariate:
    #         html += '<h3>95% Confidence Intervals</h3><table><tr><th>Column</th><th>Mean</th><th>Lower Bound</th><th>Upper Bound</th><th>Margin of Error</th></tr>'
    #         for col, info in univariate['confidence_intervals'].items():
    #             if 'error' not in info:
    #                 html += f"<tr><td>{col}</td><td>{info['mean']}</td><td>{info['lower_bound']}</td><td>{info['upper_bound']}</td><td>{info['margin_of_error']}</td></tr>"
    #         html += '</table>'
        
    #     html += '</section>'
    #     return html
    
    # def _format_bivariate_html(self, bivariate):
    #     html = '<section><h2>6. Bivariate Analysis</h2>'
        
    #     if 'numeric_vs_target' in bivariate and bivariate['numeric_vs_target']:
    #         html += '<h3>Numeric vs Target (Mann-Whitney U / Welch\'s t-test)</h3>'
    #         html += '<table><tr><th>Column</th><th>Test Type</th><th>Group1 Mean</th><th>Group2 Mean</th><th>P-Value</th><th>Significant</th><th>Cohen\'s d</th><th>Effect Size</th></tr>'
    #         for col, info in bivariate['numeric_vs_target'].items():
    #             if 'error' not in info:
    #                 sig = '✓ Yes' if info['significant'] else '✗ No'
    #                 html += f"<tr><td>{col}</td><td>{info['test_type']}</td><td>{info['group1_mean']}</td><td>{info['group2_mean']}</td><td>{info['p_value']}</td><td>{sig}</td><td>{info['cohens_d']}</td><td>{info['effect']}</td></tr>"
    #         html += '</table>'
        
    #     if 'categorical_vs_target' in bivariate and bivariate['categorical_vs_target']:
    #         html += '<h3>Categorical vs Target (Chi-Square)</h3>'
    #         html += '<table><tr><th>Column</th><th>Chi2</th><th>P-Value</th><th>Cramer\'s V</th><th>Significant</th><th>Effect Size</th></tr>'
    #         for col, info in bivariate['categorical_vs_target'].items():
    #             if 'error' not in info:
    #                 sig = '✓ Yes' if info['significant'] else '✗ No'
    #                 html += f"<tr><td>{col}</td><td>{info['chi2']}</td><td>{info['p_value']}</td><td>{info['cramers_v']}</td><td>{sig}</td><td>{info['effect']}</td></tr>"
    #         html += '</table>'
        
    #     if 'target_distribution' in bivariate:
    #         html += '<h3>Target Variable Distribution</h3>'
    #         html += '<div class="metric success">'
    #         for target_val, count in bivariate['target_distribution']['distribution'].items():
    #             pct = bivariate['target_distribution']['distribution_pct'].get(target_val, 0)
    #             html += f'<div style="display: inline-block; margin-right: 20px;"><strong>{target_val}</strong>: {count:,} ({pct}%)</div>'
    #         html += '</div>'
        
    #     html += '</section>'
    #     return html
    
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