import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys

sys.path.insert(0, Path(str(__file__)).parent.parent)


from data_loader import DataLoader
from data_quality import DataQuality
from univariate import UnivariateAnalysis
from bivariate import BivariateAnalysis
from report_generator import ReportGenerator
from utils import Logger, Timer


class EDAPipeline:
    """Main EDA execution pipeline"""
    
    def __init__(self, config_path='config.yaml'):
        self.logger = Logger().get_logger()
        self.config_path = config_path
        self.logger.info("=" * 80)
        self.logger.info("PRODUCTION-LEVEL EDA FRAMEWORK INITIALIZED")
        self.logger.info("=" * 80)
    
    @Timer.measure
    def execute(self):
        """Execute complete EDA pipeline"""
        try:
            self.logger.info("Starting EDA pipeline execution...")
            
            # Stage 1: Data Loading
            with Timer("Data Loading"):
                loader = DataLoader(self.config_path)
                df = loader.load_data()
                config = loader.get_config()
            
            # Stage 2: Data Quality Checks
            with Timer("Data Quality Checks"):
                quality_checker = DataQuality(config)
                quality_results = quality_checker.run_quality_checks(df)
            
            # Stage 3: Univariate Analysis
            with Timer("Univariate Analysis"):
                univariate = UnivariateAnalysis(config)
                univariate_results = univariate.run_analysis(df)
            
            # Stage 4: Bivariate Analysis
            target_col = config.get('target_column')
            with Timer("Bivariate Analysis"):
                bivariate = BivariateAnalysis(config)
                bivariate_results = bivariate.run_analysis(df, target_col)
            
            # Stage 5: Report Generation
            with Timer("Report Generation"):
                report_gen = ReportGenerator(config)
                report_gen.generate_all_reports(
                    quality_results, univariate_results, bivariate_results, df
                )
            
            self.logger.info("=" * 80)
            self.logger.info("EDA PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return {
                'status': 'success',
                'quality': quality_results,
                'univariate': univariate_results,
                'bivariate': bivariate_results,
                'data_shape': df.shape
            }
        
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    pipeline = EDAPipeline('config.yaml')
    results = pipeline.execute()