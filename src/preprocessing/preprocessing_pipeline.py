import pandas as pd
import joblib
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


from utils.logger import Logger
from utils.timer import Timer
from utils.file_utils import IOHandler
class PreprocessingPipeline:
    """Orchestrate all preprocessing steps"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config
        self.io_handler = IOHandler(config['output']['processed_dir'])
        
        # Initialize all preprocessors
        from preprocessing.data_splitter import DataSplitter
        from preprocessing.missing_handler import MissingHandler
        from preprocessing.business_logic import BusinessLogicHandler
        from preprocessing.duplicate_handler import DuplicateHandler
        from preprocessing.outlier_handler import OutlierHandler
        from preprocessing.datetime_features import DatetimeFeatureExtractor
        from preprocessing.feature_engineering import FeatureEngineer
        from preprocessing.encoding import FeatureEncoder
        from preprocessing.transformations import FeatureTransformer
        
        self.splitter = DataSplitter(config)
        self.missing_handler = MissingHandler(config)
        self.business_logic = BusinessLogicHandler(config)
        self.duplicate_handler = DuplicateHandler(config)
        self.outlier_handler = OutlierHandler(config)
        self.datetime_extractor = DatetimeFeatureExtractor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.encoder = FeatureEncoder(config)
        self.transformer = FeatureTransformer(config)
    
    @Timer.measure
    def fit_transform(self, df):
        """
        Fit and transform training data, then transform dev/test
        Prevents data leakage by fitting only on training set
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING PREPROCESSING PIPELINE")
            self.logger.info("=" * 80)
            
            # Stage 0: Pre-split data cleaning (before split)
            self.logger.info("\n[Stage 0] Pre-split Data Cleaning...")
            df = self.missing_handler.handle_missing(df)
            df = self.business_logic.handle_business_logic(df)
            df = self.duplicate_handler.handle_duplicates(df)
            
            # Stage 1: Split data
            self.logger.info("\n[Stage 1] Splitting Data...")
            train_set, dev_set, test_set = self.splitter.split_data(df)
            
            # Stage 2: Fit transformers on training set
            self.logger.info("\n[Stage 2] Fitting Transformers on Training Data...")
            train_set = self.outlier_handler.handle_outliers(train_set, fit=True)
            train_set = self.datetime_extractor.extract_features(train_set, fit=True)
            train_set = self.feature_engineer.engineer_features(train_set, fit=True)
            train_set = self.encoder.encode_features(train_set, fit=True)
            train_set = self.transformer.transform_features(train_set, fit=True)
            train_set = self._drop_columns(train_set)
            
            # Stage 3: Transform dev set
            self.logger.info("\n[Stage 3] Transforming Dev Set...")
            dev_set = self.outlier_handler.handle_outliers(dev_set, fit=False)
            dev_set = self.datetime_extractor.extract_features(dev_set, fit=False)
            dev_set = self.feature_engineer.engineer_features(dev_set, fit=False)
            dev_set = self.encoder.encode_features(dev_set, fit=False)
            dev_set = self.transformer.transform_features(dev_set, fit=False)
            dev_set = self._drop_columns(dev_set)
            
            # Stage 4: Transform test set
            self.logger.info("\n[Stage 4] Transforming Test Set...")
            test_set = self.outlier_handler.handle_outliers(test_set, fit=False)
            test_set = self.datetime_extractor.extract_features(test_set, fit=False)
            test_set = self.feature_engineer.engineer_features(test_set, fit=False)
            test_set = self.encoder.encode_features(test_set, fit=False)
            test_set = self.transformer.transform_features(test_set, fit=False)
            test_set = self._drop_columns(test_set)
            
            # Stage 5: Validation
            self.logger.info("\n[Stage 5] Validating Data...")
            self._validate_sets(train_set, dev_set, test_set)
            
            # Stage 6: Save outputs
            self.logger.info("\n[Stage 6] Saving Outputs...")
            self._save_datasets(train_set, dev_set, test_set)
            self._save_pipeline()
            # self._generate_report(train_set, dev_set, test_set)
            
            self.logger.info("=" * 80)
            self.logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return train_set, dev_set, test_set
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _drop_columns(self, df):
        """Drop columns specified in config"""
        cols_to_drop = self.config['columns_to_drop']
        existing_cols = [col for col in cols_to_drop if col in df.columns]
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
            self.logger.debug(f"Dropped columns: {existing_cols}")
        
        return df
    
    def _validate_sets(self, train, dev, test):
        """Validate no data leakage and check data quality"""
        try:
            self.logger.info("Validating data integrity...")
            
            # Check for overlapping rows
            train_idx = set(range(len(train)))
            dev_idx = set(range(len(dev)))
            test_idx = set(range(len(test)))
            
            overlap_train_dev = train_idx & dev_idx
            overlap_train_test = train_idx & test_idx
            overlap_dev_test = dev_idx & test_idx
            
            if overlap_train_dev or overlap_train_test or overlap_dev_test:
                self.logger.warning("Potential data leakage detected!")
            else:
                self.logger.info("✓ No data leakage detected")
            
            # Check for missing values
            train_missing = train.isnull().sum().sum()
            dev_missing = dev.isnull().sum().sum()
            test_missing = test.isnull().sum().sum()
            
            self.logger.info(f"Train missing values: {train_missing}")
            self.logger.info(f"Dev missing values: {dev_missing}")
            self.logger.info(f"Test missing values: {test_missing}")
            
            # Check shapes
            self.logger.info(f"Train shape: {train.shape}")
            self.logger.info(f"Dev shape: {dev.shape}")
            self.logger.info(f"Test shape: {test.shape}")
            
            # Check target distribution if target exists
            if 'Churn_Flag' in train.columns:
                train_churn = train['Churn_Flag'].value_counts()
                dev_churn = dev['Churn_Flag'].value_counts()
                test_churn = test['Churn_Flag'].value_counts()
                
                self.logger.info(f"Train target distribution:\n{train_churn}")
                self.logger.info(f"Dev target distribution:\n{dev_churn}")
                self.logger.info(f"Test target distribution:\n{test_churn}")
        
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
    
    def _save_datasets(self, train, dev, test):
        """Save preprocessed datasets"""
        try:
            splits_config = self.config['output']['splits']
            
            self.io_handler.save_csv(train, splits_config['train'])
            self.io_handler.save_csv(dev, splits_config['dev'])
            self.io_handler.save_csv(test, splits_config['test'])
            
            self.logger.info("All datasets saved successfully")
        
        except Exception as e:
            self.logger.error(f"Error saving datasets: {e}")
            raise
    
    def _save_pipeline(self):
        """Serialize pipeline to joblib file for inference"""
        try:
            pipeline_file = self.config['output']['pipeline_file']
            pipeline_path = Path(self.config['output']['processed_dir']) / pipeline_file
            
            # Create pipeline object with all fitted preprocessors
            pipeline_obj = {
                'outlier_handler': self.outlier_handler,
                'datetime_extractor': self.datetime_extractor,
                'feature_engineer': self.feature_engineer,
                'encoder': self.encoder,
                'transformer': self.transformer,
                'config': self.config
            }
            
            joblib.dump(pipeline_obj, pipeline_path)
            self.logger.info(f"Pipeline saved to {pipeline_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving pipeline: {e}")
            raise

def load_config(config_path='config/preprocessing_config.yaml'):
    """Load preprocessing configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        Logger().get_logger().info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        Logger().get_logger().error(f"Error loading config: {e}")
        raise


def main():
    """Main execution"""
    
    logger = Logger().get_logger()
    logger.info("=" * 80)
    logger.info("PRODUCTION PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Load config
        config = load_config()
        
        # Load raw data
        io_handler = IOHandler()
        df = io_handler.read_csv(
            config['data']['file_path'],
            encoding=config['data']['encoding']
        )
        
        logger.info(f"Raw data shape: {df.shape}")
        
        # Initialize and run pipeline
        pipeline = PreprocessingPipeline(config)
        train_set, dev_set, test_set = pipeline.fit_transform(df)
        
        logger.info("Preprocessing completed successfully!")
        
        return train_set, dev_set, test_set
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    train, dev, test = main()