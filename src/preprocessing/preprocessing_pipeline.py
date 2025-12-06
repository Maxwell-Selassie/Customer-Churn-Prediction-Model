import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from utils.logger import Logger
from utils.file_utils import IOHandler
from preprocessing.transformations import PreprocessingPipeline
import yaml

def load_config(config_path='config.yaml'):
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
        config = load_config('config.yaml')
        
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