import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from utils.file_utils import IOHandler
from utils.logger import Logger
from utils.timer import Timer
import sys

sys.path.insert(0, Path(str(__file__)).parent.parent)

from utils import IOHandler, Logger, Timer

class DataLoader:
    def __init__(self, config_path='config/eda_config.yaml'):
        self.logger = Logger().get_logger()
        self.io_handler = IOHandler()
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path):
        """Load YAML configuration"""
        filepath = Path(config_path)
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise
    
    @Timer.measure
    def load_data(self):
        """Load and preprocess data"""
        cfg = self.config['data']
        file_path = cfg['file_path']
        
        IOHandler.validate_file(file_path)
        
        df = self.io_handler.read_csv(
            file_path,
            encoding=cfg['encoding'],
            nrows=cfg['max_rows']
        )
        
        self._preprocess_types(df)
        
        self.logger.info(f"Data loading complete: {df.shape}")
        return df
    
    def _preprocess_types(self, df):
        """Convert types and handle edge cases"""
        try:
            round_cols = self.config['currencies']['round_cols']
            decimals = self.config['currencies']['decimal_places']
            
            for col in round_cols:
                if col in df.columns:
                    df[col] = df[col].round(decimals)
            
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            self.logger.debug(f"Type preprocessing completed")
        except Exception as e:
            self.logger.warning(f"Type preprocessing warning: {e}")
    
    def get_config(self):
        return self.config

