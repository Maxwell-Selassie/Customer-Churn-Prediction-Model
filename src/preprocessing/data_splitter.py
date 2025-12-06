import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.logger import Logger
from utils.timer import Timer

class DataSplitter:
    """Split data before any transformations to prevent leakage"""
    
    def __init__(self, config):
        self.logger = Logger().get_logger()
        self.config = config['data_split']
    
    @Timer.measure
    def split_data(self, df):
        """
        Split data into train/dev/test sets
        Stratified by target column to maintain class distribution
        """
        try:
            self.logger.info("Starting data split...")
            
            test_size = self.config['test_size']
            dev_size = self.config['dev_size']
            random_state = self.config['random_state']
            stratify_col = self.config['stratify_column']
            
            total_size = len(df)
            self.logger.info(f"Total observations: {total_size}")
            
            # First split: separate test set
            train_dev, test_set = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df[stratify_col] if stratify_col else None
            )
            
            # Second split: separate dev from train
            train_set, dev_set = train_test_split(
                train_dev,
                test_size=dev_size,
                random_state=random_state,
                stratify=train_dev[stratify_col] if stratify_col else None
            )
            
            self.logger.info(f"Train set: {len(train_set)} rows ({len(train_set)/total_size*100:.1f}%)")
            self.logger.info(f"Dev set: {len(dev_set)} rows ({len(dev_set)/total_size*100:.1f}%)")
            self.logger.info(f"Test set: {len(test_set)} rows ({len(test_set)/total_size*100:.1f}%)")
            
            # Validate class distribution
            self._validate_split(df, train_set, dev_set, test_set, stratify_col)
            
            return train_set.reset_index(drop=True), dev_set.reset_index(drop=True), test_set.reset_index(drop=True)
        
        except Exception as e:
            self.logger.error(f"Error during data split: {e}")
            raise
    
    def _validate_split(self, full, train, dev, test, stratify_col):
        """Validate class distribution across splits"""
        if stratify_col:
            full_dist = full[stratify_col].value_counts(normalize=True)
            train_dist = train[stratify_col].value_counts(normalize=True)
            dev_dist = dev[stratify_col].value_counts(normalize=True)
            test_dist = test[stratify_col].value_counts(normalize=True)
            
            self.logger.info(f"Full dataset {stratify_col} distribution: {full_dist.to_dict()}")
            self.logger.info(f"Train {stratify_col} distribution: {train_dist.to_dict()}")
            self.logger.info(f"Dev {stratify_col} distribution: {dev_dist.to_dict()}")
            self.logger.info(f"Test {stratify_col} distribution: {test_dist.to_dict()}")

