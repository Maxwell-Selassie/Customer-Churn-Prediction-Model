
"""
Custom Logger Module for EDA Pipeline
Provides consistent logging across all modules
"""

import logging
import os
from pathlib import Path
from datetime import datetime


class EDAPipelineLogger:
    """Singleton logger for EDA pipeline with file and console handlers"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EDAPipelineLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_file=None, level=logging.INFO):
        if self._initialized:
            return
        
        self.logger = logging.getLogger('EDAPipeline')
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self._initialized = True
    
    def get_logger(self):
        """Return the configured logger instance"""
        return self.logger


def get_logger(module_name, log_file=None, level=logging.INFO):
    """Convenience function to get logger for specific module"""
    pipeline_logger = EDAPipelineLogger(log_file=log_file, level=level)
    return pipeline_logger.logger.getChild(module_name)