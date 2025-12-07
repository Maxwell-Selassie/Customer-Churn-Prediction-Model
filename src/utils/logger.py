import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.log_dir = Path('./logs')
        self.log_dir.mkdir(exist_ok=True)
        
        LOG_FILE = self.log_dir/f'customer_churn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log'

        self.logger = logging.getLogger('CUSTOMER CHURN PREDICTION')
        self.logger.setLevel(logging.INFO)
        
        file_handler = TimedRotatingFileHandler(
            filename=LOG_FILE,
            when='midnight',
            backupCount=7,
        )
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self._initialized = True
    
    def get_logger(self):
        return self.logger