import time
from functools import wraps
from utils.logger import Logger

class Timer:
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.logger = Logger().get_logger()
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} completed in {elapsed:.4f} seconds")
    
    @staticmethod
    def measure(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = Logger().get_logger()
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} executed in {elapsed:.4f} seconds")
            return result
        return wrapper