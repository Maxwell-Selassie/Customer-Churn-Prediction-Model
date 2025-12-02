import pandas as pd
import json
import sqlite3
from pathlib import Path
from utils.logger import Logger
from utils.timer import Timer

class IOHandler:
    def __init__(self, results_dir='./results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = Logger().get_logger()
    
    @Timer.measure
    def read_csv(self, filepath, encoding='utf-8', **kwargs):
        """Read CSV with error handling"""
        try:
            self.logger.info(f"Reading CSV: {filepath}")
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            
            if df.empty:
                raise ValueError("DataFrame is empty after reading")
            
            self.logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("CSV file is empty")
            raise
        except Exception as e:
            self.logger.error(f"Error reading CSV: {e}")
            raise
    
    @Timer.measure
    def save_csv(self, df, filename, index=False):
        """Save DataFrame to CSV"""
        try:
            filepath = self.results_dir / f"{filename}.csv"
            df.to_csv(filepath, index=index)
            self.logger.info(f"Saved CSV: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
            raise
    
    @Timer.measure
    def save_json(self, data, filename):
        """Save data to JSON"""
        try:
            filepath = self.results_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"Saved JSON: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}")
            raise
    
    @Timer.measure
    def save_to_sqlite(self, data, table_name, db_name='eda_results.db'):
        """Save results to SQLite database"""
        try:
            db_path = self.results_dir / db_name
            conn = sqlite3.connect(db_path)
            
            if isinstance(data, pd.DataFrame):
                data.to_sql(table_name, conn, if_exists='replace', index=False)
            else:
                df = pd.DataFrame([data])
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            conn.close()
            self.logger.info(f"Saved to SQLite table '{table_name}' in {db_path}")
            return db_path
        except Exception as e:
            self.logger.error(f"Error saving to SQLite: {e}")
            raise
    
    @staticmethod
    def validate_file(filepath):
        """Validate file exists and is readable"""
        try:
            p = Path(filepath)
            if not p.exists():
                raise FileNotFoundError(f"File does not exist: {filepath}")
            if not p.is_file():
                raise ValueError(f"Path is not a file: {filepath}")
            if p.stat().st_size == 0:
                raise ValueError(f"File is empty: {filepath}")
            return True
        except Exception as e:
            logger = Logger().get_logger()
            logger.error(f"File validation failed: {e}")
            raise