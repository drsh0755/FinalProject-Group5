"""
Comprehensive Logging Utility
Usage: from utils.logger import setup_logger
       logger = setup_logger('script_name', log_dir='Exhibition/logs')
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_dir='Exhibition/logs', level=logging.INFO):
    """
    Setup logger with both file and console output
    
    Args:
        name: Logger name (usually script name)
        log_dir: Directory to store log files
        level: Logging level (default: INFO)
    
    Returns:
        logger: Configured logger instance
        log_file: Path to log file
    """
    # Create logs directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(message)s'  # Clean console output
    )
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (clean)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log the log file location
    logger.info(f"Log file: {log_file}")
    
    return logger, log_file

def log_section(logger, title):
    """Log a section header"""
    separator = "=" * 60
    logger.info(separator)
    logger.info(title)
    logger.info(separator)

def log_dict(logger, data, title="Data"):
    """Log dictionary contents nicely"""
    logger.info(f"{title}:")
    for key, value in data.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

def log_dataframe_info(logger, df, name="DataFrame"):
    """Log DataFrame summary"""
    logger.info(f"{name} Info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Date range: {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}" if 'Date' in df.columns else "  No date column")
    logger.info(f"  Missing values: {df.isnull().sum().sum()}")
