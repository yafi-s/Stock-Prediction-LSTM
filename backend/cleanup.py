# ----------------------------------
# Model and Metrics Cleanup
# ----------------------------------
# This script provides utilities for cleaning up:
# - Old model files
# - Outdated metrics
# - Temporary training files
# Use this to maintain a clean workspace and remove unnecessary files.

import os
import argparse
import logging
from database import engine, get_session
from models import Prediction
from sqlmodel import select

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_old_files(days: int = 30):
    """
    Remove model and metrics files older than specified days.
    Args:
        days: Number of days after which files should be removed
    """
    pass

def cleanup_symbol_files(symbol: str):
    """
    Remove all model and metrics files for a specific symbol.
    Args:
        symbol: Stock ticker symbol to clean up
    """
    try:
        # List of file patterns to remove
        file_patterns = [
            f"models/{symbol}_best.keras",
            f"models/{symbol}_feature_scaler.joblib",
            f"models/{symbol}_target_scaler.joblib",
            f"models/{symbol}_history.txt",
            f"models/{symbol}_metrics.txt"
        ]
        
        # Remove files
        for file_pattern in file_patterns:
            if os.path.exists(file_pattern):
                os.remove(file_pattern)
                logger.info(f"Removed {file_pattern}")
            else:
                logger.info(f"File not found: {file_pattern}")
        
        # Remove predictions from database
        with get_session() as session:
            predictions = session.exec(
                select(Prediction).where(Prediction.symbol == symbol)
            ).all()
            
            for prediction in predictions:
                session.delete(prediction)
            
            session.commit()
            logger.info(f"Removed {len(predictions)} predictions from database")
        
        logger.info(f"Cleanup completed for {symbol}")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean up model files and database entries for a stock')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to clean up (e.g., AAPL)')
    args = parser.parse_args()
    
    cleanup_symbol_files(args.symbol) 