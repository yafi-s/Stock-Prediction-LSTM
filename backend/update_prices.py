# ----------------------------------
# Price Update Script
# ----------------------------------
# This script updates actual prices for past predictions.

import requests
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_actual_prices():
    """Update actual prices for past predictions."""
    try:
        # Call the update endpoint
        response = requests.post("http://localhost:8000/api/update-actual-prices")
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Updated {result['updated']} predictions with actual prices")
        
    except Exception as e:
        logger.error(f"Error updating prices: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info(f"Starting price update at {datetime.now()}")
    update_actual_prices()
    logger.info("Price update completed") 