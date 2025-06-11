# ----------------------------------
# Database Models
# ----------------------------------
# This module defines the SQL database models for storing:
# - Historical stock data
# - Model predictions
# - Training metrics

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class StockData(SQLModel, table=True):
    """
    Database model for storing historical stock data.
    Fields:
    - id: Primary key
    - symbol: Stock ticker symbol
    - date: Trading date
    - open/high/low/close: Price data
    - volume: Trading volume
    - created_at: Record creation timestamp
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    date: datetime = Field(index=True)
    open: float
    high: float
    low: float
    close: float
    volume: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Prediction(SQLModel, table=True):
    """
    Database model for storing model predictions.
    Fields:
    - id: Primary key
    - symbol: Stock ticker symbol
    - date: Prediction date
    - predicted_close: Model's predicted closing price
    - actual_close: Actual closing price
    - created_at: Record creation timestamp
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    date: datetime = Field(index=True)
    predicted_close: float
    actual_close: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
