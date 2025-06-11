# ----------------------------------
# Database Configuration
# ----------------------------------
# This module handles database connection and session management.
# It provides:
# - Database engine configuration
# - Session management
# - Database initialization

from sqlmodel import SQLModel, create_engine, Session
import os

# Create database directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Create SQLite database engine
DATABASE_URL = "sqlite:///data/stock_prediction.db"
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    # Drop all tables and recreate them
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)

def get_session():
    """
    Create and return a new database session.
    Returns:
        SQLModel Session object for database operations
    """
    return Session(engine) 