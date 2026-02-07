"""Database connection utilities."""

import mysql.connector
from mysql.connector import Error
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from churn.config import DB_CONFIG


def get_db_connection():
    """
    Create and return a MySQL database connection.
    
    Returns:
        MySQL connection object
    
    Raises:
        Error: If connection fails
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        
        if connection.is_connected():
            return connection
        else:
            raise Error("Failed to connect to MySQL database")
    
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise


def test_connection():
    """Test database connection."""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        print(f"✓ Connected to database: {db_name}")
        cursor.close()
        connection.close()
        return True
    except Error as e:
        print(f"✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
