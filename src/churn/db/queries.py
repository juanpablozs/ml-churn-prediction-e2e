"""Database query utilities for fetching training data."""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from churn.db.connection import get_db_connection


def fetch_training_data() -> pd.DataFrame:
    """
    Fetch all customer data from MySQL for training.
    
    Returns:
        DataFrame with customer data
    """
    connection = get_db_connection()
    
    query = """
        SELECT 
            customer_id,
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
            churn
        FROM customers
    """
    
    try:
        df = pd.read_sql(query, connection)
        return df
    finally:
        connection.close()


def fetch_customer_by_id(customer_id: str) -> pd.DataFrame:
    """
    Fetch a specific customer by ID.
    
    Args:
        customer_id: Customer ID
    
    Returns:
        DataFrame with customer data
    """
    connection = get_db_connection()
    
    query = """
        SELECT * FROM customers
        WHERE customer_id = %s
    """
    
    try:
        df = pd.read_sql(query, connection, params=(customer_id,))
        return df
    finally:
        connection.close()


def get_churn_statistics() -> dict:
    """
    Get basic statistics about churn in the database.
    
    Returns:
        Dictionary with churn statistics
    """
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Total customers
        cursor.execute("SELECT COUNT(*) as total FROM customers")
        total = cursor.fetchone()["total"]
        
        # Churned customers
        cursor.execute("SELECT COUNT(*) as churned FROM customers WHERE churn = 'Yes'")
        churned = cursor.fetchone()["churned"]
        
        # Average tenure
        cursor.execute("SELECT AVG(tenure) as avg_tenure FROM customers")
        avg_tenure = cursor.fetchone()["avg_tenure"]
        
        # Average monthly charges
        cursor.execute("SELECT AVG(monthly_charges) as avg_charges FROM customers")
        avg_charges = cursor.fetchone()["avg_charges"]
        
        return {
            "total_customers": total,
            "churned_customers": churned,
            "churn_rate": churned / total if total > 0 else 0,
            "average_tenure_months": float(avg_tenure) if avg_tenure else 0,
            "average_monthly_charges": float(avg_charges) if avg_charges else 0,
        }
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    # Test queries
    print("Testing database queries...")
    
    stats = get_churn_statistics()
    print("\nChurn Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nFetching sample data...")
    df = fetch_training_data()
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
