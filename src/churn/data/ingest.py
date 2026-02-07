"""Ingest CSV data into MySQL database."""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from churn.config import RAW_DATA_DIR
from churn.db.connection import get_db_connection


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw churn data.
    
    Args:
        df: Raw dataframe
    
    Returns:
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Handle missing values in total_charges (replace with 0 or median)
    if df_clean["total_charges"].dtype == "object":
        df_clean["total_charges"] = pd.to_numeric(df_clean["total_charges"], errors="coerce")
    
    df_clean["total_charges"] = df_clean["total_charges"].fillna(0)
    
    # Ensure senior_citizen is int
    if "senior_citizen" in df_clean.columns:
        df_clean["senior_citizen"] = df_clean["senior_citizen"].astype(int)
    
    return df_clean


def ingest_to_mysql(df: pd.DataFrame, batch_size: int = 1000) -> int:
    """
    Ingest dataframe into MySQL customers table.
    
    Args:
        df: Cleaned dataframe
        batch_size: Number of rows to insert per batch
    
    Returns:
        Number of rows inserted
    """
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Clear existing data
        cursor.execute("TRUNCATE TABLE customers")
        print("✓ Cleared existing data from customers table")
        
        # Prepare insert statement
        columns = df.columns.tolist()
        placeholders = ", ".join(["%s"] * len(columns))
        insert_query = f"""
            INSERT INTO customers ({", ".join(columns)})
            VALUES ({placeholders})
        """
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            values = [tuple(row) for row in batch.values]
            cursor.executemany(insert_query, values)
            total_inserted += len(values)
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"  Inserted {total_inserted}/{len(df)} rows...")
        
        connection.commit()
        print(f"✓ Successfully inserted {total_inserted} rows into customers table")
        
        return total_inserted
        
    except Exception as e:
        connection.rollback()
        print(f"✗ Error during ingestion: {e}")
        raise
    
    finally:
        cursor.close()
        connection.close()


def main():
    """Main ingestion pipeline."""
    print("Starting data ingestion pipeline...")
    
    # Load raw data
    csv_path = RAW_DATA_DIR / "churn_data.csv"
    
    if not csv_path.exists():
        print(f"✗ Error: {csv_path} not found!")
        print("  Please run: python -m churn.data.generate_data first")
        sys.exit(1)
    
    print(f"✓ Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Clean data
    print("✓ Cleaning data...")
    df_clean = clean_data(df)
    
    # Ingest to MySQL
    print("✓ Ingesting data to MySQL...")
    rows_inserted = ingest_to_mysql(df_clean)
    
    print(f"\n✓ Ingestion completed successfully!")
    print(f"  Total rows in database: {rows_inserted}")


if __name__ == "__main__":
    main()
