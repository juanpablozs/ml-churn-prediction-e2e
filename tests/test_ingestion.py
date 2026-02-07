"""Tests for data ingestion pipeline."""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from churn.data.generate_data import generate_synthetic_churn_data
from churn.data.ingest import clean_data


def test_generate_synthetic_data():
    """Test synthetic data generation."""
    n_samples = 100
    df = generate_synthetic_churn_data(n_samples=n_samples, seed=42)
    
    # Check shape
    assert len(df) == n_samples
    assert 'customer_id' in df.columns
    assert 'churn' in df.columns
    
    # Check customer IDs are unique
    assert df['customer_id'].nunique() == n_samples
    
    # Check churn values
    assert set(df['churn'].unique()).issubset({'Yes', 'No'})
    
    # Check numeric columns
    assert df['tenure'].dtype in [int, 'int64']
    assert df['monthly_charges'].dtype in [float, 'float64']
    assert df['total_charges'].dtype in [float, 'float64']
    
    # Check for reasonable ranges
    assert df['tenure'].min() >= 0
    assert df['monthly_charges'].min() >= 0
    assert df['total_charges'].min() >= 0


def test_synthetic_data_reproducibility():
    """Test that data generation is reproducible with same seed."""
    df1 = generate_synthetic_churn_data(n_samples=50, seed=42)
    df2 = generate_synthetic_churn_data(n_samples=50, seed=42)
    
    # Should be identical
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_data_churn_distribution():
    """Test that synthetic data has reasonable churn distribution."""
    df = generate_synthetic_churn_data(n_samples=1000, seed=42)
    
    churn_rate = (df['churn'] == 'Yes').sum() / len(df)
    
    # Churn rate should be reasonable (typically 15-35%)
    assert 0.10 < churn_rate < 0.50


def test_clean_data():
    """Test data cleaning function."""
    # Create sample data with issues
    df = pd.DataFrame({
        'customer_id': ['CUST001', 'CUST002', 'CUST003'],
        'tenure': [12, 24, 0],
        'monthly_charges': [50.0, 75.5, 25.0],
        'total_charges': ['600.0', ' ', '150.0'],  # One missing value as space
        'senior_citizen': ['0', '1', '0'],
        'churn': ['Yes', 'No', 'No']
    })
    
    df_clean = clean_data(df)
    
    # Check that total_charges is numeric
    assert df_clean['total_charges'].dtype in [float, 'float64']
    
    # Check that missing value was filled with 0
    assert df_clean['total_charges'].iloc[1] == 0
    
    # Check that senior_citizen is int
    assert df_clean['senior_citizen'].dtype in [int, 'int64']


def test_clean_data_handles_nulls():
    """Test that clean_data handles null values properly."""
    df = pd.DataFrame({
        'customer_id': ['CUST001'],
        'tenure': [12],
        'monthly_charges': [50.0],
        'total_charges': [None],
        'senior_citizen': [0],
        'churn': ['Yes']
    })
    
    df_clean = clean_data(df)
    
    # Should fill null with 0
    assert df_clean['total_charges'].iloc[0] == 0


def test_synthetic_data_column_types():
    """Test that synthetic data has correct column types."""
    df = generate_synthetic_churn_data(n_samples=100, seed=42)
    
    # Numeric columns
    assert df['tenure'].dtype in [int, 'int64']
    assert df['senior_citizen'].dtype in [int, 'int64']
    assert df['monthly_charges'].dtype in [float, 'float64']
    assert df['total_charges'].dtype in [float, 'float64']
    
    # Categorical columns
    categorical_cols = ['gender', 'partner', 'dependents', 'phone_service', 
                       'internet_service', 'contract', 'churn']
    for col in categorical_cols:
        assert df[col].dtype == object or df[col].dtype == 'string'


def test_synthetic_data_business_rules():
    """Test that synthetic data follows business rules."""
    df = generate_synthetic_churn_data(n_samples=1000, seed=42)
    
    # Total charges should generally be tenure * monthly_charges (with some variance)
    # For tenure > 0
    df_positive_tenure = df[df['tenure'] > 0].copy()
    
    if len(df_positive_tenure) > 0:
        df_positive_tenure['expected_total'] = df_positive_tenure['tenure'] * df_positive_tenure['monthly_charges']
        df_positive_tenure['ratio'] = df_positive_tenure['total_charges'] / df_positive_tenure['expected_total']
        
        # Ratio should be close to 1 (allowing for variance)
        assert df_positive_tenure['ratio'].mean() > 0.5
        assert df_positive_tenure['ratio'].mean() < 1.5
    
    # Tenure 0 should have total_charges 0
    zero_tenure = df[df['tenure'] == 0]
    if len(zero_tenure) > 0:
        assert (zero_tenure['total_charges'] == 0).all()
