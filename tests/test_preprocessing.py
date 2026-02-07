"""Tests for feature preprocessing pipeline."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from churn.features.preprocessing import (
    ChurnPreprocessor, prepare_features_and_target,
    get_preprocessing_pipeline
)
from churn.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def sample_data():
    """Create sample customer data for testing."""
    return pd.DataFrame({
        'customer_id': ['CUST001', 'CUST002', 'CUST003'],
        'gender': ['Male', 'Female', 'Female'],
        'senior_citizen': [0, 1, 0],
        'partner': ['Yes', 'No', 'Yes'],
        'dependents': ['No', 'No', 'Yes'],
        'tenure': [12, 24, 6],
        'phone_service': ['Yes', 'Yes', 'No'],
        'multiple_lines': ['No', 'Yes', 'No phone service'],
        'internet_service': ['DSL', 'Fiber optic', 'No'],
        'online_security': ['Yes', 'No', 'No internet service'],
        'online_backup': ['No', 'Yes', 'No internet service'],
        'device_protection': ['Yes', 'No', 'No internet service'],
        'tech_support': ['No', 'Yes', 'No internet service'],
        'streaming_tv': ['Yes', 'No', 'No internet service'],
        'streaming_movies': ['No', 'Yes', 'No internet service'],
        'contract': ['Month-to-month', 'One year', 'Two year'],
        'paperless_billing': ['Yes', 'No', 'Yes'],
        'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'monthly_charges': [50.0, 75.5, 25.0],
        'total_charges': [600.0, 1812.0, 150.0],
        'churn': ['Yes', 'No', 'No']
    })


def test_prepare_features_and_target(sample_data):
    """Test feature and target preparation."""
    X, y = prepare_features_and_target(sample_data)
    
    # Check shapes
    assert X.shape[0] == 3
    assert len(y) == 3
    
    # Check target values
    assert y.dtype == np.int64 or y.dtype == int
    assert y.iloc[0] == 1  # First customer churned
    assert y.iloc[1] == 0  # Second customer didn't churn
    
    # Check features are present
    expected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert all(col in X.columns for col in expected_features)


def test_preprocessor_fit_transform(sample_data):
    """Test preprocessor fitting and transformation."""
    X, y = prepare_features_and_target(sample_data)
    
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    
    # Check transformation
    assert X_transformed.shape[0] == 3
    assert X_transformed.shape[1] > len(NUMERIC_FEATURES)  # Should have more features after one-hot encoding
    
    # Check feature names are extracted
    feature_names = preprocessor.get_feature_names()
    assert feature_names is not None
    assert len(feature_names) == X_transformed.shape[1]


def test_preprocessor_handles_missing_total_charges():
    """Test that preprocessor handles missing total_charges."""
    data = pd.DataFrame({
        'gender': ['Male'],
        'senior_citizen': [0],
        'partner': ['Yes'],
        'dependents': ['No'],
        'tenure': [12],
        'phone_service': ['Yes'],
        'multiple_lines': ['No'],
        'internet_service': ['DSL'],
        'online_security': ['Yes'],
        'online_backup': ['No'],
        'device_protection': ['Yes'],
        'tech_support': ['No'],
        'streaming_tv': ['Yes'],
        'streaming_movies': ['No'],
        'contract': ['Month-to-month'],
        'paperless_billing': ['Yes'],
        'payment_method': ['Electronic check'],
        'monthly_charges': [50.0],
        'total_charges': [np.nan],  # Missing value
        'churn': ['Yes']
    })
    
    X, y = prepare_features_and_target(data)
    
    # Should fill NaN with 0
    assert X['total_charges'].iloc[0] == 0


def test_get_preprocessing_pipeline():
    """Test getting preprocessing pipeline."""
    pipeline = get_preprocessing_pipeline()
    
    assert isinstance(pipeline, ChurnPreprocessor)
    assert pipeline.pipeline is None  # Not fitted yet


def test_preprocessor_reproducibility(sample_data):
    """Test that preprocessing is reproducible."""
    X, y = prepare_features_and_target(sample_data)
    
    # Transform twice
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(X)
    
    X_transformed_1 = preprocessor.transform(X)
    X_transformed_2 = preprocessor.transform(X)
    
    # Should be identical
    np.testing.assert_array_equal(X_transformed_1, X_transformed_2)


def test_preprocessor_handles_unknown_categories(sample_data):
    """Test that preprocessor handles unknown categorical values."""
    X_train, y = prepare_features_and_target(sample_data)
    
    # Fit on training data
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(X_train)
    
    # Create test data with unknown category
    X_test = X_train.copy()
    X_test.loc[0, 'contract'] = 'Unknown Contract Type'
    
    # Should handle unknown category gracefully
    X_transformed = preprocessor.transform(X_test)
    
    assert X_transformed is not None
    assert not np.isnan(X_transformed).any()
