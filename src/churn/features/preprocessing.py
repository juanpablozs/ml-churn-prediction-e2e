"""Feature preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from churn.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor for churn prediction data."""
    
    def __init__(self):
        self.pipeline = None
        self.feature_names = None
    
    def fit(self, X, y=None):
        """Fit the preprocessing pipeline."""
        # Numeric features: StandardScaler
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        
        # Categorical features: OneHotEncoder
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ])
        
        # Combine transformers
        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ],
            remainder="drop"
        )
        
        self.pipeline.fit(X)
        
        # Store feature names
        self._extract_feature_names()
        
        return self
    
    def transform(self, X):
        """Transform the data."""
        if self.pipeline is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return self.pipeline.transform(X)
    
    def _extract_feature_names(self):
        """Extract feature names after fitting."""
        feature_names = []
        
        # Numeric features
        feature_names.extend(NUMERIC_FEATURES)
        
        # Categorical features (after one-hot encoding)
        if hasattr(self.pipeline.named_transformers_["cat"]["onehot"], "get_feature_names_out"):
            cat_features = self.pipeline.named_transformers_["cat"]["onehot"].get_feature_names_out(
                CATEGORICAL_FEATURES
            )
            feature_names.extend(cat_features)
        
        self.feature_names = feature_names
    
    def get_feature_names(self):
        """Get feature names after transformation."""
        return self.feature_names


def prepare_features_and_target(df: pd.DataFrame):
    """
    Prepare features and target from raw dataframe.
    
    Args:
        df: Raw dataframe with customer data
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Convert target to binary
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    
    # Select features
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns].copy()
    
    # Handle total_charges if it's object type (some datasets have spaces for missing values)
    if X["total_charges"].dtype == "object":
        X["total_charges"] = pd.to_numeric(X["total_charges"], errors="coerce")
    
    # Fill missing values in total_charges
    X["total_charges"] = X["total_charges"].fillna(0)
    
    # Ensure senior_citizen is string for categorical encoding consistency
    X["senior_citizen"] = X["senior_citizen"].astype(str)
    
    return X, y


def get_preprocessing_pipeline():
    """
    Get the preprocessing pipeline.
    
    Returns:
        ChurnPreprocessor instance
    """
    return ChurnPreprocessor()


if __name__ == "__main__":
    # Test preprocessing
    from churn.db.queries import fetch_training_data
    
    print("Testing preprocessing pipeline...")
    
    df = fetch_training_data()
    print(f"✓ Loaded {len(df)} rows")
    
    X, y = prepare_features_and_target(df)
    print(f"✓ Prepared features: {X.shape}")
    print(f"✓ Target distribution: {y.value_counts().to_dict()}")
    
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    
    print(f"✓ Transformed features: {X_transformed.shape}")
    print(f"✓ Feature names: {len(preprocessor.get_feature_names())} total")
