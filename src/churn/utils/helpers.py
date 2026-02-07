"""Helper utilities for churn prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any


def format_prediction_response(
    customer_id: str,
    churn_probability: float,
    threshold: float,
    features: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Format prediction response for API.
    
    Args:
        customer_id: Customer identifier
        churn_probability: Predicted churn probability
        threshold: Classification threshold
        features: Optional input features
    
    Returns:
        Formatted response dictionary
    """
    will_churn = churn_probability >= threshold
    
    response = {
        "customer_id": customer_id,
        "churn_probability": round(float(churn_probability), 4),
        "will_churn": bool(will_churn),
        "threshold_used": float(threshold),
        "risk_level": get_risk_level(churn_probability),
    }
    
    if features:
        response["input_features"] = features
    
    return response


def get_risk_level(probability: float) -> str:
    """
    Categorize churn risk level.
    
    Args:
        probability: Churn probability
    
    Returns:
        Risk level string
    """
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"


def validate_customer_data(data: Dict[str, Any], required_fields: list) -> tuple[bool, str]:
    """
    Validate customer data for prediction.
    
    Args:
        data: Customer data dictionary
        required_fields: List of required field names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Check for null values
    null_fields = [field for field in required_fields if data[field] is None]
    
    if null_fields:
        return False, f"Null values in fields: {', '.join(null_fields)}"
    
    return True, ""


def calculate_feature_importance_summary(model, feature_names: list, top_n: int = 10) -> Dict[str, float]:
    """
    Extract feature importance from model.
    
    Args:
        model: Trained model with feature_importances_ or coef_
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        Dictionary of feature importances
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return {}
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    return {
        feature_names[i]: float(importances[i])
        for i in indices
    }
