"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src and api to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data
    assert data["status"] in ["healthy", "degraded"]


def test_model_info_endpoint():
    """Test model info endpoint."""
    response = client.get("/model-info")
    
    # Model might not be loaded in test environment
    if response.status_code == 200:
        data = response.json()
        assert "model_version" in data
        assert "threshold" in data
        assert "features" in data
    else:
        # If model not loaded, should return 503
        assert response.status_code == 503


def test_predict_endpoint_valid_payload():
    """Test prediction endpoint with valid payload."""
    payload = {
        "customer_id": "CUST00001",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 85.50,
        "total_charges": 1026.00
    }
    
    response = client.post("/predict", json=payload)
    
    # Model might not be loaded in test environment
    if response.status_code == 200:
        data = response.json()
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "will_churn" in data
        assert "risk_level" in data
        assert "threshold_used" in data
        
        # Validate data types and ranges
        assert isinstance(data["churn_probability"], float)
        assert 0 <= data["churn_probability"] <= 1
        assert isinstance(data["will_churn"], bool)
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        
    else:
        # If model not loaded, should return 503
        assert response.status_code == 503


def test_predict_endpoint_missing_field():
    """Test prediction endpoint with missing required field."""
    payload = {
        "customer_id": "CUST00001",
        "gender": "Female",
        # Missing required fields
    }
    
    response = client.post("/predict", json=payload)
    
    # Should return 422 Unprocessable Entity for validation error
    assert response.status_code == 422


def test_predict_endpoint_invalid_data_type():
    """Test prediction endpoint with invalid data type."""
    payload = {
        "customer_id": "CUST00001",
        "gender": "Female",
        "senior_citizen": "invalid",  # Should be int
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 85.50,
        "total_charges": 1026.00
    }
    
    response = client.post("/predict", json=payload)
    
    # Should return 422 for validation error
    assert response.status_code == 422


def test_predict_endpoint_negative_values():
    """Test prediction endpoint with negative values."""
    payload = {
        "customer_id": "CUST00001",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": -5,  # Invalid negative value
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 85.50,
        "total_charges": 1026.00
    }
    
    response = client.post("/predict", json=payload)
    
    # Should return 422 for validation error
    assert response.status_code == 422


def test_docs_endpoint():
    """Test that API documentation is accessible."""
    response = client.get("/docs")
    
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is accessible."""
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
