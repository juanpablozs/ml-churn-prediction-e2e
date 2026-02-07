"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional


class CustomerFeatures(BaseModel):
    """Input features for churn prediction."""
    
    customer_id: str = Field(..., description="Unique customer identifier")
    gender: str = Field(..., description="Customer gender (Male/Female)")
    senior_citizen: int = Field(..., ge=0, le=1, description="Senior citizen flag (0 or 1)")
    partner: str = Field(..., description="Has partner (Yes/No)")
    dependents: str = Field(..., description="Has dependents (Yes/No)")
    tenure: int = Field(..., ge=0, description="Months as customer")
    phone_service: str = Field(..., description="Has phone service (Yes/No)")
    multiple_lines: str = Field(..., description="Has multiple lines (Yes/No/No phone service)")
    internet_service: str = Field(..., description="Internet service type (DSL/Fiber optic/No)")
    online_security: str = Field(..., description="Has online security (Yes/No/No internet service)")
    online_backup: str = Field(..., description="Has online backup (Yes/No/No internet service)")
    device_protection: str = Field(..., description="Has device protection (Yes/No/No internet service)")
    tech_support: str = Field(..., description="Has tech support (Yes/No/No internet service)")
    streaming_tv: str = Field(..., description="Has streaming TV (Yes/No/No internet service)")
    streaming_movies: str = Field(..., description="Has streaming movies (Yes/No/No internet service)")
    contract: str = Field(..., description="Contract type (Month-to-month/One year/Two year)")
    paperless_billing: str = Field(..., description="Has paperless billing (Yes/No)")
    payment_method: str = Field(..., description="Payment method")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges in dollars")
    total_charges: float = Field(..., ge=0, description="Total charges in dollars")
    
    @validator('senior_citizen', pre=True)
    def convert_senior_citizen(cls, v):
        """Convert senior_citizen to int if string."""
        if isinstance(v, str):
            return int(v)
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for churn prediction."""
    
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    will_churn: bool
    risk_level: str
    threshold_used: float
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST00001",
                "churn_probability": 0.7234,
                "will_churn": True,
                "risk_level": "HIGH",
                "threshold_used": 0.5
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_version: str
    model_name: Optional[str] = None
    threshold: float
    features: list
    created_at: Optional[str] = None
