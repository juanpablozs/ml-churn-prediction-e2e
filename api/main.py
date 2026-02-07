"""FastAPI application for churn prediction."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.models import (
    CustomerFeatures, PredictionResponse,
    HealthResponse, ModelInfoResponse
)
from churn.config import MODEL_THRESHOLD, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from churn.utils.helpers import format_prediction_response, get_risk_level

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Production-ready API for customer churn prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
model_metadata = None
model_loaded = False


def load_model():
    """Load the trained model at startup."""
    global model, model_metadata, model_loaded
    
    try:
        # Get model path from environment or use default
        model_path = os.getenv("MODEL_PATH", "/app/models/best_model.joblib")
        
        # Fallback to relative path for local development
        if not Path(model_path).exists():
            model_path = Path(__file__).parent.parent / "models" / "best_model.joblib"
        
        if not Path(model_path).exists():
            print(f"WARNING: Model not found at {model_path}")
            print("API will start but /predict endpoint will return errors")
            return
        
        # Load model
        model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
        
        # Load metadata
        metadata_path = Path(model_path).parent / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            print(f"✓ Model metadata loaded")
        else:
            # Create basic metadata
            model_metadata = {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "threshold": MODEL_THRESHOLD,
                "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            }
            print(f"! Model metadata not found, using defaults")
        
        model_loaded = True
        print("✓ Model ready for predictions")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    load_model()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service health status and model availability.
    """
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns model version, threshold, and feature schema.
    """
    if not model_loaded or model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfoResponse(
        model_version=model_metadata.get("version", "unknown"),
        model_name=model_metadata.get("model_name"),
        threshold=model_metadata.get("threshold", MODEL_THRESHOLD),
        features=model_metadata.get("features", []),
        created_at=model_metadata.get("created_at")
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn probability for a customer.
    
    Takes customer features as input and returns:
    - Churn probability (0-1)
    - Binary prediction (will_churn: True/False)
    - Risk level (LOW/MEDIUM/HIGH)
    - Threshold used for classification
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert Pydantic model to dict and then to DataFrame
        customer_dict = customer.dict()
        customer_id = customer_dict.pop("customer_id")
        
        # Create DataFrame with correct column order
        feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        
        # Ensure all features are present
        customer_data = {col: customer_dict.get(col) for col in feature_columns}
        df = pd.DataFrame([customer_data])
        
        # Convert senior_citizen to string for consistency with training
        df['senior_citizen'] = df['senior_citizen'].astype(str)
        
        # Make prediction
        churn_probability = model.predict_proba(df)[0, 1]
        
        # Get threshold from metadata or use default
        threshold = model_metadata.get("threshold", MODEL_THRESHOLD)
        
        # Format response
        response = format_prediction_response(
            customer_id=customer_id,
            churn_probability=churn_probability,
            threshold=threshold
        )
        
        return PredictionResponse(**response)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
