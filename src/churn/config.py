"""Configuration management for churn prediction."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "database": os.getenv("MYSQL_DATABASE", "churn_db"),
    "user": os.getenv("MYSQL_USER", "churn_user"),
    "password": os.getenv("MYSQL_PASSWORD", "churn_password"),
}

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "best_model.joblib"))
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))

# Training configuration
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.2"))

# Feature configuration
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

TARGET_COLUMN = "churn"

# Model metadata
MODEL_METADATA = {
    "version": "1.0.0",
    "created_at": None,  # Will be set during training
    "threshold": MODEL_THRESHOLD,
    "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
}
