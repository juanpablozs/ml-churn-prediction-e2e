# Churn Prediction - End-to-End ML Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, end-to-end machine learning project for predicting customer churn in subscription businesses. This project demonstrates industry best practices for ML engineering, from data ingestion to model deployment.

## 📋 Table of Contents

- [Business Problem](#business-problem)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [API Usage](#api-usage)
- [Model Performance](#model-performance)
- [Key Decisions](#key-decisions)
- [Testing](#testing)
- [Development](#development)

## 🎯 Business Problem

**Customer churn** is the rate at which customers stop doing business with a company. For subscription-based businesses, predicting churn is critical because:

- **Retention is cheaper than acquisition**: Acquiring new customers costs 5-25x more than retaining existing ones
- **Revenue impact**: A 5% increase in retention can increase profits by 25-95%
- **Proactive intervention**: Early identification allows targeted retention campaigns

This project builds a binary classification model to predict which customers are likely to churn, enabling:
- Targeted retention offers
- Improved customer service allocation
- Data-driven business decisions

## 🏗️ Architecture

```
┌─────────────────┐
│  MySQL Database │ ← Data Source
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  ETL Pipeline   │ ← Data Ingestion
│ (generate_data, │
│    ingest.py)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Training Pipeline│ ← Model Training
│  (3 models:     │   - Logistic Regression
│   LR, RF, GB)   │   - Random Forest
└────────┬────────┘   - Gradient Boosting
         │
         ↓
┌─────────────────┐
│   Evaluation    │ ← Threshold Tuning
│  (ROC-AUC,      │   - PR-AUC
│   PR-AUC, F1)   │   - Confusion Matrix
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Model Artifact │ ← Serialized Pipeline
│  (best_model.   │
│    joblib)      │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  FastAPI Service│ ← Production API
│  (Docker + REST)│   - /predict
└─────────────────┘   - /health
                      - /model-info
```

## ✨ Features

### Data Pipeline
- ✅ Synthetic data generation mimicking real-world churn patterns
- ✅ MySQL database for persistent storage
- ✅ SQL-based data retrieval (production-like workflow)
- ✅ Automated ETL with data validation

### Model Training
- ✅ 3 model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- ✅ Proper train/validation/test split with stratification
- ✅ Scikit-learn Pipeline with preprocessing
- ✅ Handling imbalanced classes (class weights)
- ✅ Reproducible with fixed random seeds

### Evaluation
- ✅ Comprehensive metrics (ROC-AUC, PR-AUC, F1, Precision, Recall)
- ✅ Threshold tuning for optimal F1-score
- ✅ Confusion matrix analysis
- ✅ Business-focused interpretation
- ✅ Evaluation reports (JSON + Markdown)

### API Service
- ✅ RESTful API with FastAPI
- ✅ Request/response validation with Pydantic
- ✅ Health checks and model info endpoints
- ✅ Error handling and logging
- ✅ Docker containerization
- ✅ API documentation (Swagger/ReDoc)

### Testing
- ✅ Unit tests for preprocessing
- ✅ Integration tests for API
- ✅ Data validation tests
- ✅ 90%+ code coverage goal

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| ML Framework | scikit-learn, XGBoost |
| Data Processing | pandas, numpy |
| Database | MySQL 8.0 (Docker) |
| API Framework | FastAPI + Uvicorn |
| Validation | Pydantic |
| Containerization | Docker + docker-compose |
| Testing | pytest |
| Linting | ruff |

## 📁 Project Structure

```
ml-churn-prediction-e2e/
├── api/                        # FastAPI application
│   ├── __init__.py
│   ├── main.py                # API endpoints
│   ├── models.py              # Pydantic models
│   └── Dockerfile             # API container
│
├── data/                       # Data storage
│   ├── raw/                   # Raw CSV data
│   └── processed/             # Processed data
│
├── models/                     # Trained model artifacts
│   ├── best_model.joblib      # Best model
│   └── model_metadata.json    # Model info
│
├── notebooks/                  # EDA (optional)
│   └── 01_eda.ipynb
│
├── reports/                    # Evaluation reports
│   ├── evaluation_report.json
│   └── evaluation_summary.md
│
├── sql/                        # Database schemas
│   └── init.sql               # MySQL initialization
│
├── src/churn/                  # Source code
│   ├── config.py              # Configuration
│   ├── data/                  # Data pipeline
│   │   ├── generate_data.py  # Synthetic data
│   │   └── ingest.py         # MySQL ingestion
│   ├── db/                    # Database layer
│   │   ├── connection.py     # DB connection
│   │   └── queries.py        # SQL queries
│   ├── features/              # Feature engineering
│   │   └── preprocessing.py  # Preprocessing pipeline
│   ├── train/                 # Training
│   │   └── train.py          # Training script
│   ├── evaluate/              # Evaluation
│   │   └── evaluate.py       # Evaluation script
│   └── utils/                 # Utilities
│       └── helpers.py        # Helper functions
│
├── tests/                      # Test suite
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_ingestion.py
│
├── .env.example                # Environment template
├── .gitignore
├── docker-compose.yml          # Container orchestration
├── Makefile                    # Automation commands
├── pytest.ini                  # Test configuration
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Make (optional, for convenience)

### Option 1: Using Make (Recommended)

```bash
# 1. Clone and navigate
cd ml-churn-prediction-e2e

# 2. Copy environment file
cp .env.example .env

# 3. Run complete pipeline
make install      # Install dependencies
make db-up        # Start MySQL
make ingest       # Generate and ingest data
make train        # Train models
make evaluate     # Evaluate model
make api          # Start API service

# 4. Test API
curl http://localhost:8000/health
```

### Option 2: Manual Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Start MySQL
docker-compose up -d mysql
sleep 10  # Wait for MySQL

# 3. Generate and ingest data
python -m churn.data.generate_data
python -m churn.data.ingest

# 4. Train model
python -m churn.train.train

# 5. Evaluate model
python -m churn.evaluate.evaluate

# 6. Start API
docker-compose up api
# OR for local development:
# uvicorn api.main:app --reload
```

## 📖 Detailed Setup

### 1. Environment Configuration

Copy `.env.example` to `.env` and customize:

```bash
# Database
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=churn_db
MYSQL_USER=churn_user
MYSQL_PASSWORD=churn_password

# Model
MODEL_THRESHOLD=0.5  # Update after evaluation
RANDOM_SEED=42
```

### 2. Database Setup

The MySQL container automatically initializes with the schema from `sql/init.sql`:

```sql
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    tenure INT,
    monthly_charges DECIMAL(10, 2),
    churn VARCHAR(10),
    ...
);
```

### 3. Data Generation

Generate 7,043 synthetic customer records:

```bash
python -m churn.data.generate_data
```

Features include:
- Demographics (gender, senior citizen, partner, dependents)
- Service usage (phone, internet, streaming)
- Contract details (type, billing, payment method)
- Charges (monthly, total)

### 4. Model Training

Train and compare 3 models:

```bash
python -m churn.train.train
```

**Models trained:**
1. **Logistic Regression** (baseline) - Fast, interpretable
2. **Random Forest** - Handles non-linearity, feature importance
3. **Gradient Boosting** - Often best performance

**Best model selection:** Based on validation PR-AUC

### 5. Model Evaluation

```bash
python -m churn.evaluate.evaluate
```

**Outputs:**
- `reports/evaluation_report.json` - Detailed metrics
- `reports/evaluation_summary.md` - Human-readable summary

**Metrics calculated:**
- ROC-AUC: Overall discrimination ability
- **PR-AUC**: Precision-Recall (better for imbalanced data)
- F1, Precision, Recall at various thresholds
- Confusion matrix
- Optimal threshold recommendation

## 🔌 API Usage

### Start the API

```bash
# Docker (recommended)
docker-compose up api

# Local
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-02-05T10:30:00"
}
```

#### 2. Model Information

```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_version": "1.0.0",
  "model_name": "gradient_boosting",
  "threshold": 0.487,
  "features": ["tenure", "monthly_charges", ...],
  "created_at": "2026-02-05T09:15:00"
}
```

#### 3. Predict Churn

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "customer_id": "CUST00001",
  "churn_probability": 0.7234,
  "will_churn": true,
  "risk_level": "HIGH",
  "threshold_used": 0.487
}
```

**Risk Levels:**
- `LOW`: < 30% probability
- `MEDIUM`: 30-60% probability
- `HIGH`: > 60% probability

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📊 Model Performance

### Example Results (Synthetic Data)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.85+ |
| PR-AUC | 0.70+ |
| F1-Score | 0.65+ |
| Precision | 0.68+ |
| Recall | 0.72+ |

### Confusion Matrix (Example)

```
                 Predicted No  Predicted Yes
Actual No        850           150
Actual Yes       90            320
```

**Business Interpretation:**
- 78% of churners correctly identified (320 out of 410)
- 85% precision on non-churners (minimizing false alarms)
- Can target 320 at-risk customers with retention campaigns

## 🧠 Key Decisions

### 1. Why PR-AUC over ROC-AUC?

Churn is typically imbalanced (20-30% churn rate). PR-AUC is more informative because:
- Focuses on the minority class (churners)
- More sensitive to improvements in precision/recall
- Better reflects business impact

**Example:** A model with high ROC-AUC but low precision would generate too many false alarms, wasting retention budget.

### 2. Threshold Tuning

Default threshold (0.5) is often suboptimal. We optimize for:
- **F1-Score**: Balances precision and recall
- **Business constraints**: Can adjust based on retention budget

**Recommended threshold:** Found via precision-recall curve analysis

### 3. Model Selection

**Gradient Boosting** often wins because:
- Handles non-linear relationships
- Feature interactions (e.g., contract type + tenure)
- Robust to feature scaling

**Logistic Regression** is kept as baseline for:
- Interpretability (coefficient analysis)
- Speed
- Regulatory compliance (explainability)

### 4. Pipeline Design

Scikit-learn Pipeline ensures:
- **No data leakage**: Preprocessing fit only on training data
- **Reproducibility**: Same transforms in training and inference
- **Simplicity**: Single object for entire workflow

### 5. SQL-Based Data Layer

Why MySQL instead of CSV:
- **Production realism**: Mirrors real data infrastructure
- **Scalability**: Handles larger datasets
- **Query flexibility**: Easy to slice data for analysis
- **ACID compliance**: Data integrity

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_api.py -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov=api --cov-report=html
```

### Test Coverage

- **Preprocessing**: Feature transformations, missing value handling
- **API**: Endpoints, validation, error handling
- **Data Ingestion**: Synthetic data generation, cleaning

## 👨‍💻 Development

### Running Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Run API with auto-reload
uvicorn api.main:app --reload
```

### Code Quality

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type checking (if using mypy)
mypy src/
```

### Database Management

```bash
# Connect to MySQL
docker exec -it churn_mysql mysql -u churn_user -p

# Reset database
make db-reset

# View logs
docker-compose logs mysql
```

## 🚧 Future Enhancements

- [ ] MLflow integration for experiment tracking
- [ ] Feature drift monitoring
- [ ] A/B testing framework
- [ ] Model versioning (multiple models in production)
- [ ] SHAP/LIME for model explainability
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Real-time predictions (Kafka/streaming)

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📧 Contact

For questions or feedback, please open an issue.

---

**Built with ❤️ as a demonstration of production ML engineering best practices**