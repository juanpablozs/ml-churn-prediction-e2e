.PHONY: help install db-up db-down db-reset ingest train evaluate api test clean all

help:
	@echo "Available commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make db-up      - Start MySQL database"
	@echo "  make db-down    - Stop MySQL database"
	@echo "  make db-reset   - Reset database (drop and recreate)"
	@echo "  make ingest     - Generate and ingest data into MySQL"
	@echo "  make train      - Train churn prediction models"
	@echo "  make evaluate   - Evaluate trained model"
	@echo "  make api        - Start FastAPI server"
	@echo "  make test       - Run all tests"
	@echo "  make clean      - Clean generated files"
	@echo "  make all        - Run complete pipeline (db-up, ingest, train, evaluate)"

install:
	pip install -r requirements.txt
	pip install -e .

db-up:
	docker-compose up -d mysql
	@echo "Waiting for MySQL to be ready..."
	@sleep 10

db-down:
	docker-compose down

db-reset: db-down db-up

ingest:
	python -m churn.data.generate_data
	python -m churn.data.ingest

train:
	python -m churn.train.train

evaluate:
	python -m churn.evaluate.evaluate

api:
	docker-compose up api

api-local:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf mlruns
	rm -rf models/*.joblib
	rm -rf reports/*.json

all: db-up ingest train evaluate
	@echo "Pipeline completed successfully!"
