"""Model training pipeline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from churn.config import (
    RANDOM_SEED, TEST_SIZE, VAL_SIZE, MODELS_DIR, 
    MODEL_METADATA, REPORTS_DIR
)
from churn.db.queries import fetch_training_data
from churn.features.preprocessing import prepare_features_and_target, get_preprocessing_pipeline


def train_models(X_train, y_train, X_val, y_val):
    """
    Train multiple models and select the best one.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
    
    Returns:
        Dictionary with model results
    """
    # Define models to train
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ),
    }
    
    results = {}
    
    print("\nTraining models...")
    print("-" * 80)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Create pipeline with preprocessing
        preprocessor = get_preprocessing_pipeline()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        train_roc_auc = roc_auc_score(y_train, y_train_proba)
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        
        train_pr_auc = average_precision_score(y_train, y_train_proba)
        val_pr_auc = average_precision_score(y_val, y_val_proba)
        
        results[model_name] = {
            "pipeline": pipeline,
            "train_roc_auc": train_roc_auc,
            "val_roc_auc": val_roc_auc,
            "train_pr_auc": train_pr_auc,
            "val_pr_auc": val_pr_auc,
        }
        
        print(f"  Train ROC-AUC: {train_roc_auc:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")
        print(f"  Train PR-AUC:  {train_pr_auc:.4f} | Val PR-AUC:  {val_pr_auc:.4f}")
    
    return results


def select_best_model(results):
    """
    Select the best model based on validation PR-AUC.
    
    For imbalanced classification, PR-AUC is often more informative than ROC-AUC.
    
    Args:
        results: Dictionary with model results
    
    Returns:
        Tuple of (best_model_name, best_pipeline)
    """
    best_model_name = None
    best_val_pr_auc = -1
    
    for model_name, result in results.items():
        if result["val_pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = result["val_pr_auc"]
            best_model_name = model_name
    
    print(f"\n{'=' * 80}")
    print(f"Best model: {best_model_name}")
    print(f"Validation PR-AUC: {best_val_pr_auc:.4f}")
    print(f"{'=' * 80}\n")
    
    return best_model_name, results[best_model_name]["pipeline"]


def save_model(pipeline, model_name, metadata):
    """
    Save trained model and metadata.
    
    Args:
        pipeline: Trained sklearn pipeline
        model_name: Name of the model
        metadata: Model metadata dictionary
    """
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{model_name}_{timestamp}.joblib"
    model_path = MODELS_DIR / model_filename
    
    joblib.dump(pipeline, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Also save as best_model.joblib for easy reference
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(pipeline, best_model_path)
    print(f"✓ Model saved to: {best_model_path}")
    
    # Save metadata
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return best_model_path


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("CHURN PREDICTION MODEL TRAINING")
    print("=" * 80)
    
    # Fetch data from MySQL
    print("\n1. Fetching data from MySQL...")
    df = fetch_training_data()
    print(f"   ✓ Loaded {len(df)} customer records")
    
    # Prepare features and target
    print("\n2. Preparing features and target...")
    X, y = prepare_features_and_target(df)
    print(f"   ✓ Features shape: {X.shape}")
    print(f"   ✓ Target distribution: Churn={y.sum()} ({y.mean()*100:.2f}%), No Churn={len(y)-y.sum()}")
    
    # Split data
    print("\n3. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"   ✓ Train: {len(X_train)} samples ({y_train.mean()*100:.2f}% churn)")
    print(f"   ✓ Val:   {len(X_val)} samples ({y_val.mean()*100:.2f}% churn)")
    print(f"   ✓ Test:  {len(X_test)} samples ({y_test.mean()*100:.2f}% churn)")
    
    # Train models
    print("\n4. Training models...")
    results = train_models(X_train, y_train, X_val, y_val)
    
    # Select best model
    print("\n5. Selecting best model...")
    best_model_name, best_pipeline = select_best_model(results)
    
    # Prepare metadata
    metadata = MODEL_METADATA.copy()
    metadata["model_name"] = best_model_name
    metadata["created_at"] = datetime.now().isoformat()
    metadata["train_samples"] = len(X_train)
    metadata["val_samples"] = len(X_val)
    metadata["test_samples"] = len(X_test)
    metadata["churn_rate_train"] = float(y_train.mean())
    metadata["metrics"] = {
        "train_roc_auc": float(results[best_model_name]["train_roc_auc"]),
        "val_roc_auc": float(results[best_model_name]["val_roc_auc"]),
        "train_pr_auc": float(results[best_model_name]["train_pr_auc"]),
        "val_pr_auc": float(results[best_model_name]["val_pr_auc"]),
    }
    
    # Save model
    print("\n6. Saving model...")
    model_path = save_model(best_pipeline, best_model_name, metadata)
    
    # Save test data for evaluation
    print("\n7. Saving test data for evaluation...")
    test_data_path = MODELS_DIR / "test_data.joblib"
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_data_path)
    print(f"   ✓ Test data saved to: {test_data_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nBest model: {best_model_name}")
    print(f"Model path: {model_path}")
    print(f"\nNext step: Run evaluation with 'python -m churn.evaluate.evaluate'")


if __name__ == "__main__":
    main()
