"""Model evaluation and threshold tuning."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
import joblib
import json
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from churn.config import MODELS_DIR, REPORTS_DIR


def evaluate_at_threshold(y_true, y_proba, threshold=0.5):
    """
    Evaluate model performance at a specific threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary with metrics
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def find_optimal_threshold(y_true, y_proba, metric="f1"):
    """
    Find optimal threshold based on precision-recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ("f1", "precision", "recall")
    
    Returns:
        Optimal threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    if metric == "f1":
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element (threshold artifact)
        return thresholds[optimal_idx]
    elif metric == "precision":
        # Find threshold for 80% precision
        target_precision = 0.80
        valid_indices = precisions[:-1] >= target_precision
        if valid_indices.any():
            return thresholds[valid_indices][0]
        return 0.5
    elif metric == "recall":
        # Find threshold for 80% recall
        target_recall = 0.80
        valid_indices = recalls[:-1] >= target_recall
        if valid_indices.any():
            return thresholds[valid_indices][-1]
        return 0.5
    else:
        return 0.5


def comprehensive_evaluation(model, X_test, y_test):
    """
    Perform comprehensive model evaluation.
    
    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with evaluation results
    """
    print("\nPerforming comprehensive evaluation...")
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    
    # Find optimal thresholds
    print("\nFinding optimal thresholds...")
    optimal_f1_threshold = find_optimal_threshold(y_test, y_proba, metric="f1")
    optimal_precision_threshold = find_optimal_threshold(y_test, y_proba, metric="precision")
    optimal_recall_threshold = find_optimal_threshold(y_test, y_proba, metric="recall")
    
    print(f"  Optimal F1 threshold: {optimal_f1_threshold:.3f}")
    print(f"  Optimal Precision threshold (80%): {optimal_precision_threshold:.3f}")
    print(f"  Optimal Recall threshold (80%): {optimal_recall_threshold:.3f}")
    
    # Evaluate at different thresholds
    thresholds_to_test = [0.3, 0.4, 0.5, optimal_f1_threshold, 0.6, 0.7]
    threshold_results = {}
    
    print("\nEvaluating at different thresholds:")
    print("-" * 80)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for threshold in thresholds_to_test:
        result = evaluate_at_threshold(y_test, y_proba, threshold)
        threshold_results[f"threshold_{threshold:.2f}"] = result
        
        print(f"{threshold:<12.2f} {result['precision']:<12.3f} {result['recall']:<12.3f} {result['f1_score']:<12.3f}")
    
    # Recommended threshold (optimal F1)
    recommended_threshold = optimal_f1_threshold
    recommended_metrics = evaluate_at_threshold(y_test, y_proba, recommended_threshold)
    
    print("\n" + "=" * 80)
    print(f"RECOMMENDED THRESHOLD: {recommended_threshold:.3f}")
    print("=" * 80)
    print(f"Precision: {recommended_metrics['precision']:.3f}")
    print(f"Recall:    {recommended_metrics['recall']:.3f}")
    print(f"F1-Score:  {recommended_metrics['f1_score']:.3f}")
    
    cm = np.array(recommended_metrics['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    # Business interpretation
    print("\nBusiness Interpretation:")
    print(f"  - Out of {len(y_test)} customers:")
    print(f"    • {cm[1, 1]} churners correctly identified (True Positives)")
    print(f"    • {cm[1, 0]} churners missed (False Negatives)")
    print(f"    • {cm[0, 1]} false alarms (False Positives)")
    print(f"    • Prevention rate: {cm[1, 1] / (cm[1, 1] + cm[1, 0]) * 100:.1f}% of churners can be targeted")
    
    # Compile results
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "test_samples": len(y_test),
        "churn_rate": float(y_test.mean()),
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
        "optimal_thresholds": {
            "f1_optimized": float(optimal_f1_threshold),
            "precision_optimized": float(optimal_precision_threshold),
            "recall_optimized": float(optimal_recall_threshold),
        },
        "recommended_threshold": float(recommended_threshold),
        "recommended_threshold_metrics": {
            "precision": float(recommended_metrics["precision"]),
            "recall": float(recommended_metrics["recall"]),
            "f1_score": float(recommended_metrics["f1_score"]),
            "confusion_matrix": recommended_metrics["confusion_matrix"],
        },
        "threshold_comparison": {
            k: {
                "precision": float(v["precision"]),
                "recall": float(v["recall"]),
                "f1_score": float(v["f1_score"]),
            }
            for k, v in threshold_results.items()
        },
    }
    
    return evaluation_results


def save_evaluation_report(results, report_name="evaluation_report.json"):
    """
    Save evaluation report to file.
    
    Args:
        results: Evaluation results dictionary
        report_name: Name of the report file
    """
    report_path = REPORTS_DIR / report_name
    
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation report saved to: {report_path}")
    
    # Also save a markdown summary
    md_path = REPORTS_DIR / "evaluation_summary.md"
    with open(md_path, "w") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")
        f.write(f"**Test Samples:** {results['test_samples']}\n\n")
        f.write(f"**Churn Rate:** {results['churn_rate']*100:.2f}%\n\n")
        
        f.write("## Model Performance\n\n")
        f.write(f"- **ROC-AUC:** {results['metrics']['roc_auc']:.4f}\n")
        f.write(f"- **PR-AUC:** {results['metrics']['pr_auc']:.4f}\n\n")
        
        f.write("## Recommended Configuration\n\n")
        f.write(f"**Threshold:** {results['recommended_threshold']:.3f}\n\n")
        
        metrics = results['recommended_threshold_metrics']
        f.write(f"- **Precision:** {metrics['precision']:.3f}\n")
        f.write(f"- **Recall:** {metrics['recall']:.3f}\n")
        f.write(f"- **F1-Score:** {metrics['f1_score']:.3f}\n\n")
        
        cm = metrics['confusion_matrix']
        f.write("### Confusion Matrix\n\n")
        f.write("```\n")
        f.write(f"                 Predicted No  Predicted Yes\n")
        f.write(f"Actual No        {cm[0][0]:<13} {cm[0][1]:<13}\n")
        f.write(f"Actual Yes       {cm[1][0]:<13} {cm[1][1]:<13}\n")
        f.write("```\n\n")
        
        f.write("## Why PR-AUC Matters for Churn\n\n")
        f.write("In imbalanced classification (churn is typically <30%), PR-AUC is more informative than ROC-AUC because:\n\n")
        f.write("- Focuses on the minority class (churners)\n")
        f.write("- More sensitive to improvements in precision and recall\n")
        f.write("- Better reflects real-world business impact\n\n")
        
        f.write("## Threshold Selection Rationale\n\n")
        f.write("The recommended threshold maximizes F1-score, balancing:\n\n")
        f.write("- **Precision:** Avoiding wasting resources on false alarms\n")
        f.write("- **Recall:** Catching as many churners as possible\n\n")
        f.write("Adjust threshold based on business priorities:\n\n")
        f.write("- Lower threshold → Higher recall (catch more churners, but more false positives)\n")
        f.write("- Higher threshold → Higher precision (fewer false alarms, but miss some churners)\n")
    
    print(f"✓ Evaluation summary saved to: {md_path}")


def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Load model
    print("\n1. Loading model...")
    model_path = MODELS_DIR / "best_model.joblib"
    
    if not model_path.exists():
        print(f"✗ Error: Model not found at {model_path}")
        print("  Please run training first: python -m churn.train.train")
        sys.exit(1)
    
    model = joblib.load(model_path)
    print(f"   ✓ Model loaded from: {model_path}")
    
    # Load test data
    print("\n2. Loading test data...")
    test_data_path = MODELS_DIR / "test_data.joblib"
    
    if not test_data_path.exists():
        print(f"✗ Error: Test data not found at {test_data_path}")
        print("  Please run training first: python -m churn.train.train")
        sys.exit(1)
    
    test_data = joblib.load(test_data_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    print(f"   ✓ Test data loaded: {len(y_test)} samples")
    
    # Evaluate
    print("\n3. Evaluating model...")
    results = comprehensive_evaluation(model, X_test, y_test)
    
    # Save report
    print("\n4. Saving evaluation report...")
    save_evaluation_report(results)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nRecommended threshold: {results['recommended_threshold']:.3f}")
    print(f"Update MODEL_THRESHOLD in .env to use this threshold in production")


if __name__ == "__main__":
    main()
