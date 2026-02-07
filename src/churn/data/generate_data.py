"""Generate synthetic churn dataset."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from churn.config import RAW_DATA_DIR, RANDOM_SEED


def generate_synthetic_churn_data(n_samples: int = 7043, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset.
    
    This mimics the structure of the Telco Customer Churn dataset.
    
    Args:
        n_samples: Number of customer records to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic customer churn data
    """
    np.random.seed(seed)
    
    # Generate customer IDs
    customer_ids = [f"CUST{i:05d}" for i in range(1, n_samples + 1)]
    
    # Generate features
    data = {
        "customer_id": customer_ids,
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "senior_citizen": np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        "partner": np.random.choice(["Yes", "No"], n_samples, p=[0.52, 0.48]),
        "dependents": np.random.choice(["Yes", "No"], n_samples, p=[0.30, 0.70]),
        "tenure": np.random.randint(0, 73, n_samples),
        "phone_service": np.random.choice(["Yes", "No"], n_samples, p=[0.90, 0.10]),
    }
    
    # Multiple lines depends on phone service
    data["multiple_lines"] = [
        np.random.choice(["Yes", "No"]) if ps == "Yes" else "No phone service"
        for ps in data["phone_service"]
    ]
    
    # Internet service
    data["internet_service"] = np.random.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
    )
    
    # Services that depend on internet
    internet_dependent_features = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies"
    ]
    
    for feature in internet_dependent_features:
        data[feature] = [
            np.random.choice(["Yes", "No"]) if internet != "No" else "No internet service"
            for internet in data["internet_service"]
        ]
    
    # Contract and payment
    data["contract"] = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.21, 0.24]
    )
    data["paperless_billing"] = np.random.choice(["Yes", "No"], n_samples, p=[0.59, 0.41])
    data["payment_method"] = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_samples, p=[0.33, 0.23, 0.22, 0.22]
    )
    
    # Monthly charges (dependent on services)
    base_charge = np.random.uniform(18, 30, n_samples)
    fiber_bonus = np.where(np.array(data["internet_service"]) == "Fiber optic", 
                          np.random.uniform(20, 40, n_samples), 0)
    service_bonus = np.random.uniform(0, 20, n_samples)
    data["monthly_charges"] = np.round(base_charge + fiber_bonus + service_bonus, 2)
    
    # Total charges (tenure * monthly_charges with some noise)
    data["total_charges"] = np.round(
        np.array(data["tenure"]) * np.array(data["monthly_charges"]) * 
        np.random.uniform(0.95, 1.05, n_samples), 2
    )
    
    # Set total_charges to 0 for tenure 0
    data["total_charges"] = np.where(np.array(data["tenure"]) == 0, 0, data["total_charges"])
    
    # Generate churn (target variable)
    # Churn is more likely for:
    # - Month-to-month contracts
    # - Higher monthly charges
    # - Lower tenure
    # - Fiber optic internet (higher price dissatisfaction)
    
    churn_prob = np.zeros(n_samples)
    
    # Base probability
    churn_prob += 0.15
    
    # Contract effect
    churn_prob += np.where(np.array(data["contract"]) == "Month-to-month", 0.35, 0)
    churn_prob += np.where(np.array(data["contract"]) == "One year", 0.05, 0)
    
    # Tenure effect (inverse relationship)
    churn_prob += np.where(np.array(data["tenure"]) < 12, 0.25, 0)
    churn_prob += np.where((np.array(data["tenure"]) >= 12) & (np.array(data["tenure"]) < 24), 0.10, 0)
    churn_prob -= np.where(np.array(data["tenure"]) > 48, 0.15, 0)
    
    # Monthly charges effect
    churn_prob += np.where(np.array(data["monthly_charges"]) > 70, 0.15, 0)
    
    # Internet service effect
    churn_prob += np.where(np.array(data["internet_service"]) == "Fiber optic", 0.10, 0)
    
    # Electronic check (easier to cancel)
    churn_prob += np.where(np.array(data["payment_method"]) == "Electronic check", 0.10, 0)
    
    # Clip probabilities
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate churn based on probabilities
    data["churn"] = np.random.binomial(1, churn_prob, n_samples)
    data["churn"] = ["Yes" if c == 1 else "No" for c in data["churn"]]
    
    df = pd.DataFrame(data)
    
    return df


def main():
    """Generate and save synthetic churn data."""
    print("Generating synthetic churn dataset...")
    
    df = generate_synthetic_churn_data()
    
    # Save to CSV
    output_path = RAW_DATA_DIR / "churn_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} customer records")
    print(f"✓ Saved to: {output_path}")
    print(f"\nDataset statistics:")
    print(f"  - Churn rate: {(df['churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
    print(f"  - Features: {len(df.columns) - 2}")  # Exclude customer_id and churn
    print(f"  - Numeric features: {df.select_dtypes(include=[np.number]).columns.tolist()}")
    print(f"  - Missing values: {df.isnull().sum().sum()}")


if __name__ == "__main__":
    main()
