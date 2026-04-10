"""
Model Training Pipeline - Data-Driven Dynamic Pricing System

This module handles:
  1. Data loading & preprocessing
  2. Feature engineering (one-hot encoding for day_of_week)
  3. Train/test split (80/20)
  4. Training 3 models: Linear Regression, Random Forest, XGBoost
  5. Evaluation using RMSE & MAE
  6. Saving the best model + scaler + metadata
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_pricing_dataset.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess_data(path=DATA_PATH):
    """Load raw data and apply the same cleaning as the EDA notebook."""
    print("=" * 70)
    print("  STEP 1: Loading & Preprocessing Data")
    print("=" * 70)

    df = pd.read_excel(path, header=0)
    print(f"  [OK] Loaded {len(df):,} rows x {df.shape[1]} columns")

    # Drop 'month' column (same as notebook)
    if "month" in df.columns:
        df = df.drop("month", axis=1)

    # Temperature effect on demand (same as notebook)
    df["demand"] = df["demand"] + (df["temperature"] - 25) * 2
    df["demand"] = df["demand"].clip(lower=0)

    # Compute profit
    df["profit"] = (df["price"] - df["cost"]) * df["demand"]

    # Convert date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Filter extreme outliers (prices > 99th percentile distort the model)
    price_cap = df["price"].quantile(0.99)
    original_len = len(df)
    df = df[df["price"] <= price_cap].reset_index(drop=True)
    print(f"  [OK] Removed {original_len - len(df)} extreme-price outliers (>{price_cap:.2f})")

    print(f"  [OK] Preprocessing complete - shape: {df.shape}")
    print(f"  [OK] Demand range: [{df['demand'].min():.1f}, {df['demand'].max():.1f}]")
    print(f"  [OK] Price range:  [{df['price'].min():.2f}, {df['price'].max():.2f}]")
    print()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════
def engineer_features(df):
    """
    Prepare features for modeling:
      - Base: price, cost, temperature, is_weekend, is_festival, promo, competitor_price
      - Engineered: price_ratio, markup, price_x_promo
      - Categorical: day_of_week -> one-hot encoded
      - Target: demand
    """
    print("=" * 70)
    print("  STEP 2: Feature Engineering")
    print("=" * 70)

    # --- Engineered features ---
    # Price relative to competitor (key demand driver)
    df["price_ratio"] = df["price"] / df["competitor_price"].clip(lower=0.01)
    # Markup over cost
    df["markup"] = df["price"] - df["cost"]
    # Interaction: does promo amplify price sensitivity?
    df["price_x_promo"] = df["price"] * df["promo"]

    # Define feature groups
    numeric_features = [
        "price", "cost", "temperature", "is_weekend",
        "is_festival", "promo", "competitor_price",
        "price_ratio", "markup", "price_x_promo"
    ]

    # One-hot encode day_of_week
    day_dummies = pd.get_dummies(df["day_of_week"], prefix="day", drop_first=True)
    day_dummies = day_dummies.astype(int)

    # Combine features
    X = pd.concat([df[numeric_features], day_dummies], axis=1)
    y = df["demand"]

    # Store feature names for later use
    feature_names = list(X.columns)

    print(f"  [OK] Base features (7) + engineered (3) + day dummies ({len(day_dummies.columns)})")
    print(f"  [OK] Engineered: price_ratio, markup, price_x_promo")
    print(f"  [OK] Total features: {len(feature_names)}")
    print(f"  [OK] Target: demand")
    print()

    return X, y, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Train / Test Split & Scaling
# ═══════════════════════════════════════════════════════════════════════════════
def split_and_scale(X, y, test_size=0.2, random_state=42):
    """80/20 train-test split with StandardScaler."""
    print("=" * 70)
    print("  STEP 3: Train/Test Split & Scaling")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  [OK] Training set: {X_train.shape[0]:,} samples")
    print(f"  [OK] Test set:     {X_test.shape[0]:,} samples")
    print(f"  [OK] StandardScaler fitted on training data")
    print()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Model Training & Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def train_and_compare(X_train, X_test, y_train, y_test):
    """
    Train 3 models and compare them on RMSE, MAE, and R² score.

    Models:
      1. Linear Regression  — baseline
      2. Random Forest       — handles non-linearity well
      3. XGBoost             — state-of-the-art gradient boosting

    Returns dict of {model_name: (model, metrics_dict)}
    """
    print("=" * 70)
    print("  STEP 4: Model Training & Evaluation")
    print("=" * 70)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n  > Training {name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
        results[name] = (model, metrics)

        print(f"    RMSE : {rmse:>10.4f}")
        print(f"    MAE  : {mae:>10.4f}")
        print(f"    R2   : {r2:>10.4f}")

    # Comparison Table
    print("\n" + "-" * 70)
    print(f"  {'Model':<25} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print("-" * 70)
    for name, (_, m) in results.items():
        print(f"  {name:<25} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['R2']:>10.4f}")
    print("-" * 70)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Save the Best Model
# ═══════════════════════════════════════════════════════════════════════════════
def save_best_model(results, scaler, feature_names, model_dir=MODEL_DIR):
    """
    Pick the model with the lowest RMSE and persist:
      - best_model.pkl   (the trained model)
      - scaler.pkl        (fitted StandardScaler)
      - model_metadata.json (feature names, metrics, timestamp)
    """
    print("\n" + "=" * 70)
    print("  STEP 5: Saving Best Model")
    print("=" * 70)

    # Find best by lowest RMSE
    best_name = min(results, key=lambda k: results[k][1]["RMSE"])
    best_model, best_metrics = results[best_name]

    print(f"\n  [BEST] Best Model: {best_name}")
    print(f"     RMSE = {best_metrics['RMSE']:.4f}")
    print(f"     MAE  = {best_metrics['MAE']:.4f}")
    print(f"     R2   = {best_metrics['R2']:.4f}")

    # Save model
    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n  [OK] Model saved -> {model_path}")

    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  [OK] Scaler saved -> {scaler_path}")

    # Save metadata
    metadata = {
        "best_model_name": best_name,
        "metrics": best_metrics,
        "all_model_metrics": {k: v[1] for k, v in results.items()},
        "feature_names": feature_names,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Metadata saved -> {meta_path}")

    print()
    return best_name, best_model, best_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline():
    """Execute the full training pipeline end-to-end."""
    print("\n" + "=" * 70)
    print("  DATA-DRIVEN DYNAMIC PRICING - MODEL TRAINING PIPELINE")
    print("=" * 70 + "\n")

    # 1. Load data
    df = load_and_preprocess_data()

    # 2. Feature engineering
    X, y, feature_names = engineer_features(df)

    # 3. Split & scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # 4. Train & compare
    results = train_and_compare(X_train, X_test, y_train, y_test)

    # 5. Save best model
    best_name, best_model, best_metrics = save_best_model(
        results, scaler, feature_names
    )

    print("=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70 + "\n")

    return results, best_name, best_model, scaler, feature_names


if __name__ == "__main__":
    run_pipeline()
