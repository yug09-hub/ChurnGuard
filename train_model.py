"""
Customer Churn Prediction — Model Training Pipeline
====================================================
Loads the Telco customer churn dataset, preprocesses the data,
engineers features, trains multiple classifiers, and saves the best model.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "customer_churn_500_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.json")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
STATS_PATH = os.path.join(BASE_DIR, "dataset_stats.json")


def load_data():
    """Load and return the raw dataset."""
    print("[1/6] Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    print(f"       Loaded {len(df)} rows x {len(df.columns)} columns")
    return df


def preprocess(df: pd.DataFrame):
    """Clean the data and prepare it for modelling."""
    print("[2/6] Preprocessing ...")

    # Drop columns that do not carry predictive value
    drop_cols = ["customer_id"]
    # Keep 'churn' temporarily for target creation
    target = df["churn"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.drop(columns=["churn"], errors="ignore")

    # Convert total_charges to numeric if needed
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        df["total_charges"].fillna(df["total_charges"].median(), inplace=True)

    return df, target


def engineer_features(df: pd.DataFrame):
    """Create additional features from the raw columns."""
    print("[3/6] Engineering features ...")

    # Tenure buckets
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 12, 24, 48, 72, 200],
        labels=["0-12", "13-24", "25-48", "49-72", "72+"],
    )

    # Avg monthly charge relative to tenure
    df["avg_charge_per_month"] = np.where(
        df["tenure_months"] > 0,
        df["total_charges"] / df["tenure_months"],
        df["monthly_charges"],
    )

    # Has internet service (simplified - just based on internet_service column)
    if "internet_service" in df.columns:
        df["has_internet"] = df["internet_service"].apply(lambda x: 0 if x == "No" else 1)

    return df


def encode_and_scale(df: pd.DataFrame):
    """Encode categoricals and scale numerics."""
    print("[4/6] Encoding & scaling ...")
    label_encoders = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Fill any remaining NaNs (can appear from cut/encoding edge cases)
    df = df.fillna(0)

    return df, label_encoders, scaler


def train(X, y):
    """Train multiple classifiers and return the best one with metrics."""
    print("[5/6] Training models ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"       Training {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
        }

        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        results[name] = {"model": model, "metrics": metrics}
        print(f"       {name}: F1={metrics['f1']}, AUC={metrics['roc_auc']}, CV-F1={metrics['cv_f1_mean']}+/-{metrics['cv_f1_std']}")

    # Pick best by CV F1
    best_name = max(results, key=lambda k: results[k]["metrics"]["cv_f1_mean"])
    print(f"\n  [OK] Best model: {best_name}")
    return results[best_name]["model"], results, best_name


def compute_dataset_stats(df_raw: pd.DataFrame):
    """Compute summary statistics for the dashboard."""
    stats = {}

    # Overall churn rate (churn is 0/1 in new dataset)
    stats["total_customers"] = int(len(df_raw))
    stats["churned"] = int(df_raw["churn"].sum())
    stats["not_churned"] = stats["total_customers"] - stats["churned"]
    stats["churn_rate"] = round(stats["churned"] / stats["total_customers"] * 100, 2)

    # Avg monthly charges
    stats["avg_monthly_charges"] = round(float(df_raw["monthly_charges"].mean()), 2)
    stats["avg_total_charges"] = round(float(df_raw["total_charges"].mean()), 2)
    stats["avg_tenure"] = round(float(df_raw["tenure_months"].mean()), 1)

    # Churn by contract type
    contract_churn = df_raw.groupby("contract_type")["churn"].apply(
        lambda x: round(x.sum() / len(x) * 100, 2)
    ).to_dict()
    stats["churn_by_contract"] = contract_churn

    # Churn by internet service
    internet_churn = df_raw.groupby("internet_service")["churn"].apply(
        lambda x: round(x.sum() / len(x) * 100, 2)
    ).to_dict()
    stats["churn_by_internet"] = internet_churn

    # Churn by payment method
    payment_churn = df_raw.groupby("payment_method")["churn"].apply(
        lambda x: round(x.sum() / len(x) * 100, 2)
    ).to_dict()
    stats["churn_by_payment"] = payment_churn

    # Tenure distribution for churned vs not-churned
    tenure_bins = [0, 12, 24, 36, 48, 60, 72, 200]
    tenure_labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72", "72+"]
    df_raw["tenure_bin"] = pd.cut(df_raw["tenure_months"], bins=tenure_bins, labels=tenure_labels)
    tenure_churn = df_raw.groupby("tenure_bin", observed=True)["churn"].apply(
        lambda x: round(x.sum() / len(x) * 100, 2)
    ).to_dict()
    stats["churn_by_tenure"] = tenure_churn

    # Monthly charges distribution
    charge_bins = [0, 30, 50, 70, 90, 120]
    charge_labels = ["$0-30", "$30-50", "$50-70", "$70-90", "$90-120"]
    df_raw["charge_bin"] = pd.cut(df_raw["monthly_charges"], bins=charge_bins, labels=charge_labels)
    charge_dist = {}
    for label in charge_labels:
        subset = df_raw[df_raw["charge_bin"] == label]
        charge_dist[label] = {"total": int(len(subset)), "churned": int(subset["churn"].sum())}
    stats["charges_distribution"] = charge_dist

    # Gender split
    gender_churn = df_raw.groupby("gender")["churn"].apply(
        lambda x: round(x.sum() / len(x) * 100, 2)
    ).to_dict()
    stats["churn_by_gender"] = gender_churn

    # Senior citizen churn (senior_citizen is 0/1)
    senior_churn = df_raw.groupby("senior_citizen")["churn"].apply(
        lambda x: round(x.sum() / len(x) * 100, 2)
    ).to_dict()
    # Convert 0/1 keys to readable labels
    stats["churn_by_senior"] = {"No": senior_churn.get(0, 0), "Yes": senior_churn.get(1, 0)}

    return stats


def save_artifacts(model, label_encoders, scaler, feature_cols, all_results, best_name, stats):
    """Save all trained artifacts to disk."""
    print("[6/6] Saving artifacts ...")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    metrics_out = {}
    for name, data in all_results.items():
        metrics_out[name] = data["metrics"]
    metrics_out["best_model"] = best_name

    # Feature importances from the best model
    importances = model.feature_importances_
    importance_dict = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    metrics_out["feature_importances"] = [
        {"feature": feat, "importance": round(imp, 4)}
        for feat, imp in importance_dict[:20]
    ]

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  [OK] Model saved to {MODEL_PATH}")
    print(f"  [OK] Encoders saved to {ENCODERS_PATH}")
    print(f"  [OK] Scaler saved to {SCALER_PATH}")
    print(f"  [OK] Feature columns saved to {FEATURES_PATH}")
    print(f"  [OK] Metrics saved to {METRICS_PATH}")
    print(f"  [OK] Dataset stats saved to {STATS_PATH}")


def main():
    print("=" * 60)
    print("  Customer Churn Prediction - Model Training Pipeline")
    print("=" * 60 + "\n")

    # Load raw data (keep a copy for stats)
    df_raw = load_data()
    stats = compute_dataset_stats(df_raw.copy())

    # Preprocess
    df, y = preprocess(df_raw.copy())

    # Feature engineering
    df = engineer_features(df)

    # Encode & scale
    df, label_encoders, scaler = encode_and_scale(df)

    feature_cols = df.columns.tolist()

    # Train
    best_model, all_results, best_name = train(df, y)

    # Save everything
    save_artifacts(best_model, label_encoders, scaler, feature_cols, all_results, best_name, stats)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
