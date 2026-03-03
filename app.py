"""
Customer Churn Prediction - Flask Backend API
===============================================
Serves the dashboard and provides REST API endpoints for
real-time churn prediction, customer analytics, and batch processing.
"""

import os
import json
import io
import csv
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_file

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.json")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
STATS_PATH = os.path.join(BASE_DIR, "dataset_stats.json")
DATA_PATH = os.path.join(BASE_DIR, "Telco_customer_churn.csv")

# ── Flask App ────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load ML Artifacts ────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_columns = json.load(f)

with open(METRICS_PATH, "r") as f:
    model_metrics = json.load(f)

with open(STATS_PATH, "r") as f:
    dataset_stats = json.load(f)

# Load raw dataset for customer exploration
df_raw = pd.read_csv(DATA_PATH)

# ── Retention Strategies Database ────────────────────────────────────────
RETENTION_STRATEGIES = {
    "high_monthly_charges": {
        "title": "Offer Competitive Pricing",
        "description": "Customer's monthly charges are above average. Consider offering a loyalty discount of 15-25% or a bundled package deal.",
        "icon": "dollar-sign",
        "priority": "high"
    },
    "month_to_month_contract": {
        "title": "Incentivize Long-Term Contract",
        "description": "Customer is on a month-to-month contract. Offer a discounted annual plan with additional perks like free premium channels for 3 months.",
        "icon": "file-contract",
        "priority": "high"
    },
    "no_tech_support": {
        "title": "Provide Premium Tech Support",
        "description": "Customer lacks tech support. Offer complimentary tech support for 6 months to improve their experience and reduce friction.",
        "icon": "headset",
        "priority": "medium"
    },
    "no_online_security": {
        "title": "Add Security Package",
        "description": "Customer has no online security. Bundle a free security package to increase perceived value and stickiness.",
        "icon": "shield",
        "priority": "medium"
    },
    "fiber_optic_user": {
        "title": "Fiber Optic Retention Package",
        "description": "Fiber optic customers churn more frequently. Offer speed upgrades or a premium entertainment bundle at current pricing.",
        "icon": "wifi",
        "priority": "medium"
    },
    "low_tenure": {
        "title": "New Customer Onboarding Program",
        "description": "Customer is relatively new. Assign a dedicated account manager and provide a personalized welcome kit with usage tips.",
        "icon": "user-plus",
        "priority": "high"
    },
    "no_online_backup": {
        "title": "Free Cloud Backup Trial",
        "description": "Offer a 3-month free trial of online backup services to increase engagement and service dependency.",
        "icon": "cloud",
        "priority": "low"
    },
    "electronic_check_payment": {
        "title": "Autopay Incentive",
        "description": "Electronic check users churn more. Offer a $5/month discount for switching to automatic credit card or bank transfer payments.",
        "icon": "credit-card",
        "priority": "medium"
    },
    "no_device_protection": {
        "title": "Device Protection Plan",
        "description": "Offer a complimentary device protection plan for 6 months to increase service stickiness.",
        "icon": "mobile",
        "priority": "low"
    },
    "paperless_billing": {
        "title": "Billing Communication Enhancement",
        "description": "Ensure clear, detailed billing statements. Send monthly usage summaries highlighting value received vs. cost.",
        "icon": "envelope",
        "priority": "low"
    },
    "senior_citizen": {
        "title": "Senior-Friendly Support Program",
        "description": "Provide dedicated senior support line, simplified billing, and in-home setup assistance.",
        "icon": "heart",
        "priority": "medium"
    },
    "multiple_streaming": {
        "title": "Entertainment Loyalty Bonus",
        "description": "Customer uses streaming services. Offer exclusive content access or a partner streaming service subscription at no extra cost.",
        "icon": "tv",
        "priority": "low"
    }
}


def build_feature_vector(data: dict) -> pd.DataFrame:
    """
    Take raw customer data (from the prediction form) and build a
    feature vector matching the trained model's expected input.
    """
    # Map form fields to dataframe columns
    row = {
        "Gender": data.get("gender", "Male"),
        "Senior Citizen": 1 if data.get("senior_citizen", "No") == "Yes" else 0,
        "Partner": data.get("partner", "No"),
        "Dependents": data.get("dependents", "No"),
        "Tenure Months": int(data.get("tenure_months", 1)),
        "Phone Service": data.get("phone_service", "Yes"),
        "Multiple Lines": data.get("multiple_lines", "No"),
        "Internet Service": data.get("internet_service", "No"),
        "Online Security": data.get("online_security", "No"),
        "Online Backup": data.get("online_backup", "No"),
        "Device Protection": data.get("device_protection", "No"),
        "Tech Support": data.get("tech_support", "No"),
        "Streaming TV": data.get("streaming_tv", "No"),
        "Streaming Movies": data.get("streaming_movies", "No"),
        "Contract": data.get("contract", "Month-to-month"),
        "Paperless Billing": data.get("paperless_billing", "Yes"),
        "Payment Method": data.get("payment_method", "Electronic check"),
        "Monthly Charges": float(data.get("monthly_charges", 50)),
        "Total Charges": float(data.get("total_charges", 50)),
    }

    df = pd.DataFrame([row])

    # ── Feature engineering (must match train_model.py exactly) ──
    # Tenure buckets
    df["Tenure_Bucket"] = pd.cut(
        df["Tenure Months"],
        bins=[0, 12, 24, 48, 72, 200],
        labels=["0-12", "13-24", "25-48", "49-72", "72+"],
    )

    # Avg monthly charge relative to tenure
    df["Avg_Charge_Per_Month"] = np.where(
        df["Tenure Months"] > 0,
        df["Total Charges"] / df["Tenure Months"],
        df["Monthly Charges"],
    )

    # Service flags
    service_cols = [
        "Phone Service", "Multiple Lines", "Online Security", "Online Backup",
        "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    ]
    for col in service_cols:
        if col in df.columns:
            df[col + "_flag"] = df[col].apply(lambda x: 1 if x == "Yes" else 0)

    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    df["Service_Count"] = df[flag_cols].sum(axis=1)

    # Has internet
    if "Internet Service" in df.columns:
        df["Has_Internet"] = df["Internet Service"].apply(lambda x: 0 if x == "No" else 1)

    # Charge per service
    df["Charge_Per_Service"] = np.where(
        df["Service_Count"] > 0,
        df["Monthly Charges"] / df["Service_Count"],
        df["Monthly Charges"],
    )

    # Encode categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
        else:
            df[col] = 0

    # Ensure columns match model's expectations
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # Scale
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = scaler.transform(df[num_cols])

    df = df.fillna(0)

    return df


def get_retention_strategies(data: dict, churn_prob: float) -> list:
    """Determine personalized retention strategies based on customer profile."""
    strategies = []

    monthly_charges = float(data.get("monthly_charges", 0))
    avg_monthly = dataset_stats.get("avg_monthly_charges", 64)

    if monthly_charges > avg_monthly * 1.1:
        strategies.append(RETENTION_STRATEGIES["high_monthly_charges"])

    if data.get("contract", "") == "Month-to-month":
        strategies.append(RETENTION_STRATEGIES["month_to_month_contract"])

    tenure = int(data.get("tenure_months", 0))
    if tenure < 12:
        strategies.append(RETENTION_STRATEGIES["low_tenure"])

    if data.get("tech_support", "No") == "No" and data.get("internet_service", "No") != "No":
        strategies.append(RETENTION_STRATEGIES["no_tech_support"])

    if data.get("online_security", "No") == "No" and data.get("internet_service", "No") != "No":
        strategies.append(RETENTION_STRATEGIES["no_online_security"])

    if data.get("internet_service", "") == "Fiber optic":
        strategies.append(RETENTION_STRATEGIES["fiber_optic_user"])

    if data.get("online_backup", "No") == "No" and data.get("internet_service", "No") != "No":
        strategies.append(RETENTION_STRATEGIES["no_online_backup"])

    if data.get("payment_method", "") == "Electronic check":
        strategies.append(RETENTION_STRATEGIES["electronic_check_payment"])

    if data.get("device_protection", "No") == "No" and data.get("internet_service", "No") != "No":
        strategies.append(RETENTION_STRATEGIES["no_device_protection"])

    if data.get("senior_citizen", "No") == "Yes":
        strategies.append(RETENTION_STRATEGIES["senior_citizen"])

    if data.get("streaming_tv", "No") == "Yes" and data.get("streaming_movies", "No") == "Yes":
        strategies.append(RETENTION_STRATEGIES["multiple_streaming"])

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    strategies.sort(key=lambda s: priority_order.get(s["priority"], 3))

    return strategies[:5]  # Top 5 strategies


def get_risk_level(prob: float) -> dict:
    """Classify churn probability into risk levels."""
    if prob >= 0.75:
        return {"level": "Critical", "color": "#ff2d55", "action": "Immediate intervention required"}
    elif prob >= 0.50:
        return {"level": "High", "color": "#ff9500", "action": "Proactive retention outreach needed"}
    elif prob >= 0.25:
        return {"level": "Medium", "color": "#ffcc00", "action": "Monitor closely and engage periodically"}
    else:
        return {"level": "Low", "color": "#34c759", "action": "Standard engagement is sufficient"}


def get_contributing_factors(data: dict) -> list:
    """Identify top contributing factors for the churn prediction."""
    factors = []

    if data.get("contract", "") == "Month-to-month":
        factors.append({"factor": "Month-to-month contract", "impact": "high", "description": "Short-term contracts have 42% churn rate"})

    tenure = int(data.get("tenure_months", 0))
    if tenure < 12:
        factors.append({"factor": f"Low tenure ({tenure} months)", "impact": "high", "description": "New customers (<12 months) are 3x more likely to churn"})

    monthly = float(data.get("monthly_charges", 0))
    if monthly > 70:
        factors.append({"factor": f"High monthly charges (${monthly:.2f})", "impact": "medium", "description": "Above-average charges correlate with higher churn"})

    if data.get("internet_service", "") == "Fiber optic":
        factors.append({"factor": "Fiber optic service", "impact": "medium", "description": "Fiber optic users show 41.9% churn vs 18.9% for DSL"})

    if data.get("tech_support", "No") == "No":
        factors.append({"factor": "No tech support", "impact": "medium", "description": "Customers without tech support are more likely to churn"})

    if data.get("online_security", "No") == "No":
        factors.append({"factor": "No online security", "impact": "medium", "description": "Missing security services reduce perceived value"})

    if data.get("payment_method", "") == "Electronic check":
        factors.append({"factor": "Electronic check payment", "impact": "medium", "description": "Electronic check users have the highest churn rate by payment method"})

    if data.get("paperless_billing", "") == "Yes":
        factors.append({"factor": "Paperless billing", "impact": "low", "description": "Paperless billing slightly correlates with churn"})

    if data.get("senior_citizen", "No") == "Yes":
        factors.append({"factor": "Senior citizen", "impact": "medium", "description": "Senior citizens have a 41.7% churn rate vs 23.6% for others"})

    if data.get("partner", "No") == "No" and data.get("dependents", "No") == "No":
        factors.append({"factor": "No partner or dependents", "impact": "low", "description": "Single customers without dependents are more likely to switch providers"})

    impact_order = {"high": 0, "medium": 1, "low": 2}
    factors.sort(key=lambda f: impact_order.get(f["impact"], 3))

    return factors[:5]


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict churn for a single customer."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        feature_vector = build_feature_vector(data)
        proba = model.predict_proba(feature_vector)[0]
        churn_prob = float(proba[1])

        risk = get_risk_level(churn_prob)
        strategies = get_retention_strategies(data, churn_prob)
        factors = get_contributing_factors(data)

        result = {
            "churn_probability": round(churn_prob * 100, 1),
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "risk_action": risk["action"],
            "contributing_factors": factors,
            "retention_strategies": strategies,
            "prediction": "Will Churn" if churn_prob >= 0.5 else "Will Stay",
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    """Return dataset statistics and model metrics."""
    return jsonify({
        "dataset": dataset_stats,
        "model": model_metrics,
    })


@app.route("/api/customers")
def customers():
    """Return paginated customer list with churn info."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    search = request.args.get("search", "", type=str)
    sort_by = request.args.get("sort_by", "Churn Score", type=str)
    sort_dir = request.args.get("sort_dir", "desc", type=str)

    df = df_raw.copy()

    # Search
    if search:
        mask = df.apply(lambda row: search.lower() in str(row).lower(), axis=1)
        df = df[mask]

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_dir == "asc"))

    total = len(df)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    end = start + per_page

    page_df = df.iloc[start:end]

    customers_list = []
    for _, row in page_df.iterrows():
        customers_list.append({
            "customer_id": row.get("CustomerID", ""),
            "gender": row.get("Gender", ""),
            "senior_citizen": row.get("Senior Citizen", "No"),
            "partner": row.get("Partner", "No"),
            "dependents": row.get("Dependents", "No"),
            "tenure_months": int(row.get("Tenure Months", 0)),
            "contract": row.get("Contract", ""),
            "monthly_charges": float(row.get("Monthly Charges", 0)),
            "total_charges": str(row.get("Total Charges", "")),
            "churn_label": row.get("Churn Label", ""),
            "churn_score": int(row.get("Churn Score", 0)),
            "internet_service": row.get("Internet Service", ""),
            "payment_method": row.get("Payment Method", ""),
            "phone_service": row.get("Phone Service", ""),
        })

    return jsonify({
        "customers": customers_list,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
    })


@app.route("/api/customer/<customer_id>")
def customer_detail(customer_id):
    """Return detailed customer profile with prediction."""
    row = df_raw[df_raw["CustomerID"] == customer_id]
    if row.empty:
        return jsonify({"error": "Customer not found"}), 404

    row = row.iloc[0]
    data = {
        "gender": row.get("Gender", "Male"),
        "senior_citizen": row.get("Senior Citizen", "No"),
        "partner": row.get("Partner", "No"),
        "dependents": row.get("Dependents", "No"),
        "tenure_months": str(int(row.get("Tenure Months", 1))),
        "phone_service": row.get("Phone Service", "Yes"),
        "multiple_lines": row.get("Multiple Lines", "No"),
        "internet_service": row.get("Internet Service", "No"),
        "online_security": row.get("Online Security", "No"),
        "online_backup": row.get("Online Backup", "No"),
        "device_protection": row.get("Device Protection", "No"),
        "tech_support": row.get("Tech Support", "No"),
        "streaming_tv": row.get("Streaming TV", "No"),
        "streaming_movies": row.get("Streaming Movies", "No"),
        "contract": row.get("Contract", "Month-to-month"),
        "paperless_billing": row.get("Paperless Billing", "Yes"),
        "payment_method": row.get("Payment Method", "Electronic check"),
        "monthly_charges": str(float(row.get("Monthly Charges", 50))),
        "total_charges": str(row.get("Total Charges", "50")),
    }

    # Run prediction
    feature_vector = build_feature_vector(data)
    proba = model.predict_proba(feature_vector)[0]
    churn_prob = float(proba[1])

    risk = get_risk_level(churn_prob)
    strategies = get_retention_strategies(data, churn_prob)
    factors = get_contributing_factors(data)

    return jsonify({
        "customer": {
            "customer_id": row.get("CustomerID", ""),
            "city": row.get("City", ""),
            "state": row.get("State", ""),
            "churn_label": row.get("Churn Label", ""),
            "churn_reason": row.get("Churn Reason", ""),
            "cltv": int(row.get("CLTV", 0)),
            **data,
        },
        "prediction": {
            "churn_probability": round(churn_prob * 100, 1),
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "risk_action": risk["action"],
            "contributing_factors": factors,
            "retention_strategies": strategies,
        }
    })


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """Accept CSV upload and return batch predictions."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "File must be a CSV"}), 400

        df_upload = pd.read_csv(file)
        results = []

        for _, row in df_upload.iterrows():
            data = {
                "gender": row.get("Gender", "Male"),
                "senior_citizen": str(row.get("Senior Citizen", "No")),
                "partner": row.get("Partner", "No"),
                "dependents": row.get("Dependents", "No"),
                "tenure_months": str(int(row.get("Tenure Months", 1))),
                "phone_service": row.get("Phone Service", "Yes"),
                "multiple_lines": row.get("Multiple Lines", "No"),
                "internet_service": row.get("Internet Service", "No"),
                "online_security": row.get("Online Security", "No"),
                "online_backup": row.get("Online Backup", "No"),
                "device_protection": row.get("Device Protection", "No"),
                "tech_support": row.get("Tech Support", "No"),
                "streaming_tv": row.get("Streaming TV", "No"),
                "streaming_movies": row.get("Streaming Movies", "No"),
                "contract": row.get("Contract", "Month-to-month"),
                "paperless_billing": row.get("Paperless Billing", "Yes"),
                "payment_method": row.get("Payment Method", "Electronic check"),
                "monthly_charges": str(float(row.get("Monthly Charges", 50))),
                "total_charges": str(row.get("Total Charges", "50")),
            }

            try:
                feature_vector = build_feature_vector(data)
                proba = model.predict_proba(feature_vector)[0]
                churn_prob = float(proba[1])
                risk = get_risk_level(churn_prob)

                results.append({
                    "customer_id": row.get("CustomerID", "N/A"),
                    "churn_probability": round(churn_prob * 100, 1),
                    "risk_level": risk["level"],
                    "prediction": "Will Churn" if churn_prob >= 0.5 else "Will Stay",
                })
            except Exception:
                results.append({
                    "customer_id": row.get("CustomerID", "N/A"),
                    "churn_probability": None,
                    "risk_level": "Error",
                    "prediction": "Error processing",
                })

        # Summary
        valid = [r for r in results if r["churn_probability"] is not None]
        summary = {
            "total_processed": len(results),
            "predicted_churn": len([r for r in valid if r["churn_probability"] >= 50]),
            "predicted_stay": len([r for r in valid if r["churn_probability"] < 50]),
            "avg_churn_probability": round(np.mean([r["churn_probability"] for r in valid]), 1) if valid else 0,
            "errors": len(results) - len(valid),
        }

        return jsonify({
            "results": results,
            "summary": summary,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Customer Churn Prediction - Server")
    print("=" * 60)
    print(f"  Model: {model_metrics.get('best_model', 'Unknown')}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Dataset: {dataset_stats.get('total_customers', 0)} customers")
    print(f"  Churn Rate: {dataset_stats.get('churn_rate', 0)}%")
    print("=" * 60 + "\n")

    app.run(debug=True, host="127.0.0.1", port=5000)
