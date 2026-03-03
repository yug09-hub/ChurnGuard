# Churn Guard AI

A customer churn prediction system built with Python and Flask. It uses machine learning to figure out which customers are likely to leave, and suggests ways to keep them.

I built this to solve a real business problem — telecom companies lose a lot of revenue when customers churn, and most of the time they don't see it coming until it's too late. This tool helps catch those signals early.

## What it does

- **Predicts churn** — Enter a customer's details (contract type, tenure, services, charges, etc.) and the model tells you how likely they are to leave, with a risk level from Low to Critical
- **Shows why** — Breaks down the top contributing factors behind each prediction, so you know *what's* driving the risk
- **Suggests retention strategies** — Based on the customer's profile, it recommends specific actions (discount offers, support upgrades, contract incentives, etc.)
- **Batch predictions** — Upload a CSV of customers and get predictions for all of them at once
- **Customer explorer** — Browse through the full dataset with search, sort, and pagination
- **Dashboard analytics** — Visual stats on churn rates by contract type, payment method, tenure, internet service, and more

## Tech stack

- **Backend** — Python, Flask
- **ML** — scikit-learn (Random Forest, Gradient Boosting)
- **Data processing** — pandas, NumPy
- **Frontend** — HTML, CSS, JavaScript (vanilla, no framework)

## How the model works

The training pipeline (`train_model.py`) does the following:

1. Loads the Telco Customer Churn dataset (~7,000 customers)
2. Cleans and preprocesses the data — handles missing values, encodes categoricals, etc.
3. Engineers extra features like tenure buckets, average charge per month, service count, charge-per-service ratio
4. Trains both Random Forest and Gradient Boosting classifiers with 5-fold stratified cross-validation
5. Picks the best model based on CV F1 score and saves it along with the encoders and scaler

The best model ended up being **Random Forest** with:
- Accuracy: ~78%
- ROC AUC: ~0.85
- F1 Score: ~0.64 (cross-validated)

Top features that matter most: contract type, charge per service, tenure, total charges, and monthly charges.

## Project structure

```
├── app.py                  # Flask API server + all routes
├── train_model.py          # ML training pipeline
├── Telco_customer_churn.csv  # Dataset
├── model.pkl               # Trained model
├── label_encoders.pkl      # Label encoders for categorical features
├── scaler.pkl              # StandardScaler for numeric features
├── feature_columns.json    # Feature column names (model expects these)
├── model_metrics.json      # Evaluation metrics for all models
├── dataset_stats.json      # Pre-computed dataset statistics
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Main dashboard page
└── static/
    ├── css/style.css        # Styling
    └── js/app.js            # Frontend logic
```

## Setup

1. Clone the repo:

```bash
git clone https://github.com/yug09-hub/Churn-Guard-AI.git
cd Churn-Guard-AI
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Retrain the model:

```bash
python train_model.py
```

This will regenerate `model.pkl`, `scaler.pkl`, `label_encoders.pkl`, and the JSON config files. You only need to do this if you want to train from scratch — the pre-trained model is already included.

4. Run the app:

```bash
python app.py
```

5. Open `http://127.0.0.1:5000` in your browser.

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main dashboard |
| `/api/predict` | POST | Single customer churn prediction |
| `/api/batch-predict` | POST | Batch CSV upload + predictions |
| `/api/stats` | GET | Dataset stats and model metrics |
| `/api/customers` | GET | Paginated customer list |
| `/api/customer/<id>` | GET | Detailed customer profile + prediction |

## Dataset

Uses the [IBM Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset. It has ~7,000 customer records with 33 columns including demographics, account info, services subscribed, and churn status.

Key stats from the data:
- Overall churn rate: 26.5%
- Month-to-month contracts: 42.7% churn rate
- Two-year contracts: 2.8% churn rate
- Fiber optic users: 41.9% churn rate
- Electronic check payers: 45.3% churn rate

## License

MIT
