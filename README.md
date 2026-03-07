# 📉 Customer Churn Prediction (Machine Learning)

This project uses machine learning to predict customer churn using the Telco Customer dataset.

## 📦 Project Files
- `model_training.ipynb` — Complete training notebook (data cleaning, EDA, model building, threshold tuning, evaluation)
- `churn_rf_bundle.joblib` — Saved RandomForest model + preprocessing pipeline + tuned threshold

## 🔧 How to Use
1. Install dependencies:

pip install pandas numpy scikit-learn joblib

2. Load the model:
```python
import joblib
model_bundle = joblib.load("churn_rf_bundle.joblib")
model = model_bundle["model"]
threshold = model_bundle["threshold"]

predict on a new customer 
import pandas as pd

sample = { ... customer feature dictionary ... }
df = pd.DataFrame([sample])
prob = model.predict_proba(df)[:,1][0]
pred = int(prob >= threshold)

print(prob, pred)
