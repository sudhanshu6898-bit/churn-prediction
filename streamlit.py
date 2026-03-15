# app.py
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_bundle(bundle_path: str):
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Could not find model bundle at: {bundle_path}")
    bundle = joblib.load(bundle_path)
    # Expected keys: "model", "threshold" (as described in your README)
    model = bundle.get("model", None)
    threshold = bundle.get("threshold", None)
    if model is None:
        raise ValueError("Bundle is missing 'model'.")
    if threshold is None:
        # fallback if threshold not in bundle
        threshold = 0.5
    return model, threshold, bundle

def predict_single(model, df_one_row: pd.DataFrame) -> float:
    # Model is assumed to include preprocessing pipeline internally or wrap it.
    proba = model.predict_proba(df_one_row)[:, 1][0]
    return float(proba)

def predict_batch(model, df: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(df)[:, 1]
    return proba

def coerce_telco_types(row: dict) -> dict:
    """Best-effort type coercion where common Telco fields appear as strings."""
    out = dict(row)
    # Numeric candidates
    for k in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if k in out and out[k] is not None and out[k] != "":
            try:
                if k == "tenure":
                    out[k] = int(float(out[k]))
                else:
                    out[k] = float(out[k])
            except Exception:
                pass
    # SeniorCitizen may be 0/1 or Yes/No; standardize to int 0/1 if possible
    if "SeniorCitizen" in out:
        v = str(out["SeniorCitizen"]).strip()
        if v.lower() in ["yes", "y", "1", "true"]:
            out["SeniorCitizen"] = 1
        elif v.lower() in ["no", "n", "0", "false"]:
            out["SeniorCitizen"] = 0
        else:
            try:
                out["SeniorCitizen"] = int(float(v))
            except Exception:
                pass
    return out

def telco_default_sample():
    # Reasonable defaults for Telco dataset-style inputs
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 70.35,
    }

# -----------------------------
# Sidebar - Model Loading
# -----------------------------
st.sidebar.title("⚙️ Configuration")

model_source = st.sidebar.radio(
    "Load Model Bundle",
    options=["Use local file path", "Upload .joblib"],
    index=0,
)

bundle_path = None
uploaded_bundle = None

if model_source == "Use local file path":
    bundle_path = st.sidebar.text_input(
        "Bundle path",
        value="churn_rf_bundle.joblib",
        help="Path to your joblib bundle with {'model','threshold'}."
    )
else:
    uploaded_bundle = st.sidebar.file_uploader(
        "Upload churn_rf_bundle.joblib",
        type=["joblib"],
        accept_multiple_files=False,
    )

with st.sidebar:
    st.markdown("---")
    st.caption("Tip: The bundle should include pre-processing and a tuned threshold.")

# Try load model
model, default_threshold, _bundle_obj = None, 0.5, None
load_clicked = st.sidebar.button("🔄 Load / Reload Model", use_container_width=True)

if load_clicked:
    try:
        if uploaded_bundle is not None:
            # Read into bytes buffer then joblib.load from buffer
            bytes_buf = io.BytesIO(uploaded_bundle.read())
            _bundle_obj = joblib.load(bytes_buf)
            model = _bundle_obj["model"]
            default_threshold = _bundle_obj.get("threshold", 0.5)
        else:
            model, default_threshold, _bundle_obj = load_bundle(bundle_path)
        st.sidebar.success("Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

# Persist load state across reruns
if "model" not in st.session_state and model is not None:
    st.session_state["model"] = model
    st.session_state["threshold"] = default_threshold
elif load_clicked and model is not None:
    st.session_state["model"] = model
    st.session_state["threshold"] = default_threshold

# If previously loaded, retrieve
model = st.session_state.get("model", None)
if "threshold" not in st.session_state:
    st.session_state["threshold"] = default_threshold

# -----------------------------
# Header
# -----------------------------
st.title("📉 Customer Churn Prediction")
st.markdown(
    """
Use your saved RandomForest **churn bundle** (model + preprocessing + tuned threshold) to score customers.

- Fill the **Single Customer** form or upload a **CSV** for batch predictions.
- The app uses `model.predict_proba(... )` and the **bundle threshold** to generate predictions.
"""
)

# -----------------------------
# Tabs
# -----------------------------
tab_single, tab_batch, tab_debug = st.tabs(["🔍 Single Customer", "📦 Batch Scoring (CSV)", "🧪 Debug / Advanced"])

# -----------------------------
# Single Customer Tab
# -----------------------------
with tab_single:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    # default values
    defaults = telco_default_sample()

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        senior = st.selectbox("SeniorCitizen", ["0", "1", "Yes", "No"], index=3)
        partner = st.selectbox("Partner", ["Yes", "No"], index=1)
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=1)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=defaults["tenure"])
        phone_service = st.selectbox("PhoneService", ["Yes", "No"], index=0)

    with col2:
        multiple_lines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"], index=0)
        internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"], index=1)
        online_security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"], index=0)
        online_backup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"], index=0)
        device_protection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"], index=0)
        tech_support = st.selectbox("TechSupport", ["No", "Yes", "No internet service"], index=0)

    with col3:
        streaming_tv = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"], index=0)
        streaming_movies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"], index=0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
        paperless = st.selectbox("PaperlessBilling", ["Yes", "No"], index=0)
        payment_method = st.selectbox(
            "PaymentMethod",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=0
        )
        monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, max_value=1000.0, value=float(defaults["MonthlyCharges"]), step=0.05)
        total_charges = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=float(defaults["TotalCharges"]), step=0.05)

    # Compose sample row
    sample = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Advanced JSON override
    with st.expander("🔧 Advanced: Paste a raw JSON row (override form values)"):
        raw_json = st.text_area(
            "One JSON object with feature: value pairs matching your model’s training schema.",
            height=120,
            placeholder=json.dumps(defaults, indent=2)
        )
        use_json = st.checkbox("Use JSON instead of form", value=False)
        if use_json and raw_json.strip():
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, dict):
                    sample = parsed
                else:
                    st.warning("JSON must be a single object (not an array). Using form values instead.")
            except Exception as e:
                st.error(f"Invalid JSON. Using form values instead. Error: {e}")

    left, right = st.columns([1, 2])

    with left:
        thresh = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("threshold", 0.5)),
            step=0.01,
            help="Override tuned threshold from bundle for experimentation."
        )
        run_pred = st.button("🚀 Predict", type="primary")

    with right:
        st.caption("The tuned threshold from the bundle is shown as default. Adjust if you like.")

    if run_pred:
        if model is None:
            st.error("Please load the model bundle from the sidebar first.")
        else:
            # Coerce types & predict
            sample = coerce_telco_types(sample)
            df_one = pd.DataFrame([sample])
            try:
                proba = predict_single(model, df_one)
                pred = int(proba >= thresh)
                st.success("Prediction completed.")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Churn Probability", f"{proba:.3f}")
                with c2:
                    st.metric("Decision Threshold", f"{thresh:.2f}")
                with c3:
                    st.metric("Predicted Label", "Churn" if pred == 1 else "No Churn")

                st.progress(min(max(proba, 0.0), 1.0), text="Probability scale")
                st.json(sample)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------------
# Batch Scoring Tab
# -----------------------------
with tab_batch:
    st.subheader("Upload CSV for Batch Inference")
    st.markdown(
        """
**Instructions**
- CSV should contain one row per customer with the **same feature names** used during training.
- No target column is necessary.
"""
    )
    batch_file = st.file_uploader("Upload CSV", type=["csv"])
    run_batch = st.button("📦 Run Batch Predictions")
    batch_threshold = st.slider("Batch Decision Threshold", 0.0, 1.0, float(st.session_state.get("threshold", 0.5)), 0.01)

    if run_batch:
        if model is None:
            st.error("Please load the model bundle from the sidebar first.")
        elif batch_file is None:
            st.error("Please upload a CSV first.")
        else:
            try:
                df = pd.read_csv(batch_file)
                # Best-effort coercion for common numeric/boolean columns
                df_coerced = df.apply(lambda row: pd.Series(coerce_telco_types(row.to_dict())), axis=1)
                # Ensure original columns order preserved when possible
                df_coerced = df_coerced[df.columns] if set(df.columns) <= set(df_coerced.columns) else df_coerced

                probs = predict_batch(model, df_coerced)
                preds = (probs >= batch_threshold).astype(int)

                out = df.copy()
                out["churn_probability"] = probs
                out["prediction"] = preds

                st.success(f"Scored {len(out)} rows.")
                st.dataframe(out.head(50))

                # Download
                csv_buf = io.StringIO()
                out.to_csv(csv_buf, index=False)
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_buf.getvalue(),
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# -----------------------------
# Debug / Advanced Tab
# -----------------------------
with tab_debug:
    st.subheader("Model & Explainability (Lightweight)")

    if model is None:
        st.info("Load a model bundle first to inspect.")
    else:
        st.markdown("**Model Class**")
        st.code(type(model).__name__)

        st.markdown("**Tuned Threshold (from bundle)**")
        st.code(st.session_state.get("threshold", 0.5))

        st.markdown("**Permutation Importance (quick, on synthetic samples)**")
        st.caption(
            "This runs a small permutation importance using synthetic samples generated from current single-customer defaults. "
            "For robust explanations, use your validation set."
        )

        # Try a crude local permutation importance using small samples around defaults
        try:
            base = telco_default_sample()
            # create a tiny dataset by varying some fields
            toys = []
            for c in ["Contract", "InternetService", "PaperlessBilling", "MonthlyCharges", "tenure"]:
                b = base.copy()
                if c == "Contract":
                    b[c] = "Two year"
                elif c == "InternetService":
                    b[c] = "DSL"
                elif c == "PaperlessBilling":
                    b[c] = "No"
                elif c == "MonthlyCharges":
                    b[c] = base["MonthlyCharges"] + 20
                elif c == "tenure":
                    b[c] = base["tenure"] + 12
                toys.append(b)
            X_toy = pd.DataFrame([base] + toys)
            X_toy = X_toy.apply(lambda row: pd.Series(coerce_telco_types(row.to_dict())), axis=1)

            # Simple permutation importance wrapper that expects predict_proba
            def proba_1(X):
                return model.predict_proba(X)[:, 1]

            # Use a small n_repeats to keep it snappy
            pi = permutation_importance(
                estimator=model,
                X=X_toy,
                y=np.array([0] * len(X_toy)),  # dummy (not used by proba_1), but required by signature
                scoring=None,
                n_repeats=3,
                random_state=42
            )
            imp_df = pd.DataFrame({
                "feature": X_toy.columns,
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std
            }).sort_values("importance_mean", ascending=False)

            st.dataframe(imp_df.head(15), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute permutation importance here: {e}")

st.markdown("---")
st.caption("Built with Streamlit. Remember to align form field names with the features used in your training pipeline.")
