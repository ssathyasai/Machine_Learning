import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Anomaly Detection System", layout="wide")

st.title("🔍 Anomaly Detection System")
st.markdown("This application uses Isolation Forest to detect anomalies in air quality data.")
st.markdown("---")

# Load model safely
model_path = Path("isolation_forest_model.pkl")

if not model_path.exists():
    st.error("❌ Model file not found. Please retrain model.")
    st.stop()

model = joblib.load(model_path)

st.success("✅ Model Loaded Successfully")

# Debug: Show trained features
st.write("Model expects features:", model.feature_names_in_)

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---------------- SINGLE PREDICTION ----------------
with tab1:

    st.subheader("Single Data Point Prediction")

    co_gt = st.number_input("CO(GT)", min_value=0.0, value=2.0)
    c6h6_gt = st.number_input("C6H6(GT)", min_value=0.0, value=10.0)
    nox_gt = st.number_input("NOx(GT)", min_value=0.0, value=150.0)
    no2_gt = st.number_input("NO2(GT)", min_value=0.0, value=100.0)

    if st.button("🔮 Predict"):

        input_data = pd.DataFrame({
            'CO(GT)': [co_gt],
            'C6H6(GT)': [c6h6_gt],
            'NOx(GT)': [nox_gt],
            'NO2(GT)': [no2_gt]
        })

        prediction = model.predict(input_data)[0]
        score = model.score_samples(input_data)[0]

        if prediction == -1:
            st.error("⚠️ ANOMALY DETECTED")
        else:
            st.success("✅ NORMAL")

        st.metric("Anomaly Score", f"{score:.4f}")

# ---------------- BATCH PREDICTION ----------------
with tab2:

    uploaded_file = st.file_uploader("Upload CSV with required columns", type="csv")

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        required = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

        if not all(col in df.columns for col in required):
            st.error(f"CSV must contain: {required}")
        else:
            df = df[required]

            predictions = model.predict(df)
            scores = model.score_samples(df)

            df['Prediction'] = predictions
            df['Score'] = scores

            st.dataframe(df)
