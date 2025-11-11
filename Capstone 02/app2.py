import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ===========================
# Load trained model pipeline
# ===========================
MODEL_PATH = Path("output/final_model_pipeline.pkl")

st.set_page_config(page_title="Heart Disease Detection", page_icon="💓", layout="centered")

st.title("💓 Heart Disease Detection System")
st.markdown("""
This app uses a **Machine Learning model** trained on clinical and demographic data  
to predict the **likelihood of heart disease**.
""")

# Load the model pipeline
if not MODEL_PATH.exists():
    st.error("❌ Model file not found. Please ensure `final_model_pipeline.pkl` exists in the output folder.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.success("✅ Model loaded successfully!")

# ===========================
# User Input Section
# ===========================

st.header("🧍‍♂️ Patient Information")

# Demographic info
age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["Female", "Male"])

# Clinical measurements
st.header("🩺 Clinical Measurements")
chest_pain_type = st.selectbox(
    "Chest Pain Type",
    ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
)
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", min_value=70, max_value=250, value=120)
cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

# Diagnostic test results
st.header("📊 Diagnostic Test Results")
resting_ecg = st.selectbox("Resting ECG Results", [0, 1, 2])
max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
st_depression = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0)
st_slope = st.selectbox("ST Slope", [0, 1, 2])
num_major_vessels = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
thalassemia = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# ===========================
# Data Preprocessing (match training)
# ===========================

sex = 1 if sex == "Male" else 0
fasting_blood_sugar = 1 if fasting_blood_sugar == "Yes" else 0
exercise_induced_angina = 1 if exercise_induced_angina == "Yes" else 0

# Encode chest pain as numeric (consistent with dataset: 0–3)
cp_map = {
    "Typical angina": 0,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "Asymptomatic": 3
}
chest_pain_type = cp_map[chest_pain_type]

# Construct feature vector
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'chest_pain_type': chest_pain_type,
    'resting_blood_pressure': resting_blood_pressure,
    'cholesterol': cholesterol,
    'fasting_blood_sugar': fasting_blood_sugar,
    'resting_ecg': resting_ecg,
    'max_heart_rate': max_heart_rate,
    'exercise_induced_angina': exercise_induced_angina,
    'st_depression': st_depression,
    'st_slope': st_slope,
    'num_major_vessels': num_major_vessels,
    'thalassemia': thalassemia
}])

st.divider()

# ===========================
# Prediction
# ===========================
if st.button("🔍 Predict Heart Disease"):
    with st.spinner("Analyzing patient data..."):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100  # Probability of class=1

    st.subheader("🧠 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High risk of **heart disease** detected. (Probability: {probability:.2f}%)")
    else:
        st.success(f"✅ Low risk of heart disease. (Probability: {probability:.2f}%)")

# ===========================
# Optional: Batch prediction
# ===========================
st.divider()
st.header("📁 Batch Prediction (Upload CSV)")

st.markdown("Upload a CSV file with the same feature columns as used in training.")

uploaded_file = st.file_uploader("Upload your patient dataset (.csv)", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        preds = model.predict(data)
        probs = model.predict_proba(data)[:, 1]

        data['HeartDiseasePrediction'] = preds
        data['Probability(%)'] = (probs * 100).round(2)

        st.success("✅ Predictions generated successfully!")
        st.dataframe(data.head())

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", data=csv, file_name="heart_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
