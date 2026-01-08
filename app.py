import os
import streamlit as st
import pandas as pd
import joblib

from pdf_report import generate_patient_report

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="wide"
)

# ===============================
# SESSION STATE
# ===============================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# ===============================
# LOAD MODEL
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_cardio_model.pkl")

model = joblib.load(MODEL_PATH)

# ===============================
# FEATURE ENGINEERING
# ===============================
def prepare_features(df):
    df["age"] = df["age_years"]

    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["map_pressure"] = df["ap_lo"] + (df["pulse_pressure"] / 3)

    if df["ap_hi"].iloc[0] >= 140 or df["ap_lo"].iloc[0] >= 90:
        df["bp_category"] = 2
    elif df["ap_hi"].iloc[0] >= 120:
        df["bp_category"] = 1
    else:
        df["bp_category"] = 0

    df["lifestyle_risk"] = (
        df["smoke"] * 3 +
        df["alco"] * 2 +
        (1 - df["active"]) * 2
    )

    df["metabolic_risk"] = (
        (df["cholesterol"] - 1) * 10 +
        (df["gluc"] - 1) * 10 +
        df["bp_category"] * 10
    )

    df["combined_risk"] = df["lifestyle_risk"] * 5 + df["metabolic_risk"]

    return df[
        [
            "age", "gender", "height", "weight",
            "ap_hi", "ap_lo", "cholesterol", "gluc",
            "smoke", "alco", "active",
            "age_years", "bmi", "bmi_category",
            "pulse_pressure", "map_pressure",
            "bp_category", "lifestyle_risk",
            "metabolic_risk", "combined_risk"
        ]
    ]

# ===============================
# HEADER
# ===============================
st.title("Heart Disease Risk Predictor")
st.caption("AI-based Cardiovascular Risk Assessment System")

st.info(
    "Enter patient details, click 'Assess Risk', "
    "and scroll down to view the results."
)

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Patient Details", "Risk Analysis", "Report", "Model Info"]
)

# ===============================
# TAB 1: PATIENT DETAILS
# ===============================
with tab1:
    st.subheader("Patient Information")

    age = st.number_input("Age (years)", 18, 100, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 140, 200, 170)
    weight = st.number_input("Weight (kg)", 40, 120, 70)
    ap_hi = st.number_input("Systolic BP (mmHg)", 90, 200, 140)
    ap_lo = st.number_input("Diastolic BP (mmHg)", 60, 130, 90)

    st.subheader("Lifestyle Factors")
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    alco = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    active = st.selectbox("Physically Active", ["Yes", "No"])

    if st.button("Assess Risk"):
        st.session_state.predicted = True

    if st.session_state.predicted:
        patient_df = pd.DataFrame([{
            "age_years": age,
            "gender": 1 if gender == "Male" else 0,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 1 if smoke == "Yes" else 0,
            "alco": 1 if alco == "Yes" else 0,
            "active": 1 if active == "Yes" else 0
        }])

        with st.spinner("Analyzing risk..."):
            X = prepare_features(patient_df)
            risk_prob = model.predict_proba(X)[0][1]

        st.subheader("Risk Result")
        st.write(f"Estimated Heart Disease Risk: {risk_prob * 100:.1f}%")

        if risk_prob >= 0.75:
            st.error("High risk detected. Medical consultation advised.")
        elif risk_prob >= 0.45:
            st.warning("Moderate risk detected. Lifestyle changes recommended.")
        else:
            st.success("Low cardiovascular risk detected.")

        st.caption("Open the Risk Analysis tab for explanation.")

# ===============================
# TAB 2: RISK ANALYSIS
# ===============================
with tab2:
    if st.session_state.predicted:
        st.subheader("Why this risk was predicted")

        bmi = X["bmi"].iloc[0]

        reasons = []
        if bmi >= 25:
            reasons.append("Elevated BMI increases heart workload.")
        if ap_hi >= 140 or ap_lo >= 90:
            reasons.append("High blood pressure strains blood vessels.")
        if smoke == "Yes":
            reasons.append("Smoking damages cardiovascular health.")
        if active == "No":
            reasons.append("Low physical activity weakens heart efficiency.")
        if age >= 55:
            reasons.append("Age increases cardiovascular vulnerability.")

        for r in reasons:
            st.write("- " + r)
    else:
        st.info("Please assess risk first.")

# ===============================
# TAB 3: REPORT
# ===============================
with tab3:
    if st.session_state.predicted:
        pdf_path = generate_patient_report(
            patient_df,
            risk_prob,
            "High Risk" if risk_prob >= 0.5 else "Low Risk"
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Clinical Report",
                f,
                file_name="CardioPredict_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Generate risk assessment first.")

# ===============================
# TAB 4: MODEL INFO
# ===============================
with tab4:
    st.subheader("Model Information")
    st.write("""
    - Dataset: Cardiovascular Disease Dataset (Kaggle)
    - Records: ~70,000
    - Model: XGBoost Classifier
    - ROC-AUC: ~0.80
    - Designed for screening and educational use
    """)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "Heart Disease Risk Predictor | Developed by "
    "S. Vishnu Vardhan Reddy"
)
