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
# LOAD MODEL
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "xgboost_cardio_model.pkl"   # model file at root
)

model = joblib.load(MODEL_PATH)

# ===============================
# FEATURE ENGINEERING
# ===============================
def prepare_features(df):
    df["age"] = df["age_years"]

    # BMI
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    if df["bmi"].iloc[0] < 18.5:
        df["bmi_category"] = 0
    elif df["bmi"].iloc[0] < 25:
        df["bmi_category"] = 1
    elif df["bmi"].iloc[0] < 30:
        df["bmi_category"] = 2
    else:
        df["bmi_category"] = 3

    # Blood pressure
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["map_pressure"] = df["ap_lo"] + (df["pulse_pressure"] / 3)

    if df["ap_hi"].iloc[0] >= 140 or df["ap_lo"].iloc[0] >= 90:
        df["bp_category"] = 2
    elif df["ap_hi"].iloc[0] >= 120:
        df["bp_category"] = 1
    else:
        df["bp_category"] = 0

    # Lifestyle and metabolic risk
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

    final_cols = [
        'age','gender','height','weight','ap_hi','ap_lo',
        'cholesterol','gluc','smoke','alco','active',
        'age_years','bmi','bmi_category','pulse_pressure',
        'map_pressure','bp_category','lifestyle_risk',
        'metabolic_risk','combined_risk'
    ]

    return df[final_cols]

# ===============================
# HEADER
# ===============================
st.title("Heart Disease Risk Predictor")
st.caption("AI-based Cardiovascular Risk Assessment System")

st.markdown("---")

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
    st.subheader("Patient Details")

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

    predict = st.button("Assess Risk")

# ===============================
# TAB 2: RISK ANALYSIS
# ===============================
with tab2:
    if predict:
        with st.spinner("Analyzing patient data, please wait..."):
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

            X = prepare_features(patient_df)
            risk_prob = model.predict_proba(X)[0][1]
            risk_percent = risk_prob * 100

        st.success("Risk analysis completed.")

        bmi = X["bmi"].iloc[0]
        bmi_text = (
            "Underweight" if bmi < 18.5 else
            "Normal" if bmi < 25 else
            "Overweight" if bmi < 30 else
            "Obese"
        )

        bp_text = (
            "Normal" if X["bp_category"].iloc[0] == 0 else
            "Elevated" if X["bp_category"].iloc[0] == 1 else
            "Hypertension"
        )

        st.subheader("Risk Assessment Result")
        st.write(f"Estimated heart disease risk: {risk_percent:.1f}%")

        if risk_prob >= 0.75:
            st.error("High risk detected. Medical consultation is advised.")
        elif risk_prob >= 0.45:
            st.warning("Moderate risk detected. Lifestyle changes are recommended.")
        else:
            st.success("Low risk detected.")

        st.subheader("Clinical Summary")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("BMI", f"{bmi:.1f}", bmi_text)

        with c2:
            st.metric("Blood Pressure", f"{ap_hi}/{ap_lo}", bp_text)

        with c3:
            st.metric("Lifestyle Risk Score", X["lifestyle_risk"].iloc[0])

        st.subheader("Risk Explanation")
        reasons = []

        if bp_text != "Normal":
            reasons.append("Elevated blood pressure increases cardiovascular strain.")
        if bmi >= 25:
            reasons.append("Excess body weight increases heart workload.")
        if smoke == "Yes":
            reasons.append("Smoking damages blood vessels.")
        if active == "No":
            reasons.append("Low physical activity affects heart health.")
        if age >= 55:
            reasons.append("Age increases cardiovascular risk.")

        for r in reasons:
            st.markdown(f"- {r}")

    else:
        st.info("Please enter patient details in the first tab and click 'Assess Risk'.")

# ===============================
# TAB 3: REPORT
# ===============================
with tab3:
    if predict:
        pdf_path = generate_patient_report(
            patient_df,
            risk_prob,
            "High Risk" if risk_prob >= 0.5 else "Low Risk"
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Clinical PDF Report",
                f,
                file_name="CardioPredict_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Generate risk analysis first to download the report.")

# ===============================
# TAB 4: MODEL INFO
# ===============================
with tab4:
    st.subheader("Model Information")
    st.write(
        """
        Dataset: Cardiovascular Disease Dataset (Kaggle)  
        Records: Approximately 70,000  
        Model: XGBoost Classifier  
        ROC-AUC: Approximately 0.80  
        Purpose: Screening and educational use
        """
    )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "AI-based cardiovascular screening tool\n"
    "Developed by S. Vishnu Vardhan Reddy"
)
