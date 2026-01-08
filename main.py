# ==============================
# CardioPredict – Heart Disease Risk Prediction
# Complete main.py (Step-by-step build)
# ==============================

import pandas as pd

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("data/cardio.csv", sep=";")

print("Dataset loaded successfully")
print("Shape of dataset:", df.shape)

# ==============================
# 2. Initial Inspection
# ==============================
print("\nColumn names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# ==============================
# 3. Age Conversion (days → years)
# ==============================
df["age_years"] = (df["age"] / 365.25).astype(int)

print("\nAge conversion check:")
print(df[["age", "age_years"]].head())

# ==============================
# 4. Remove ID column
# ==============================
df.drop(columns=["id"], inplace=True)

print("\nColumns after removing id:")
print(df.columns)

# ==============================
# 5. Gender Encoding (1=female,2=male → 0=female,1=male)
# ==============================
df["gender"] = df["gender"].map({1: 0, 2: 1})

print("\nGender encoding check:")
print(df["gender"].value_counts())

# ==============================
# 6. BMI Calculation
# ==============================
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2

print("\nBMI check:")
print(df["bmi"].head())

# ==============================
# 7. BMI Categories
# ==============================
def bmi_category(bmi):
    if bmi < 18.5:
        return 0   # Underweight
    elif bmi < 25:
        return 1   # Normal
    elif bmi < 30:
        return 2   # Overweight
    else:
        return 3   # Obese

df["bmi_category"] = df["bmi"].apply(bmi_category)

print("\nBMI category check:")
print(df[["bmi", "bmi_category"]].head())

# ==============================
# 8. Blood Pressure Features
# ==============================
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
df["map_pressure"] = df["ap_lo"] + (df["pulse_pressure"] / 3)

print("\nBlood pressure feature check:")
print(df[["ap_hi", "ap_lo", "pulse_pressure", "map_pressure"]].head())

# ==============================
# 9. Blood Pressure Categories (NHS-style)
# ==============================
def bp_category(sys, dia):
    if sys <= 120 and dia <= 80:
        return 0   # Normal
    elif sys <= 139 or dia <= 89:
        return 1   # High Normal
    elif sys <= 159 or dia <= 99:
        return 2   # Stage 1 Hypertension
    elif sys <= 179 or dia <= 119:
        return 3   # Stage 2 Hypertension
    else:
        return 4   # Severe Hypertension

df["bp_category"] = df.apply(
    lambda x: bp_category(x["ap_hi"], x["ap_lo"]), axis=1
)

print("\nBlood pressure category check:")
print(df[["ap_hi", "ap_lo", "bp_category"]].head())

# ==============================
# 10. Lifestyle Risk Score
# ==============================
df["lifestyle_risk"] = (
    df["smoke"] * 3 +
    df["alco"] * 2 +
    (1 - df["active"]) * 2
)

print("\nLifestyle risk check:")
print(df[["smoke", "alco", "active", "lifestyle_risk"]].head())

# ==============================
# 11. Metabolic Risk Score
# ==============================
df["metabolic_risk"] = (
    (df["cholesterol"] - 1) * 10 +
    (df["gluc"] - 1) * 10 +
    df["bp_category"] * 10
)

print("\nMetabolic risk check:")
print(df[["cholesterol", "gluc", "bp_category", "metabolic_risk"]].head())

# ==============================
# 12. Combined Risk Score
# ==============================
df["combined_risk"] = df["lifestyle_risk"] * 5 + df["metabolic_risk"]

print("\nCombined risk check:")
print(df[["lifestyle_risk", "metabolic_risk", "combined_risk"]].head())

# ==============================
# 13. Prepare ML Data
# ==============================
X = df.drop(columns=["cardio"])
y = df["cardio"]

print("\nML data prepared:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ==============================
# 14. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-test split done:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ==============================
# 15. Logistic Regression (Baseline)
# ==============================
log_model = LogisticRegression(max_iter=1000, n_jobs=-1)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Performance:")
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))
print(classification_report(y_test, y_pred_log))

# ==============================
# 16. XGBoost with Class Weighting
# ==============================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print("\nScale_pos_weight:", scale_pos_weight)

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost (Weighted) Performance:")
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
print(classification_report(y_test, y_pred_xgb))

import joblib
import os

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save trained XGBoost model
joblib.dump(xgb_model, "model/xgboost_cardio_model.pkl")

print("\nModel saved successfully at: model/xgboost_cardio_model.pkl")

# ==============================
# 28. Load model for prediction
# ==============================
loaded_model = joblib.load("model/xgboost_cardio_model.pkl")

print("\nModel loaded successfully for prediction")

# ==============================
# 29. New patient data (example)
# ==============================

new_patient = {
    "age": 55 * 365.25,   # age in days (same format as dataset)
    "gender": 1,          # male = 1, female = 0
    "height": 170,
    "weight": 80,
    "ap_hi": 140,
    "ap_lo": 90,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
}

patient_df = pd.DataFrame([new_patient])
# Feature engineering for patient
patient_df["age_years"] = (patient_df["age"] / 365.25).astype(int)

patient_df["bmi"] = patient_df["weight"] / (patient_df["height"] / 100) ** 2
patient_df["bmi_category"] = patient_df["bmi"].apply(bmi_category)

patient_df["pulse_pressure"] = patient_df["ap_hi"] - patient_df["ap_lo"]
patient_df["map_pressure"] = patient_df["ap_lo"] + (patient_df["pulse_pressure"] / 3)

patient_df["bp_category"] = patient_df.apply(
    lambda x: bp_category(x["ap_hi"], x["ap_lo"]), axis=1
)

patient_df["lifestyle_risk"] = (
    patient_df["smoke"] * 3 +
    patient_df["alco"] * 2 +
    (1 - patient_df["active"]) * 2
)

patient_df["metabolic_risk"] = (
    (patient_df["cholesterol"] - 1) * 10 +
    (patient_df["gluc"] - 1) * 10 +
    patient_df["bp_category"] * 10
)

patient_df["combined_risk"] = (
    patient_df["lifestyle_risk"] * 5 +
    patient_df["metabolic_risk"]
)
# Ensure column order matches training data
patient_df = patient_df[X.columns]

# Prediction
risk_probability = loaded_model.predict_proba(patient_df)[0][1]
risk_label = "High Risk" if risk_probability >= 0.5 else "Low Risk"

print("\n--- Patient Prediction Result ---")
print(f"Heart Disease Risk Probability: {risk_probability*100:.2f}%")
print(f"Risk Category: {risk_label}")

from fpdf import FPDF
from datetime import datetime

from fpdf import FPDF
from datetime import datetime
import os

# ==============================
# Generate Patient PDF Report
# ==============================
from fpdf import FPDF
from datetime import datetime
import os

def generate_patient_report(patient_df, risk_probability, risk_label):
    os.makedirs("reports", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"reports/CardioPredict_Report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()

    BLUE = (0, 60, 120)
    LIGHT_GREY = (245, 245, 245)

    # ===== HEADER =====
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 0, 210, 24, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Times", "B", 15)
    pdf.set_xy(10, 7)
    pdf.cell(0, 8, "CardioPredict - Heart Risk Assessment Report", ln=True)

    pdf.set_font("Times", size=10)
    pdf.set_xy(10, 15)
    pdf.cell(0, 6, "AI-based Cardiovascular Screening System", ln=True)

    pdf.ln(20)
    pdf.set_text_color(0, 0, 0)

    # ===== PATIENT INFO =====
    pdf.set_fill_color(*LIGHT_GREY)
    pdf.set_font("Times", "B", 11)
    pdf.cell(0, 7, "Patient Information", ln=True, fill=True)

    pdf.set_font("Times", size=10)
    pdf.multi_cell(
        0, 6,
        f"Age: {patient_df['age_years'].iloc[0]} years | "
        f"Gender: {'Male' if patient_df['gender'].iloc[0]==1 else 'Female'} | "
        f"Height: {patient_df['height'].iloc[0]} cm | "
        f"Weight: {patient_df['weight'].iloc[0]} kg\n"
        f"Blood Pressure: {patient_df['ap_hi'].iloc[0]}/{patient_df['ap_lo'].iloc[0]} mmHg"
    )

    # ===== RISK ASSESSMENT =====
    pdf.set_font("Times", "B", 11)
    pdf.cell(0, 7, "Risk Assessment", ln=True, fill=True)

    pdf.set_font("Times", size=10)
    pdf.multi_cell(
        0, 6,
        f"Risk Score: {risk_probability*100:.0f}% | "
        f"Risk Category: {risk_label} | "
        f"BMI: {patient_df['bmi'].iloc[0]:.2f}"
    )

    # ===== RISK CONTRIBUTORS =====
    pdf.set_font("Times", "B", 11)
    pdf.cell(0, 7, "Key Risk Contributors", ln=True, fill=True)

    pdf.set_font("Times", size=10)
    if patient_df["bp_category"].iloc[0] >= 2:
        pdf.cell(0, 6, "- Blood pressure above normal (treatable)", ln=True)
    if patient_df["cholesterol"].iloc[0] > 1:
        pdf.cell(0, 6, "- Cholesterol above normal (modifiable)", ln=True)
    if patient_df["gluc"].iloc[0] > 1:
        pdf.cell(0, 6, "- Glucose above normal (modifiable)", ln=True)
    if patient_df["smoke"].iloc[0] == 1:
        pdf.cell(0, 6, "- Smoking habit present (modifiable)", ln=True)

    # ===== DIET + CLINICAL (MERGED) =====
    pdf.set_font("Times", "B", 11)
    pdf.cell(0, 7, "Dietary and Clinical Recommendations", ln=True, fill=True)

    pdf.set_font("Times", size=10)
    pdf.multi_cell(
        0, 6,
        "- Follow DASH or Mediterranean diet; limit salt to < 2 g/day\n"
        "- Increase fruits, vegetables, whole grains; avoid fried foods\n"
        "- Target blood pressure below 130/80 mmHg\n"
        "- Engage in 30 minutes of physical activity daily"
    )

    # ===== INTERPRETATION + FOLLOW-UP (MERGED) =====
    pdf.set_font("Times", "B", 11)
    pdf.cell(0, 7, "Clinical Interpretation and Follow-Up", ln=True, fill=True)

    pdf.set_font("Times", size=10)
    pdf.multi_cell(
        0, 6,
        "The patient demonstrates elevated cardiovascular risk primarily due to blood "
        "pressure and metabolic factors. Early lifestyle modification and medical "
        "follow-up can significantly reduce long-term complications.\n"
        "Follow-up advised: BP review in 2-4 weeks, annual cardiovascular screening."
    )

    # ===== FOOTER =====
    pdf.set_font("Times", "I", 8)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This AI-generated report is intended for screening and educational "
        "purposes only and does not replace professional medical advice."
    )

    pdf.ln(1)
    pdf.set_font("Times", "B", 9)
    pdf.cell(0, 5, "Project Developed By: S. Vishnu Vardhan Reddy", ln=True)

    pdf.output(file_path)
    return file_path




    # SAVE PDF (ONLY ONCE)
    pdf.output(file_path)

    return file_path
report_path = generate_patient_report(
    patient_df,
    risk_probability,
    risk_label
)

print("\nPatient report generated successfully:", report_path)
