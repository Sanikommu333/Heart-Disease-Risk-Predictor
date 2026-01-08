from fpdf import FPDF
from datetime import datetime
import os

def generate_patient_report(patient_df, risk_probability, risk_label):
    os.makedirs("reports", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"reports/CardioPredict_Report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()

    # ===============================
    # COLORS
    # ===============================
    HEADER_BLUE = (0, 70, 140)
    LIGHT_GREY = (245, 245, 245)
    TEXT_GREY = (90, 90, 90)

    # ===============================
    # HOSPITAL / COLLEGE HEADER
    # ===============================
    pdf.set_fill_color(*HEADER_BLUE)
    pdf.rect(0, 0, 210, 28, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Times", "B", 14)
    pdf.set_xy(10, 6)
    pdf.cell(0, 8, "Clinical AI Demonstration Report", ln=True)

    pdf.set_font("Times", size=10)
    pdf.set_xy(10, 14)
    pdf.cell(
        0, 6,
        "Department of Computer Science | Academic & Clinical Project",
        ln=True
    )

    pdf.ln(22)
    pdf.set_text_color(0, 0, 0)

    # ===============================
    # PATIENT INFORMATION
    # ===============================
    pdf.set_fill_color(*LIGHT_GREY)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 8, "Patient Information", ln=True, fill=True)

    pdf.set_font("Times", size=11)
    pdf.multi_cell(
        0, 7,
        "Age: {} years\n"
        "Gender: {}\n"
        "Height: {} cm\n"
        "Weight: {} kg\n"
        "Blood Pressure: {}/{} mmHg".format(
            patient_df["age_years"].iloc[0],
            "Male" if patient_df["gender"].iloc[0] == 1 else "Female",
            patient_df["height"].iloc[0],
            patient_df["weight"].iloc[0],
            patient_df["ap_hi"].iloc[0],
            patient_df["ap_lo"].iloc[0],
        )
    )

    # ===============================
    # RISK SUMMARY + BADGE
    # ===============================
    pdf.ln(1)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 8, "Risk Assessment Summary", ln=True)

    pdf.set_font("Times", size=11)

    if risk_probability >= 0.75:
        badge = "VERY HIGH RISK"
    elif risk_probability >= 0.45:
        badge = "MODERATE RISK"
    else:
        badge = "LOW RISK"

    pdf.multi_cell(
        0, 7,
        "Estimated Heart Disease Risk: {:.1f}%\n"
        "Risk Category: {}".format(risk_probability * 100, badge)
    )

    # ===============================
    # CLINICAL INDICATORS
    # ===============================
    bmi = patient_df["weight"].iloc[0] / ((patient_df["height"].iloc[0] / 100) ** 2)

    if bmi < 18.5:
        bmi_text = "Underweight"
    elif bmi < 25:
        bmi_text = "Normal"
    elif bmi < 30:
        bmi_text = "Overweight"
    else:
        bmi_text = "Obese"

    if patient_df["ap_hi"].iloc[0] >= 140 or patient_df["ap_lo"].iloc[0] >= 90:
        bp_text = "Hypertension"
    elif patient_df["ap_hi"].iloc[0] >= 120:
        bp_text = "Elevated"
    else:
        bp_text = "Normal"

    pdf.ln(1)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 8, "Clinical Indicators", ln=True)

    pdf.set_font("Times", size=11)
    pdf.multi_cell(
        0, 7,
        "Body Mass Index (BMI): {:.2f} ({})\n"
        "Blood Pressure Category: {}".format(bmi, bmi_text, bp_text)
    )

    # ===============================
    # LIFESTYLE SUMMARY
    # ===============================
    pdf.ln(1)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 8, "Lifestyle Summary", ln=True)

    pdf.set_font("Times", size=11)
    pdf.multi_cell(
        0, 7,
        "Smoking: {}\n"
        "Alcohol Consumption: {}\n"
        "Physical Activity Level: {}".format(
            "Yes" if patient_df["smoke"].iloc[0] == 1 else "No",
            "Yes" if patient_df["alco"].iloc[0] == 1 else "No",
            "Active" if patient_df["active"].iloc[0] == 1 else "Low Activity",
        )
    )

    # ===============================
    # SMART RECOMMENDATIONS
    # ===============================
    pdf.ln(1)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 8, "Personalized Recommendations", ln=True)

    pdf.set_font("Times", size=11)

    if bp_text != "Normal":
        pdf.cell(0, 6, "- Maintain blood pressure below 130/80 mmHg", ln=True)

    if bmi >= 25:
        pdf.cell(0, 6, "- Gradual weight reduction through balanced diet", ln=True)

    if patient_df["smoke"].iloc[0] == 1:
        pdf.cell(0, 6, "- Smoking cessation is strongly advised", ln=True)

    if patient_df["active"].iloc[0] == 0:
        pdf.cell(0, 6, "- Engage in at least 30 minutes of physical activity daily", ln=True)

    pdf.cell(0, 6, "- Schedule regular health check-ups", ln=True)

    # ===============================
    # DISCLAIMER + FOOTER
    # ===============================
    pdf.ln(2)
    pdf.set_text_color(*TEXT_GREY)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(
        0, 6,
        "Disclaimer: This AI-generated report is intended for screening and educational "
        "purposes only and does not replace professional medical advice."
    )

    pdf.ln(1)
    pdf.set_font("Times", "B", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(
        0, 6,
        "Project Developed By: S. Vishnu Vardhan Reddy and G.Rajesh Kumar",
        ln=True
    )

    pdf.output(file_path)
    return file_path