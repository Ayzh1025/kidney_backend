import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

app = Flask(__name__, static_folder="client/build", static_url_path="")
CORS(app)

@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")


@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, "index.html")

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "Waitinglist_patients.csv")

df = pd.read_csv(DATA_PATH)

def filter_data(df, hba1c, **filters):
    print(filters)
    filtered = df.copy()
    
    # Initialize filter_number with total number of rows
    filter_number = filtered.shape[0]

    for col, val in filters.items():
        if val is None or val == "" or val == []:  # skip if empty
            continue
        if isinstance(val, list):
            filtered = filtered[filtered[col].isin(val)]
        else:
            filtered = filtered[filtered[col] == val]

    # Apply hba1c-based weighting AFTER filtering
    if hba1c is not None:
        if hba1c == 1:
            filter_number = int(filtered.shape[0] * 0.115)
        elif hba1c == 2:
            filter_number = int(filtered.shape[0] * 0.29)
        elif hba1c == 3:
            filter_number = int(filtered.shape[0] * 0.405)
        elif hba1c == 4:
            filter_number = int(filtered.shape[0] * 0.195)
        else:
            filter_number = int(filtered.shape[0] * 0.05)
    else:
        filter_number = filtered.shape[0]

    return filter_number


def summarize_results(hba1c, **filters):


    # ---- Age Categories ----
    age = filters.get("AGE_CAT")
    if age is None:
        age_group = "Not Selected"
    elif age == 1:
        age_group = "0–17"
    elif age == 2:
        age_group = "18–44"
    elif age == 3:
        age_group = "45–64"
    elif age == 4:
        age_group = "65–74"
    else:
        age_group = "75+"

    # ---- BMI Categories ----
    bmi = filters.get("BMI_CAT")
    if bmi is None:
        bmi_group = "Not Selected"
    elif bmi == 1:
        bmi_group = "Underweight (≤18.5)"
    elif bmi ==2:
        bmi_group = "Normal/Overweight (18.6–29.9)"
    elif bmi == 3:
        bmi_group = "Obese Class I (30–34.9)"
    elif bmi == 4:
        bmi_group = "Obese Class II (35–39.9)"
    else:
        bmi_group = "Obese Class III (≥40)"

    # ---- Ethnicity ----
    ethnicity_map = {
        1: "White, Non-Hispanic",
        2: "Black, Non-Hispanic",
        4: "Hispanic/Latino",
        5: "Asian, Non-Hispanic",
        6: "American Indian/Alaska Native",
        7: "Native Hawaiian/Pacific Islander",
        9: "Multiracial, Non-Hispanic",
        998: "Unknown"
    }
    ethnicity = ethnicity_map.get(filters.get("ETHCAT"), "Not Selected")

    # ---- Diabetes Type ----
    diabetes_map = {
        1: "No Diabetes",
        2: "Type I",
        3: "Type II",
        4: "Other",
        5: "Unknown"
    }
    diabetes = diabetes_map.get(filters.get("DIAB"), "Not Selected")

    # ---- HbA1c ----
    hba1c_cat = hba1c
    if hba1c is None:
        hba1c_cat = "Not Selected"
    elif hba1c <= 5:
        hba1c_cat = "≤5"
    elif 5 < hba1c <= 6:
        hba1c_cat = "5–6"
    elif 6 < hba1c <= 8:
        hba1c_cat = "6–8"
    elif 8 < hba1c <= 10:
        hba1c_cat = ">8–10"
    else:
        hba1c_cat = ">10"

    # ---- cPRA ----
    cpra = filters.get("CCPRA_CAT")
    if cpra is None:
        cpra_cat = "Not Selected"
    elif cpra == 2:
        cpra_cat = "0 (No Sensitization)"
    elif cpra == 3:
        cpra_cat = "1–19 (Low Sensitization)"
    elif cpra == 4:
        cpra_cat = "20–79 (Moderate Sensitization)"
    elif cpra == 5:
        cpra_cat = "80–97 (High Sensitization)"
    else:
        cpra_cat = "98–100 (Very High Sensitization)"

    # ---- Build Summary ----
    summary = {
        "Age Group": age_group,
        "Gender": filters.get("GENDER", "Not Selected") or "Not Selected",
        "BMI Category": bmi_group,
        "Ethnicity": ethnicity,
        "Payment Type": filters.get("PAYC_CAT", "Not Selected") or "Not Selected",
        "State": filters.get("PERM_STATE", "Not Selected") or "Not Selected",
        "Region": filters.get("REGION", "Not Selected") or "Not Selected",
        "Diabetes Type": diabetes,
        "HbA1c": hba1c_cat,
        "cPRA": cpra_cat,
        "Blood Type": filters.get("ABO", "Not Selected") or "Not Selected"
    }

    return summary


@app.route('/predict', methods=['POST'])
def predict():
    print("predict reached")
    data = request.json

    # Extract data
    age = float(data.get("age")) if data.get("age") else None
    bmi = float(data.get("bmi")) if data.get("bmi") else None
    cpra = float(data.get("cpra")) if data.get("cpra") else None
    cpra_cat = None
    if cpra is not None:
        if cpra == 0:
            cpra_cat = 2
        elif 0 < cpra < 20:
            cpra_cat = 3
        elif 20 <= cpra < 80:
            cpra_cat = 4
        elif 80 <= cpra < 98:
            cpra_cat = 5
        elif bmi >= 98:
            cpra_cat = 6
    # Convert BMI to BMI category
    bmi_cat = None
    if bmi is not None:
        if 0 < bmi <= 18.5:
            bmi_cat = 1
        elif 18.5 < bmi <= 29.9:
            bmi_cat = 2
        elif 29.9 < bmi <= 34.9:
            bmi_cat = 3
        elif 35 <= bmi <= 39.9:
            bmi_cat = 4
        elif bmi >= 40:
            bmi_cat = 5

    age_cat = None
    if age is not None:
        if 0 < age <= 17:
            age_cat = 1
        elif 17 < age <= 44:
            age_cat = 2
        elif 44 < age <= 64:
            age_cat = 3
        elif 64 <= age <= 74:
            age_cat = 4
        elif age >= 40:
            age_cat = 5
    gender = data.get("gender")
    if gender == "Male":
        gender = "M"
    elif gender == "Female":
        gender = "F"
    else:
        gender = None 
    
    state = data.get("state")
    region = data.get("region")
    hba1c = float(data.get("hba1c")) if data.get("hba1c") else None
    hba1c_cat=None
    if hba1c is not None:
        if hba1c <= 5:
            hba1c_cat = 1   # Normal (≤5)
        elif 5 < hba1c <= 6:
            hba1c_cat = 2   # 5–6
        elif 6 < hba1c <= 8:
            hba1c_cat = 3   # 6–8
        elif 8 < hba1c <= 10:
            hba1c_cat = 4   # >8–10
        elif hba1c > 10:
            hba1c_cat = 5
    on_dialysis = int(data.get("onDialysis")) if data.get("onDialysis") not in [None, ""] else None
    first_dialysis_date = data.get("firstDialysisDate")
    payment_type = data.get("paymentType")
    blood_type = data.get("bloodType")
    ethnicity = data.get("ethnicity")
    comorbidities = data.get("comorbidities")
    diab = None
    diab_ty = data.get("diabetesType")
    if diab_ty is not None:
        if diab_ty == "Type 1":
            diab = 2
        elif diab_ty == "Type 2":
            diab = 3
        elif diab_ty == "Other":
            diab = 4
        elif diab_ty =="None":
            diab = 1
   
    ethnicity_map = {
        "White, Non-Hispanic": 1,
        "Black, Non-Hispanic": 2,
        "Hispanic/Latino": 4,
        "Asian, Non-Hispanic": 5,
        "Amer Ind/Alaska Native, Non-Hispanic": 6,
        "Native Hawaiian/other Pacific Islander, Non-Hispanic": 7,
        "Multiracial, Non-Hispanic": 9,
        "Unknown": 998
    }

    eth_cat = None
    if ethnicity:
        if isinstance(ethnicity, list) and len(ethnicity) > 0:
            eth_cat = ethnicity_map.get(ethnicity[0], 998)  # take first if multiple selected
        else:
            eth_cat = ethnicity_map.get(ethnicity, 998)

    filters = {
    "AGE_CAT": age_cat,
    "BMI_CAT": bmi_cat,
    "REGION": region,
    "GENDER": gender,
    "PERM_STATE": state,
    "ABO": blood_type,
    "ETHCAT": eth_cat,
    "PAYC_CAT": payment_type,
    "CCPRA_CAT": cpra_cat,
    "DIAB": diab,
    }
    similar_patients = filter_data(df, hba1c_cat, **filters)
    print(similar_patients)

    percentage = (similar_patients/89928)*100
    percentage = round(percentage, 2)
    results = summarize_results(hba1c, **filters)
    for key, val in results.items():
        print(f"{key}: {val}")

    return jsonify({'similar_patients': similar_patients, 'percentage': percentage, 'summary': results})

@app.route('/ping')
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


