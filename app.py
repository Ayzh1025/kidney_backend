import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os
import logging
logging.basicConfig(level=logging.DEBUG)

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


state_map = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"
}



region_map = {
  "Region 1 – Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Eastern Vermont": 1,
  "Region 2 – Delaware, District of Columbia, Maryland, New Jersey, Pennsylvania, West Virginia, Northern Virginia": 2,
  "Region 3 – Alabama, Arkansas, Florida, Georgia, Louisiana, Mississippi, Puerto Rico": 3,
  "Region 4 – Oklahoma, Texas": 4,
  "Region 5 – Arizona, California, Nevada, New Mexico, Utah": 5,
  "Region 6 – Alaska, Hawaii, Idaho, Montana, Oregon, Washington": 6,
  "Region 7 – Illinois, Minnesota, North Dakota, South Dakota, Wisconsin": 7,
  "Region 8 – Colorado, Iowa, Kansas, Missouri, Nebraska, Wyoming": 8,
  "Region 9 – New York, Western Vermont": 9,
  "Region 10 – Indiana, Michigan, Ohio": 10,
  "Region 11 – Kentucky, North Carolina, South Carolina, Southern Ohio, Tennessee, Virginia": 11
}

ethnicity_map = {
        1: "White, Non-Hispanic",
        2: "Black, Non-Hispanic",
        4: "Hispanic/Latino",
        5: "Asian, Non-Hispanic",
        6: "Amer Ind/Alaska Native",
        7: "Native Hawaiian/other Pacific Islander, Non-Hispanic",
        9: "Multiracial, Non-Hispanic",
        998: "Unknown"
    }

ethnicity_reverse_map = {v: k for k, v in ethnicity_map.items()}

diabetes_map = {
        1: "None",
        2: "Type I",
        3: "Type II",
        4: "Other",
        5: "Unknown"
    }

diabetes_reverse_map = {v: k for k, v in diabetes_map.items()}


def filter_data(df, hba1c, **filters):
    filtered = df.copy()
    
    

    for col, val in filters.items():
        if val is None or val == "" or val == []:
            continue

        # Multi-select fields use OR (isin)
        if col in ["PERM_STATE", "REGION", "ETHCAT", "DIAB", "ABO"]:
            if isinstance(val, list):
                # Map full names to database values if necessary
                if col == "PERM_STATE":
                    db_values = [state_map.get(v, v) for v in val]
                elif col == "REGION":
                    db_values = [region_map.get(v, v) for v in val]
                elif col == "ETHCAT":
                    db_values = [ethnicity_reverse_map.get(v, v) for v in val]
                    print("reached", db_values)
                elif col == "DIAB":
                    db_values = [diabetes_reverse_map.get(v,v) for v in val]
                elif col == "ABO":
                    db_values = val
                filtered = filtered[filtered[col].isin(db_values)]
                print(db_values)
                
            else:
                filtered = filtered[filtered[col] == val]
        else:
            # Single-value filters use AND
            filtered = filtered[filtered[col] == val]
        

    # Apply hba1c weighting if needed
    filter_number = filtered.shape[0]
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

    
    ethnicity = filters.get("ETHCAT")

    diabetes = (filters.get("DIAB"))

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
    eth_cat = filters.get("ETHCAT")

    state = filters.get("PERM_STATE")
    region = filters.get("REGION")
    abo = filters.get("ABO")
    # ---- Build Summary ----
    summary = {
        "Age Group": age_group,
        "Gender": filters.get("GENDER", "Not Selected") or "Not Selected",
        "BMI Category": bmi_group,
        "Ethnicity":  ", ".join(eth_cat) if isinstance(eth_cat, list) and eth_cat else eth_cat or "Not Selected",
        "Payment Type": filters.get("PAYC_CAT", "Not Selected") or "Not Selected",
        "State":  ", ".join(state) if isinstance(state, list) and state else state or "Not Selected",
        "Region":  ", ".join(region) if isinstance(region, list) and region else region or "Not Selected",
        "Diabetes Type":  ", ".join(diabetes) if isinstance(diabetes, list) and diabetes else diabetes or "Not Selected",
        "HbA1c": hba1c_cat,
        "cPRA": cpra_cat,
        "Blood Type":  ", ".join(abo) if isinstance(abo, list) and abo else abo or "Not Selected"
    }

    return summary









@app.route('/predict', methods=['POST'])
def predict():
    app.logger.debug("predict reached")
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
    if payment_type == "Other":
        payment_type = "Others"
    print("Payment type", payment_type)
    blood_type = data.get("bloodType")
    ethnicity = data.get("ethnicity")
    comorbidities = data.get("comorbidities")





    diab_ty_list = data.get("diabetesType")  # could be None or a list of strings

    # Mapping dictionary
    diab_map = {
        "None": 1,
        "Type 1": 2,
        "Type 2": 3,
        "Other": 4
        }

    diab = None
    if diab_ty_list:
        diab = diab_ty_list
    print("diab", diab)

    eth_cat = None
    if ethnicity:
        eth_cat = ethnicity


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
    app.run(host="0.0.0.0", port=5002, debug=True)


