import requests
import json

# API endpoint
url = "http://localhost:5000/api/predict"

# Patient data
patient_data = {
    "patient_practice_id": "39987__Ardent Family Care",
    "PatientID": "39987",
    "date_of_birth": "1948-05-04",
    "age_years": "77",
    "gender": "F",
    "race": "White",
    "marital_status": "Married",
    "icd_10": "R06",
    "date_of_icd_code": "2022-08-15",
    "jsdisease": "",
    "api_test_name": "High density lipoprotein (HDL) cholesterol measurement; Glucose measurement; Glucose measurement; Serum or plasma creatinine measurement (mass/volume); Thyroid stimulating hormone (TSH) measurement; Low density lipoprotein (LDL) cholesterol measurement; Serum total cholesterol to high density lipoprotein cholesterol ratio",
    "date_of_api_test_name": "2022-08-15; 2022-08-15; 2022-08-15; 2022-08-15; 2022-08-15; 2022-08-15; 2022-08-15",
    "lab_name_result": "High density lipoprotein (HDL) cholesterol measurement=46; Glucose measurement=NEGATIVE; Glucose measurement=105; Serum or plasma creatinine measurement (mass/volume)=0.64; Thyroid stimulating hormone (TSH) measurement=0.81; Low density lipoprotein (LDL) cholesterol measurement=117; Serum total cholesterol to high density lipoprotein cholesterol ratio=4",
    "height": "157.48; 157.48",
    "date_of_height_value": "2024-01-29; 2023-02-19",
    "weight": "78.93; 79.14",
    "date_of_weight_value": "2024-01-29; 2023-02-19",
    "bmi": "31.825; 31.911",
    "weight_category": "ExtremelyObese; Obese",
    "systolic_bp": "122; 132",
    "diastolic_bp": "84; 82",
    "smoking_status": "Current",
    "alcohol_usage_type": "Never",
    "surgery_name": "",
    "date_of_surgery_name": "",
    "gpi": "050000; 394000; 221000; 415500; 394000; 772020; 034000; 422000",
    "date_of_gpi": "2019-11-07; 2021-07-26; 2019-03-26; 2019-05-22",
    "status_of_gpi": "Active; Active; Active; Active"
}

# ============================================================================
# CHANGE THE MODEL NAME HERE
# ============================================================================
# Options: 'XGBoost', 'Ridge_Regression', 'Lasso_Regression', 
#          'ElasticNet_Regression', 'Random_Forest', 'Gradient_Boosting'

MODEL_NAME = 'Ridge_Regression'  # <-- CHANGE THIS

# ============================================================================

# Prepare request
request_data = {
    "model_name": MODEL_NAME,
    "patient_data": patient_data
}

# Make API call
print(f"Testing model: {MODEL_NAME}")
print("="*80)

response = requests.post(url, json=request_data)
result = response.json()

print(json.dumps(result, indent=2))