import pandas as pd
import joblib
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OOP.services.preprocessor import DataPreprocessor
from OOP.services.encoder import FeatureEncoder


def predict_patient_bmi(patient_data):
    """
    Simple prediction function
    
    Args:
        patient_data: Dictionary with patient information
        
    Returns:
        Dictionary with prediction and metrics
    """
    
    print("="*80)
    print("BMI PREDICTION PIPELINE")
    print("="*80)
    
    # 1. Convert to DataFrame
    print("\n[1/6] Converting patient data to DataFrame...")
    df = pd.DataFrame([patient_data])
    print(f"✓ Created DataFrame with {len(df.columns)} columns")
    
    # 2. Preprocess
    print("\n[2/6] Running preprocessing...")
    preprocessor = DataPreprocessor()
    df_preprocessed = preprocessor.transform(df)
    print(f"✓ Preprocessed: {df_preprocessed.shape}")
    
    # 3. Encode
    print("\n[3/6] Running encoding...")
    encoder = FeatureEncoder()
    df_encoded = encoder.transform(df_preprocessed)
    print(f"✓ Encoded: {df_encoded.shape}")
    
    # 4. Drop specified columns
    print("\n[4/6] Dropping unnecessary columns...")
    columns_to_drop = [
        'patient_practice_id',
        'latest_weight',
        'latest_height',
        'm5_weight',
        'm5_height',
        'latest_MAP_groups',
        'history_of_KidneyDisease',
        'Nissen_Fundoplication',
        'Obesity',
    ]
    
    # Extract actual BMI before dropping
    actual_bmi = df_encoded['latest_bmi'].values[0] if 'latest_bmi' in df_encoded.columns else None
    
    # Drop columns (including latest_bmi target)
    columns_to_drop.append('latest_bmi')
    df_features = df_encoded.drop(columns=[col for col in columns_to_drop if col in df_encoded.columns], errors='ignore')
    print(f"✓ Features after dropping: {df_features.shape}")
    
    # 5. Load model and artifacts
    print("\n[5/6] Loading model...")
    model = joblib.load('models/saved_models/XGBoost_model.pkl')
    scaler = joblib.load('models/saved_models/scaler.pkl')
    feature_names = joblib.load('models/saved_models/feature_names.pkl')
    print(f"✓ Model loaded (expects {len(feature_names)} features)")
    
    # Align features
    current_features = set(df_features.columns)
    expected_features = set(feature_names)
    
    extra_features = current_features - expected_features
    missing_features = expected_features - current_features
    
    if extra_features:
        df_features = df_features.drop(columns=list(extra_features))
    
    if missing_features:
        for feature in missing_features:
            df_features[feature] = 0
    
    # Align features with training (correct order)
    X = df_features[feature_names].fillna(0)
    X_scaled = scaler.transform(X)
    
    # 6. Predict
    print("\n[6/6] Making prediction...")
    predicted_bmi = model.predict(X_scaled)[0]
    
    # Calculate individual metrics
    results = {
        'predicted_bmi': float(predicted_bmi),
        'actual_bmi': float(actual_bmi) if actual_bmi else None
    }
        
    if actual_bmi:
        error = predicted_bmi - actual_bmi
        results.update({
            'error': float(error),
            'absolute_error': float(abs(error)),
            'percentage_error': float((error / actual_bmi) * 100),
            'absolute_percentage_error': float((abs(error) / actual_bmi) * 100),
        })
    
    return results


if __name__ == "__main__":
    
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
    
    # Run prediction
    results = predict_patient_bmi(patient_data)
    
    # Load model performance metrics
    metrics_file = 'models/model_metrics/XGBoost_metrics.json'
    with open(metrics_file, 'r') as f:
        model_metrics = json.load(f)
    
    # Display individual prediction results
    print("\n" + "="*80)
    print("INDIVIDUAL PREDICTION RESULTS")
    print("="*80)
    print(f"Patient ID:     {patient_data['patient_practice_id']}")
    print(f"Predicted BMI:  {results['predicted_bmi']:.2f}")
    
    if results['actual_bmi']:
        print(f"Actual BMI:     {results['actual_bmi']:.2f}")
        print(f"\nError:          {results['error']:+.2f}")
        print(f"Absolute Error: {results['absolute_error']:.2f}")
        print(f"% Error:        {results['percentage_error']:+.2f}%")
    
    # Display model performance metrics
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS (XGBoost)")
    print("="*80)
    
    print("\n--- TRAINING SET ---")
    train = model_metrics['train_metrics']
    print(f"R² Score:  {train['R2']:.4f}")
    print(f"RMSE:      {train['RMSE']:.4f}")
    print(f"MAE:       {train['MAE']:.4f}")
    print(f"MAPE:      {train['MAPE']:.2f}%")
    print(f"RMSLE:     {train['RMSLE']:.4f}")
    
    print("\n--- VALIDATION SET ---")
    val = model_metrics['val_metrics']
    print(f"R² Score:  {val['R2']:.4f}")
    print(f"RMSE:      {val['RMSE']:.4f}")
    print(f"MAE:       {val['MAE']:.4f}")
    print(f"MAPE:      {val['MAPE']:.2f}%")
    print(f"RMSLE:     {val['RMSLE']:.4f}")
    
    print("\n--- TEST SET ---")
    test = model_metrics['test_metrics']
    print(f"R² Score:  {test['R2']:.4f}")
    print(f"RMSE:      {test['RMSE']:.4f}")
    print(f"MAE:       {test['MAE']:.4f}")
    print(f"MAPE:      {test['MAPE']:.2f}%")
    print(f"RMSLE:     {test['RMSLE']:.4f}")
    
    print("\n--- CROSS-VALIDATION ---")
    cv = model_metrics['cv_scores']
    print(f"Mean R²:   {cv['CV_Mean_R2']:.4f} (±{cv['CV_Std_R2']:.4f})")
    
    print("="*80)