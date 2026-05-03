import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.preprocessor import DataPreprocessor
from services.encoder import FeatureEncoder

logger = logging.getLogger(__name__)


class BMIClassifier:
    """
    Complete BMI classification system for predicting BMI direction (increase/decrease).
    Predicts whether a patient's BMI will increase or decrease based on historical data.
    """
    
    def __init__(self, model_name: str = 'XGBoost_Classifier'):
        """
        Initialize classifier with trained model
        
        Args:
            model_name: Name of the saved classification model to load
        """
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_metrics = None
        self.preprocessor = DataPreprocessor()
        self.encoder = FeatureEncoder()

        self.base_path = Path(__file__).parent.parent

        
        # Paths for classification models
        # Scaler and feature names are in saved_models/classification/
        self.scaler_path = self.base_path / 'models' / 'saved_models' / 'scaler.pkl'
        self.feature_names_path = self.base_path / 'models' / 'saved_models' / 'feature_names.pkl'
        
        # Model .pkl and metrics.json are directly in models/{model_name}/
        self.model_folder = self.base_path / 'models' / model_name

        self.model_path = self.model_folder / f'{model_name}.pkl'
        self.metrics_path = self.model_folder / 'metrics.json'
        self.feature_names_path = self.model_folder / 'feature_names.pkl'
        self.scaler_path = self.model_folder / 'scaler.pkl'

        print(self.model_path)

 
        # Columns to drop during inference
        self.columns_to_drop = [
            'patient_practice_id',
            'PatientID',
            'latest_weight',
            'latest_height',
            'm5_weight',
            'm5_height',
            'latest_MAP_groups',
            'history_of_KidneyDisease',
            'Nissen_Fundoplication',
            'Obesity',
            'latest_bmi',      # Used to calculate actual direction
            'bmi_change',      # Derived variable
            'bmi_direction',    # Target variable

        ]
        
        # Class labels
        self.class_labels = {
            0: 'Decrease',
            1: 'Increase'
        }
        
        # Load artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load saved model, scaler, feature names, and metrics"""
        logger.info(f"Loading classification model artifacts for {self.model_name}...")
        
        # DEBUG: Print all paths
        print("\n" + "="*80)
        print("DEBUG: Artifact Paths")
        print("="*80)
        print(f"Base path: {self.base_path}")
        print(f"Model folder: {self.model_folder}")
        print(f"Model path: {self.model_path}")
        print(f"Model exists: {self.model_path.exists()}")
        print(f"Scaler path: {self.scaler_path}")
        print(f"Scaler exists: {self.scaler_path.exists()}")
        print(f"Feature names path: {self.feature_names_path}")
        print(f"Feature names exists: {self.feature_names_path.exists()}")
        print(f"Metrics path: {self.metrics_path}")
        print(f"Metrics exists: {self.metrics_path.exists()}")
        print("="*80 + "\n")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from: {self.model_path}")
            
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from: {self.scaler_path}")
            
            self.feature_names = joblib.load(self.feature_names_path)
            logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            
            with open(self.metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
            logger.info(f"Metrics loaded from: {self.metrics_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Model artifact not found: {e}")
            print(f"\n❌ ERROR: {e}")
            raise Exception(f"Could not load classification model artifacts. Ensure model is trained and saved at {self.model_folder}")
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main classification method - predicts BMI direction (increase/decrease)
        """
        logger.info("Starting classification pipeline...")
        
        # 1. Convert to DataFrame
        df = pd.DataFrame([patient_data])
        logger.info(f"Created DataFrame with {len(df.columns)} columns")
        
        # 2. Preprocess
        df_preprocessed = self.preprocessor.transform(df)
        logger.info(f"Preprocessed: {df_preprocessed.shape}")
        
        # 3. Encode
        df_encoded = self.encoder.transform(df_preprocessed)
        logger.info(f"Encoded: {df_encoded.shape}")
        
        # 4. Extract actual values before dropping (if available)
        actual_latest_bmi = None
        actual_m5_bmi = None
        actual_change = None
        actual_class = None
        
        if 'latest_bmi' in df_encoded.columns and 'm5_bmi' in df_encoded.columns:
            actual_latest_bmi = float(df_encoded['latest_bmi'].values[0])
            actual_m5_bmi = float(df_encoded['m5_bmi'].values[0])
            actual_change = actual_latest_bmi - actual_m5_bmi
            actual_class = int(actual_change > 0)
            logger.info(f"Actual BMI change: {actual_change:.2f} (Class: {actual_class})")
        
        # 5. Drop unnecessary columns
        df_features = df_encoded.drop(
            columns=[col for col in self.columns_to_drop if col in df_encoded.columns],
            errors='ignore'
        )
        logger.info(f"Features after dropping: {df_features.shape}")
        
        # 6. Align features with training
        df_features = self._align_features(df_features)
        
        # 7. Convert to numpy array to bypass feature name validation
        X_array = df_features.values
        
        # 8. Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # 9. Predict class and probabilities
        predicted_class = int(self.model.predict(X_scaled)[0])
        predicted_proba = self.model.predict_proba(X_scaled)[0]
        
        # 10. Extract probability values
        prob_decrease = float(predicted_proba[0])
        prob_increase = float(predicted_proba[1])
        confidence = float(np.max(predicted_proba))  # THIS LINE WAS MISSING
        
        predicted_label = self.class_labels[predicted_class]
        
        logger.info(f"Prediction: Class {predicted_class} ({predicted_label}) with {confidence:.2%} confidence")
        
        # 11. Build response
        response = {
            'patient_id': patient_data.get('patient_practice_id') or patient_data.get('PatientID'),
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'probabilities': {
                'decrease': prob_decrease,
                'increase': prob_increase
            },
            'confidence': confidence,
            'model_name': self.model_name,
        }
        
        # 12. Add actual values if available
        if actual_class is not None:
            actual_label = self.class_labels[actual_class]
            correct_prediction = (predicted_class == actual_class)
            
            response['actual_values'] = {
                'm5_bmi': actual_m5_bmi,
                'latest_bmi': actual_latest_bmi,
                'bmi_change': actual_change,
                'actual_class': actual_class,
                'actual_label': actual_label
            }
            
            response['prediction_result'] = {
                'correct_prediction': correct_prediction,
                'prediction_match': 'Correct' if correct_prediction else 'Incorrect'
            }

        perf = self.model_metrics.get('performance_metrics', {})
        cm   = self.model_metrics.get('confusion_matrix', {})
        # 13. Add model performance metrics

        response['model_metrics'] = {
            'performance_metrics': {
                'accuracy':  perf.get('accuracy'),
                'precision': perf.get('precision'),
                'recall':    perf.get('recall'),
                'f1_score':  perf.get('f1_score'),
                'roc_auc':   perf.get('roc_auc'),
            },
            'confusion_matrix': {
                'true_negative': cm.get('true_negative'),
                'false_positive': cm.get('false_positive'),
                'false_negative': cm.get('false_negative'),
                'true_positive':  cm.get('true_positive'),
            }
            }
        
        return response
    
    def _align_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Align features with those used during training"""
        current_features = set(df_features.columns)
        expected_features = set(self.feature_names)
        
        # Remove extra features
        extra_features = current_features - expected_features
        if extra_features:
            logger.warning(f"Removing extra features: {extra_features}")
            df_features = df_features.drop(columns=list(extra_features))
        
        # Add missing features with 0
        missing_features = expected_features - current_features
        if missing_features:
            logger.warning(f"Adding missing features with value 0: {missing_features}")
            for feature in missing_features:
                df_features[feature] = 0
        
        # Ensure correct order and fill NaN
        return df_features[self.feature_names].fillna(0)