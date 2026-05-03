import pandas as pd
import joblib
import json
import logging
from typing import Dict, Any
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.explainability import ModelExplainer
from services.preprocessor import DataPreprocessor
from services.encoder import FeatureEncoder

logger = logging.getLogger(__name__)


class BMIPredictor:
    """
    Complete BMI prediction system with preprocessing, encoding, and model inference.
    """
    
    def __init__(self, model_name: str = 'XGBoost'):
        """
        Initialize predictor with trained model
        
        Args:
            model_name: Name of the saved model to load
        """

        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_metrics = None
        self.preprocessor = DataPreprocessor()
        self.encoder = FeatureEncoder()
        self.explainer = ModelExplainer(model_name)
        
        # Paths
        self.base_path = Path(__file__).parent.parent
        self.model_path = self.base_path / 'models' / 'saved_models' / f'{model_name}_model.pkl'
        self.scaler_path = self.base_path / 'models' / 'saved_models' / 'scaler.pkl'
        self.feature_names_path = self.base_path / 'models' / 'saved_models' / 'feature_names.pkl'
        self.metrics_path = self.base_path / 'models' / 'model_metrics' / f'{model_name}_metrics.json'
        
        # Columns to drop during inference
        self.columns_to_drop = [
            'patient_practice_id',
            'latest_weight',
            'latest_height',
            'm5_weight',
            'm5_height',
            'latest_MAP_groups',
            'history_of_KidneyDisease',
            'Nissen_Fundoplication',
            'Obesity',
            'latest_bmi'  # Target variable
        ]
        
        # Load artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load saved model, scaler, feature names, and metrics"""
        logger.info(f"Loading model artifacts for {self.model_name}...")
        
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
            raise Exception(f"Could not load model artifacts. Ensure model is trained and saved.")
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction method
        
        Args:
            patient_data: Dictionary with patient information
            
        Returns:
            Dictionary with prediction, actual value, individual metrics, and model metrics
        """
        logger.info("Starting prediction pipeline...")
        
        # 1. Convert to DataFrame
        df = pd.DataFrame([patient_data])
        logger.info(f"Created DataFrame with {len(df.columns)} columns")
        
        # 2. Preprocess
        df_preprocessed = self.preprocessor.transform(df)
        logger.info(f"Preprocessed: {df_preprocessed.shape}")
        
        # 3. Encode
        df_encoded = self.encoder.transform(df_preprocessed)
        logger.info(f"Encoded: {df_encoded.shape}")
        
        # 4. Extract actual BMI before dropping
        actual_bmi = df_encoded['latest_bmi'].values[0] if 'latest_bmi' in df_encoded.columns else None
        
        # 5. Drop unnecessary columns
        df_features = df_encoded.drop(
            columns=[col for col in self.columns_to_drop if col in df_encoded.columns],
            errors='ignore'
        )
        logger.info(f"Features after dropping: {df_features.shape}")
        
        # 6. Align features with training
        df_features = self._align_features(df_features)
        
        # 7. Scale features
        X_scaled = self.scaler.transform(df_features)
        
        # 8. Predict
        predicted_bmi = float(self.model.predict(X_scaled)[0])
        logger.info(f"Prediction: {predicted_bmi:.2f}")
        
        response = {
            'patient_id': patient_data.get('patient_practice_id') or patient_data.get('PatientID'),
            'predicted_bmi': predicted_bmi,
            'actual_bmi': float(actual_bmi) if actual_bmi else None,
            'model_name': self.model_name,
        }
        
        # 10. Calculate individual metrics if actual BMI available
        if actual_bmi:
            error = predicted_bmi - actual_bmi
            response['individual_metrics'] = {
                'error': float(error),
                'absolute_error': float(abs(error)),
                'percentage_error': float((error / actual_bmi) * 100),
                'absolute_percentage_error': float((abs(error) / actual_bmi) * 100),
            }
        
        # 11. Add model performance metrics
        response['model_metrics'] = {
            'train': self.model_metrics['train_metrics'],
            'validation': self.model_metrics['val_metrics'],
            'test': self.model_metrics['test_metrics'],
            'cv_mean_r2': self.model_metrics['cv_scores']['CV_Mean_R2'],
            'cv_std_r2': self.model_metrics['cv_scores']['CV_Std_R2'],
        }
        
        # 12. Add feature importance (NEW)
        response['feature_importance'] = self.explainer.get_top_features()
        
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