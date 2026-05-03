from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, Tuple, Optional, Union
import logging
import threading
from datetime import datetime
from inference.pipeline import FullTrainingPipeline
from inference.predictor import BMIPredictor
from inference.classifier import BMIClassifier

logger = logging.getLogger(__name__)

prediction_bp: Blueprint = Blueprint('prediction', __name__)

# Type aliases
JsonResponse = Tuple[Response, int]
TrainingStatusDict = Dict[str, Optional[Union[bool, str, int, datetime, Dict[str, Any]]]]

# Cache for predictors (to avoid reloading models repeatedly)
_predictors: Dict[str, BMIPredictor] = {}
_classifiers: Dict[str, BMIClassifier] = {}

# Store training status
training_status: TrainingStatusDict = {
    'is_training': False,
    'current_stage': None,
    'progress': 0,
    'started_at': None,
    'completed_at': None,
    'error': None,
    'summary': None
}


def get_predictor(model_name: str) -> BMIPredictor:
    """
    Get or create predictor for the specified model
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        BMIPredictor instance for the specified model
    """
    if model_name not in _predictors:
        logger.info(f"Loading predictor for model: {model_name}")
        _predictors[model_name] = BMIPredictor(model_name=model_name)
    return _predictors[model_name]

def get_classifier(model_name: str) -> BMIClassifier:
    """Get or create classifier for the specified model"""
    if model_name not in _classifiers:
        logger.info(f"Loading classifier for model: {model_name}")
        _classifiers[model_name] = BMIClassifier(model_name=model_name)
    return _classifiers[model_name]


@prediction_bp.route('/predict', methods=['POST'])
def predict_bmi() -> JsonResponse:
    """
    Predict BMI for a patient
    
    Request body should contain:
    - patient_data: Patient information
    - model_name: (optional) Name of model to use (default: 'XGBoost')
    - include_feature_importance: (optional) Include feature importance (default: True)
    
    Returns:
        JSON response with prediction, actual BMI, individual metrics, 
        model metrics, and feature importance
    """
    try:
        # Get request data
        request_data: Optional[Dict[str, Any]] = request.get_json()
        
        if not request_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract model_name (default to XGBoost if not provided)
        model_name: str = request_data.get('model_name', 'XGBoost')
        
        # Extract patient_data
        patient_data: Optional[Dict[str, Any]] = request_data.get('patient_data')
        
        if not patient_data:
            patient_data = {k: v for k, v in request_data.items() 
                          if k not in ['model_name', 'include_feature_importance']}
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Validate model_name
        valid_models: list[str] = [
            'XGBoost', 'Ridge_Regression', 'Lasso_Regression', 
            'ElasticNet_Regression', 'Random_Forest', 'Gradient_Boosting'
        ]
        
        if model_name not in valid_models:
            return jsonify({
                'error': f'Invalid model name. Valid options: {valid_models}'
            }), 400
        
        # Get predictor for the requested model
        predictor: BMIPredictor = get_predictor(model_name)
        
        # Make prediction
        result: Dict[str, Any] = predictor.predict(patient_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    

@prediction_bp.route('/train', methods=['POST'])
def train_models() -> JsonResponse:
    """
    Run the full training pipeline: Extract → Preprocess → Encode → Train
    
    Request body (optional):
    {
        "mongo_uri": "mongodb://...",
        "db_name": "Bootcamp_2025",
        "collection_name": "Bmi_trajectory_v2",
        "target_column": "latest_bmi"
    }
    
    Returns:
        JSON response with training summary and model performance
    """
    try:
        logger.info("Starting training pipeline...")
        
        # Get configuration from request or use defaults
        config: Dict[str, Any] = request.get_json() or {}
        
        mongo_uri: str = config.get(
            'mongo_uri', 
            'mongodb://bcu25:bcu25%40226mongo@172.16.101.226:27017/'
        )
        db_name: str = config.get('db_name', 'Bootcamp_2025')
        collection_name: str = config.get('collection_name', 'Bmi_trajectory_v2')
        target_column: str = config.get('target_column', 'latest_bmi')
        
        # Initialize and run pipeline
        pipeline: FullTrainingPipeline = FullTrainingPipeline(
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name=collection_name,
            target_column=target_column
        )
        
        summary: Dict[str, Any] = pipeline.run(save_intermediate=True)
        
        # Clear predictor cache so new models are loaded
        _predictors.clear()
        
        return jsonify({
            'message': 'Training completed successfully',
            'summary': summary
        }), 200
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


    
@prediction_bp.route('/classify', methods=['POST'])
def classify_bmi() -> jsonify:
    """
    Classify BMI direction (increase/decrease) for a single patient.

    Request body should contain:
    - patient_data: Patient information (dict with required features)
    - model_name: (optional) Name of model to use (default: 'XGBoost_Classifier')

    Returns:
        JSON response with predicted class, probabilities, confidence, and metrics
    """
    try:
        # Extract JSON data from request
        request_data: Optional[Dict[str, Any]] = request.get_json()
        if not request_data:
            return jsonify({'error': 'No data provided in request body'}), 400

        # Extract model_name, default to 'XGBoost_Classifier'
        model_name: str = request_data.get('model_name', 'XGBoost_Classifier')

        # Extract patient_data
        patient_data: Optional[Dict[str, Any]] = request_data.get('patient_data')
        if not patient_data:
            # Fallback: Use all request data except model_name
            patient_data = {k: v for k, v in request_data.items() if k != 'model_name'}

        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400

        # Validate model_name
        valid_models: list[str] = ['XGBoost_Classifier', 'Gradient_Boosting_Classifier']
        if model_name not in valid_models:
            return jsonify({
                'error': f'Invalid model name. Valid options: {valid_models}'
            }), 400

        # Get classifier for the requested model
        classifier: BMIClassifier = get_classifier(model_name)

        # Make prediction
        result: Dict[str, Any] = classifier.predict(patient_data)

        return jsonify(result), 200

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@prediction_bp.route('/training/status', methods=['GET'])
def get_training_status() -> JsonResponse:
    """
    Get current training status
    
    Returns:
        JSON response with training status information
    """
    try:
        return jsonify(training_status), 200
        
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@prediction_bp.route('/health', methods=['GET'])
def health_check() -> JsonResponse:
    """
    Health check endpoint
    
    Returns:
        JSON response with service health status
    """
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'BMI Prediction API',
            'timestamp': datetime.now().isoformat(),
            'loaded_models': list(_predictors.keys())
        }), 200
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500