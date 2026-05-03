import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Handles feature importance extraction and explanation for trained models.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize explainer for a specific model
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.base_path = Path(__file__).parent.parent
        self.feature_importance_path = self.base_path / 'models' / 'feature_importance' / f'{model_name}_top20_features.json'
        self.feature_importance_data = None
        
        # Load feature importance
        self._load_feature_importance()
    
    def _load_feature_importance(self):
        """Load feature importance from saved JSON file"""
        try:
            with open(self.feature_importance_path, 'r') as f:
                self.feature_importance_data = json.load(f)
            logger.info(f"Feature importance loaded from: {self.feature_importance_path}")
        except FileNotFoundError:
            logger.warning(f"Feature importance file not found: {self.feature_importance_path}")
            self.feature_importance_data = None
    
    def get_top_features(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top N most important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.feature_importance_data:
            return {}
        
        top_features = self.feature_importance_data.get('top_20_features', {})
        
        # If requesting less than 20, slice the dict
        if top_n < 20:
            items = list(top_features.items())[:top_n]
            return dict(items)
        
        return top_features