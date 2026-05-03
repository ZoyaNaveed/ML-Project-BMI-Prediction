import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Type aliases for better readability
ModelType = Union[
    XGBRegressor, Ridge, Lasso, ElasticNet, 
    RandomForestRegressor, GradientBoostingRegressor, LinearRegression
]
MetricsDict = Dict[str, Optional[float]]
ModelResultsDict = Dict[str, Any]


class ModelTrainer:
    """
    Handles training of multiple regression models for BMI prediction.
    Includes data splitting, scaling, cross-validation, and model persistence.
    """
    
    def __init__(
        self, 
        target: str = 'latest_bmi', 
        test_size: float = 0.2, 
        val_size: float = 0.25, 
        random_state: int = 42
    ) -> None:
        """
        Initialize the trainer
        
        Args:
            target: Target variable name
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation
            random_state: Random seed for reproducibility
        """
        self.target: str = target
        self.test_size: float = test_size
        self.val_size: float = val_size
        self.random_state: int = random_state
        
        # Columns to drop from features
        self.columns_to_drop: List[str] = [
            'latest_bmi',  # Target variable
            'patient_practice_id',
            'practice_patient_id',
            'latest_weight',
            'latest_height',
            'm5_weight',
            'm5_height',
            'latest_MAP_groups_encoded',
            'PCOS',
            'Schizophrenia',
            'Bipolar',
            'history_of_KidneyDisease',
            'Nissen_Fundoplication',
            'Obesity',
        ]
        
        # Model configurations
        self.model_configs: Dict[str, Dict[str, ModelType]] = {
            'XGBoost': {
                'model': XGBRegressor(
                    n_estimators=50,
                    learning_rate=0.1,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=random_state,
                    n_jobs=-1
                )
            },
            'Ridge_Regression': {
                'model': Ridge(alpha=1.0, random_state=random_state)
            },
            'Lasso_Regression': {
                'model': Lasso(alpha=0.01, random_state=random_state, max_iter=10000)
            },
            'ElasticNet_Regression': {
                'model': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=10000)
            },
            'Random_Forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=random_state,
                    n_jobs=-1
                )
            },
            'Gradient_Boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=random_state
                )
            }
        }
        
        # Storage for trained components
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.X_train_scaled: Optional[NDArray[np.float64]] = None
        self.X_val_scaled: Optional[NDArray[np.float64]] = None
        self.X_test_scaled: Optional[NDArray[np.float64]] = None
        self.all_results: Dict[str, ModelResultsDict] = {}
        
        # Create directories
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('data/models/metrics', exist_ok=True)
        os.makedirs('data/models/feature_importance', exist_ok=True)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from dataframe
        
        Args:
            df: Input dataframe with all features
            
        Returns:
            X, y: Features and target
        """
        logger.info("Preparing data...")
        
        # Filter out rows with missing target
        df_model: pd.DataFrame = df[df[self.target].notna()].copy()
        logger.info(f"Total patients with target: {len(df_model)}")
        
        # Separate target
        y: pd.Series = df_model[self.target]
        
        # Drop target and unnecessary columns
        X: pd.DataFrame = df_model.drop(
            columns=[col for col in self.columns_to_drop if col in df_model.columns],
            errors='ignore'
        )
        
        # Keep only numeric columns
        numeric_cols: List[str] = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        # Fill missing values with median
        X = X.fillna(X.median())
        
        # Remove zero variance features
        selector: VarianceThreshold = VarianceThreshold(threshold=0)
        X_variance: NDArray[np.float64] = selector.fit_transform(X)
        X = pd.DataFrame(X_variance, columns=X.columns[selector.get_support()], index=X.index)
        
        logger.info(f"Number of features: {X.shape[1]}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into train, validation, and test sets (60/20/20)
        
        Args:
            X: Features
            y: Target
        """
        logger.info("Splitting data...")
        
        # First split: 80% train+val, 20% test
        X_temp: pd.DataFrame
        y_temp: pd.Series
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Second split: 60% train, 20% val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_size, random_state=self.random_state
        )
        
        logger.info(f"Training set:   {len(self.X_train)} patients ({len(self.X_train)/len(X)*100:.1f}%)")
        logger.info(f"Validation set: {len(self.X_val)} patients ({len(self.X_val)/len(X)*100:.1f}%)")
        logger.info(f"Test set:       {len(self.X_test)} patients ({len(self.X_test)/len(X)*100:.1f}%)")
    
    def scale_features(self) -> None:
        """Scale features using StandardScaler"""
        logger.info("Scaling features...")
        
        if self.X_train is None or self.X_val is None or self.X_test is None:
            raise ValueError("Data must be split before scaling. Run split_data() first.")
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def calculate_metrics(
        self, 
        y_true: Union[pd.Series, NDArray[np.float64]], 
        y_pred: NDArray[np.float64]
    ) -> MetricsDict:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary containing various metrics
        """
        metrics: MetricsDict = {}
        
        # Convert to numpy arrays if needed
        y_true_arr: NDArray[np.float64] = np.array(y_true)
        
        metrics['MAE'] = float(mean_absolute_error(y_true_arr, y_pred))
        metrics['MSE'] = float(mean_squared_error(y_true_arr, y_pred))
        metrics['RMSE'] = float(np.sqrt(metrics['MSE']))
        metrics['R2'] = float(r2_score(y_true_arr, y_pred))
        
        # MAPE
        mask: NDArray[np.bool_] = y_true_arr != 0
        if mask.sum() > 0:
            metrics['MAPE'] = float(np.mean(np.abs((y_true_arr[mask] - y_pred[mask]) / y_true_arr[mask])) * 100)
        else:
            metrics['MAPE'] = None
        
        # MSLE and RMSLE
        try:
            y_true_log: NDArray[np.float64] = np.log1p(np.maximum(y_true_arr, 0))
            y_pred_log: NDArray[np.float64] = np.log1p(np.maximum(y_pred, 0))
            metrics['MSLE'] = float(mean_squared_error(y_true_log, y_pred_log))
            metrics['RMSLE'] = float(np.sqrt(metrics['MSLE']))
        except Exception:
            metrics['MSLE'] = None
            metrics['RMSLE'] = None
        
        return metrics
    
    def extract_feature_importance(self, model: ModelType, model_name: str) -> Dict[str, Any]:
        """
        Extract and save feature importance
        
        Args:
            model: Trained model
            model_name: Name of the model
            
        Returns:
            Dictionary containing feature importance data
        """
        logger.info(f"Extracting feature importance for {model_name}...")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available. Run prepare_data() first.")
        
        importance_dict: Dict[str, float] = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances: NDArray[np.float64] = model.feature_importances_
            importance_dict = {feat: float(imp) for feat, imp in zip(self.feature_names, importances)}
        elif hasattr(model, 'coef_'):
            # Linear models
            importances_linear: NDArray[np.float64] = np.abs(model.coef_)
            importance_dict = {feat: float(imp) for feat, imp in zip(self.feature_names, importances_linear)}
        
        # Sort and get top 20
        sorted_importance: List[Tuple[str, float]] = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )
        top_20: Dict[str, float] = dict(sorted_importance[:20])
        
        # Log top 5
        logger.info(f"Top 5 features for {model_name}:")
        for i, (feat, imp) in enumerate(list(top_20.items())[:5], 1):
            logger.info(f"  {i}. {feat}: {imp:.6f}")
        
        feature_importance_data: Dict[str, Any] = {
            'model_name': model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'top_20_features': top_20,
            'feature_names_ordered': list(top_20.keys()),
            'importance_values': list(top_20.values())
        }
        
        # Save as JSON
        importance_filename: str = f'data/models/feature_importance/{model_name}_top20_features.json'
        with open(importance_filename, 'w') as f:
            json.dump(feature_importance_data, f, indent=4)
        
        # Save as CSV
        importance_df: pd.DataFrame = pd.DataFrame({
            'Feature': list(top_20.keys()),
            'Importance': list(top_20.values())
        })
        csv_filename: str = f'data/models/feature_importance/{model_name}_top20_features.csv'
        importance_df.to_csv(csv_filename, index=False)
        
        return feature_importance_data
    
    def train_single_model(self, model_name: str, model: ModelType) -> ModelResultsDict:
        """
        Train a single model with cross-validation
        
        Args:
            model_name: Name of the model
            model: Model instance to train
            
        Returns:
            Dictionary containing training results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING: {model_name}")
        logger.info(f"{'='*80}")
        
        if (self.X_train_scaled is None or self.y_train is None or 
            self.X_val_scaled is None or self.y_val is None or
            self.X_test_scaled is None or self.y_test is None):
            raise ValueError("Data must be prepared and scaled before training.")
        
        # Cross-validation
        kf: KFold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores: NDArray[np.float64] = cross_val_score(
            model, self.X_train_scaled, self.y_train,
            cv=kf, scoring='r2', n_jobs=-1
        )
        cv_mean: float = float(cv_scores.mean())
        cv_std: float = float(cv_scores.std())
        
        logger.info(f"5-Fold CV R² Scores: {cv_scores}")
        logger.info(f"CV Mean R²: {cv_mean:.4f} (±{cv_std:.4f})")
        
        # Train model
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_train_pred: NDArray[np.float64] = model.predict(self.X_train_scaled)
        y_val_pred: NDArray[np.float64] = model.predict(self.X_val_scaled)
        y_test_pred: NDArray[np.float64] = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_metrics: MetricsDict = self.calculate_metrics(self.y_train, y_train_pred)
        val_metrics: MetricsDict = self.calculate_metrics(self.y_val, y_val_pred)
        test_metrics: MetricsDict = self.calculate_metrics(self.y_test, y_test_pred)
        
        # Log metrics
        logger.info(f"\nMetrics Summary:")
        logger.info(f"  Train R²: {train_metrics['R2']:.4f} | Val R²: {val_metrics['R2']:.4f} | Test R²: {test_metrics['R2']:.4f}")
        logger.info(f"  Train RMSE: {train_metrics['RMSE']:.4f} | Val RMSE: {val_metrics['RMSE']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f}")
        logger.info(f"  Train MAE: {train_metrics['MAE']:.4f} | Val MAE: {val_metrics['MAE']:.4f} | Test MAE: {test_metrics['MAE']:.4f}")
        
        # Store results
        model_results: ModelResultsDict = {
            'model_name': model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cv_scores': {
                'CV_Mean_R2': cv_mean,
                'CV_Std_R2': cv_std,
                'CV_Fold_Scores': [float(score) for score in cv_scores]
            },
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'overfitting': {
                'Train_Val_R2_Diff': float(train_metrics['R2'] - val_metrics['R2']),
                'Train_Test_R2_Diff': float(train_metrics['R2'] - test_metrics['R2'])
            }
        }
        
        # Save model
        model_filename: str = f'data/models/{model_name}_model.pkl'
        joblib.dump(model, model_filename)
        logger.info(f"\nModel saved: {model_filename}")
        
        # Save metrics
        metrics_filename: str = f'data/models/metrics/{model_name}_metrics.json'
        with open(metrics_filename, 'w') as f:
            json.dump(model_results, f, indent=4)
        logger.info(f"Metrics saved: {metrics_filename}")
        
        # Extract feature importance
        self.extract_feature_importance(model, model_name)
        
        return model_results
    
    def train_all_models(self) -> Dict[str, ModelResultsDict]:
        """
        Train all configured models
        
        Returns:
            Dictionary containing results for all models
        """
        logger.info("\n" + "="*80)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*80)
        
        for model_name, config in self.model_configs.items():
            model: ModelType = config['model']
            results: ModelResultsDict = self.train_single_model(model_name, model)
            self.all_results[model_name] = results
        
        logger.info("\n" + "="*80)
        logger.info("ALL MODELS TRAINED")
        logger.info("="*80)
        
        return self.all_results
    
    def save_artifacts(self) -> None:
        """Save scaler and feature names"""
        if self.scaler is None:
            raise ValueError("Scaler not available. Run scale_features() first.")
        if self.feature_names is None:
            raise ValueError("Feature names not available. Run prepare_data() first.")
        
        # Save scaler
        joblib.dump(self.scaler, 'data/models/scaler.pkl')
        logger.info("Scaler saved: data/models/scaler.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, 'data/models/feature_names.pkl')
        logger.info("Feature names saved: data/models/feature_names.pkl")
    
    def fit(self, df: pd.DataFrame) -> 'ModelTrainer':
        """
        Complete training pipeline
        
        Args:
            df: Input dataframe (preprocessed and encoded)
            
        Returns:
            Self for method chaining
        """
        # Prepare data
        X: pd.DataFrame
        y: pd.Series
        X, y = self.prepare_data(df)
        
        # Split data
        self.split_data(X, y)
        
        # Scale features
        self.scale_features()
        
        # Train all models
        self.train_all_models()
        
        # Save artifacts
        self.save_artifacts()
        
        return self
    
    def get_best_model(self) -> Tuple[str, ModelResultsDict]:
        """
        Get the best performing model based on test R2
        
        Returns:
            Tuple of (model_name, model_results)
            
        Raises:
            ValueError: If no models have been trained
        """
        if not self.all_results:
            raise ValueError("No models have been trained yet. Run fit() first.")
        
        best_model_name: str = max(
            self.all_results.keys(),
            key=lambda x: self.all_results[x]['test_metrics']['R2']
        )
        
        return best_model_name, self.all_results[best_model_name]