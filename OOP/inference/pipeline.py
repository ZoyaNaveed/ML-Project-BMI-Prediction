import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.data_loader import MongoDataExtractor
from services.preprocessor import DataPreprocessor
from services.encoder import FeatureEncoder
from services.trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class FullTrainingPipeline:
    """
    Complete end-to-end pipeline: Data extraction → Preprocessing → Encoding → Training
    """
    
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        target_column: str = 'latest_bmi'
    ) -> None:
        """
        Initialize the full pipeline
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
            collection_name: Collection name
            target_column: Target variable for prediction
        """
        self.mongo_uri: str = mongo_uri
        self.db_name: str = db_name
        self.collection_name: str = collection_name
        self.target_column: str = target_column
        
        # Pipeline components
        self.data_extractor: Optional[MongoDataExtractor] = None
        self.preprocessor: DataPreprocessor = DataPreprocessor()
        self.encoder: FeatureEncoder = FeatureEncoder()
        self.trainer: ModelTrainer = ModelTrainer(target=target_column)
        
        # Data at each stage
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_preprocessed: Optional[pd.DataFrame] = None
        self.df_encoded: Optional[pd.DataFrame] = None
        
        # Results
        self.training_results: Optional[Dict[str, Any]] = None
    
    def run(self, save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            save_intermediate: Whether to save data at each intermediate step
            
        Returns:
            Dictionary with training results
        """
        logger.info("="*80)
        logger.info("STARTING FULL TRAINING PIPELINE")
        logger.info("="*80)
        
        start_time: datetime = datetime.now()
        
        # Step 1: Extract data from MongoDB
        self._extract_data()
        
        # Step 2: Preprocess data
        self._preprocess_data()
        
        # Step 3: Encode features
        self._encode_data()
        
        # Step 4: Train models
        self._train_models()
        
        # Save intermediate results if requested
        if save_intermediate:
            self._save_intermediate_data()
        
        # Calculate total time
        end_time: datetime = datetime.now()
        duration: float = (end_time - start_time).total_seconds()
        
        logger.info("="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        return self._get_summary()
    
    def _extract_data(self) -> None:
        """Step 1: Extract data from MongoDB"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA EXTRACTION FROM MONGODB")
        logger.info("="*80)
        
        with MongoDataExtractor(self.mongo_uri, self.db_name, self.collection_name) as extractor:
            self.df_raw = extractor.extract_all_patients()
        
        if self.df_raw is not None:
            logger.info(f"Extracted {len(self.df_raw)} patients with {len(self.df_raw.columns)} columns")
            logger.info(f"Memory usage: {self.df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def _preprocess_data(self) -> None:
        """Step 2: Preprocess data"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("="*80)
        
        if self.df_raw is None:
            raise ValueError("Raw data is None. Run _extract_data() first.")
        
        initial_count: int = len(self.df_raw)
        self.df_preprocessed = self.preprocessor.fit_transform(self.df_raw)
        final_count: int = len(self.df_preprocessed)
        
        logger.info(f"Preprocessing complete:")
        logger.info(f"  Input: {initial_count} patients")
        logger.info(f"  Output: {final_count} patients")
        logger.info(f"  Dropped: {initial_count - final_count} patients ({(initial_count - final_count)/initial_count*100:.2f}%)")
        logger.info(f"  Columns: {len(self.df_preprocessed.columns)}")
    
    def _encode_data(self) -> None:
        """Step 3: Encode features"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: FEATURE ENCODING")
        logger.info("="*80)
        
        if self.df_preprocessed is None:
            raise ValueError("Preprocessed data is None. Run _preprocess_data() first.")
        
        initial_count: int = len(self.df_preprocessed)
        self.df_encoded = self.encoder.fit_transform(self.df_preprocessed)
        final_count: int = len(self.df_encoded)
        
        logger.info(f"Encoding complete:")
        logger.info(f"  Input: {initial_count} patients")
        logger.info(f"  Output: {final_count} patients")
        logger.info(f"  Dropped: {initial_count - final_count} patients ({(initial_count - final_count)/initial_count*100:.2f}%)")
        logger.info(f"  Columns: {len(self.df_encoded.columns)}")
    
    def _train_models(self) -> None:
        """Step 4: Train all models"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("="*80)
        
        if self.df_encoded is None:
            raise ValueError("Encoded data is None. Run _encode_data() first.")
        
        self.trainer.fit(self.df_encoded)
        self.training_results = self.trainer.all_results
        
        logger.info(f"Training complete:")
        logger.info(f"  Models trained: {len(self.training_results)}")
        logger.info(f"  Features used: {len(self.trainer.feature_names)}")
    
    def _save_intermediate_data(self) -> None:
        """Save data at each pipeline stage"""
        logger.info("\n" + "="*80)
        logger.info("SAVING INTERMEDIATE DATA")
        logger.info("="*80)
        
        output_dir: Path = Path('data/pipeline_outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        if self.df_raw is not None:
            raw_path: Path = output_dir / f'01_raw_data_{timestamp}.csv'
            self.df_raw.to_csv(raw_path, index=False)
            logger.info(f"Raw data saved: {raw_path}")
        
        # Save preprocessed data
        if self.df_preprocessed is not None:
            preprocessed_path: Path = output_dir / f'02_preprocessed_data_{timestamp}.csv'
            self.df_preprocessed.to_csv(preprocessed_path, index=False)
            logger.info(f"Preprocessed data saved: {preprocessed_path}")
        
        # Save encoded data
        if self.df_encoded is not None:
            encoded_path: Path = output_dir / f'03_encoded_data_{timestamp}.csv'
            self.df_encoded.to_csv(encoded_path, index=False)
            logger.info(f"Encoded data saved: {encoded_path}")
    
    def _get_summary(self) -> Dict[str, Any]:
        """Generate pipeline summary"""
        if self.df_raw is None or self.df_preprocessed is None or self.df_encoded is None:
            raise ValueError("Pipeline not fully executed. Some data is None.")
        
        best_model_name: str
        best_results: Dict[str, Any]
        best_model_name, best_results = self.trainer.get_best_model()
        
        summary: Dict[str, Any] = {
            'pipeline_stages': {
                'extraction': {
                    'patients_extracted': len(self.df_raw),
                    'columns': len(self.df_raw.columns)
                },
                'preprocessing': {
                    'patients_in': len(self.df_raw),
                    'patients_out': len(self.df_preprocessed),
                    'patients_dropped': len(self.df_raw) - len(self.df_preprocessed)
                },
                'encoding': {
                    'patients_in': len(self.df_preprocessed),
                    'patients_out': len(self.df_encoded),
                    'patients_dropped': len(self.df_preprocessed) - len(self.df_encoded)
                },
                'training': {
                    'patients_used': len(self.df_encoded),
                    'features': len(self.trainer.feature_names),
                    'models_trained': len(self.training_results) if self.training_results else 0
                }
            },
            'best_model': {
                'name': best_model_name,
                'test_r2': best_results['test_metrics']['R2'],
                'test_rmse': best_results['test_metrics']['RMSE'],
                'test_mae': best_results['test_metrics']['MAE']
            },
            'all_models': {
                model_name: {
                    'test_r2': results['test_metrics']['R2'],
                    'test_rmse': results['test_metrics']['RMSE'],
                    'test_mae': results['test_metrics']['MAE']
                }
                for model_name, results in (self.training_results.items() if self.training_results else {}.items())
            }
        }
        
        return summary


def main() -> None:
    """Main execution function"""
    
    # Configuration
    MONGO_URI: str = "mongodb://bcu25:bcu25%40226mongo@172.16.101.226:27017/"
    DB_NAME: str = "Bootcamp_2025"
    COLLECTION: str = "Bmi_trajectory_v2"
    
    print("\n" + "="*80)
    print("BMI TRAJECTORY - FULL TRAINING PIPELINE")
    print("="*80)
    print(f"Database: {DB_NAME}")
    print(f"Collection: {COLLECTION}")
    print("="*80)
    
    # Initialize pipeline
    pipeline: FullTrainingPipeline = FullTrainingPipeline(
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION,
        target_column='latest_bmi'
    )
    
    # Run pipeline
    summary: Dict[str, Any] = pipeline.run(save_intermediate=True)
    
    # Display summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    
    print("\nData Flow:")
    print(f"  Extracted:    {summary['pipeline_stages']['extraction']['patients_extracted']} patients")
    print(f"  Preprocessed: {summary['pipeline_stages']['preprocessing']['patients_out']} patients")
    print(f"  Encoded:      {summary['pipeline_stages']['encoding']['patients_out']} patients")
    print(f"  Trained on:   {summary['pipeline_stages']['training']['patients_used']} patients")
    print(f"  Features:     {summary['pipeline_stages']['training']['features']}")
    
    print("\nBest Model:")
    print(f"  Name:    {summary['best_model']['name']}")
    print(f"  Test R²: {summary['best_model']['test_r2']:.4f}")
    print(f"  Test RMSE: {summary['best_model']['test_rmse']:.4f}")
    print(f"  Test MAE: {summary['best_model']['test_mae']:.4f}")
    
    print("\nAll Models Performance (Test Set):")
    for model_name, metrics in sorted(summary['all_models'].items(), 
                                      key=lambda x: x[1]['test_r2'], 
                                      reverse=True):
        print(f"  {model_name:25s} R²: {metrics['test_r2']:.4f}  RMSE: {metrics['test_rmse']:.4f}  MAE: {metrics['test_mae']:.4f}")
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print("\nModels saved in: models/saved_models/")
    print("Metrics saved in: models/model_metrics/")
    print("Feature importance saved in: models/feature_importance/")
    print("Intermediate data saved in: data/pipeline_outputs/")
    print("="*80)


if __name__ == "__main__":
    main()