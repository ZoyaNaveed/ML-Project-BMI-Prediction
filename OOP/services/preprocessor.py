import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all preprocessing steps for patient EMR data.
    Splits temporal features, handles missing values, creates categorical features.
    """
    
    def __init__(self):
        """Initialize preprocessor with configuration"""
        self.temporal_features = ['bmi', 'height', 'weight', 'systolic_bp', 'diastolic_bp']
        self.age_bins = [18, 30, 40, 50, 60, 70, 100, 120]
        self.age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-100', '100+']
        self.columns_to_drop = [
            'date_of_birth', 'date_of_icd_code', 'date_of_api_test_name',
            'api_test_name', 'date_of_height_value', 'date_of_weight_value',
            'bmi', 'weight_category', 'date_of_surgery_name', 'date_of_gpi',
            'height', 'weight', 'status_of_gpi'
        ]
        # Validation ranges
        self.weight_range = (30, 250)  # kg
        self.bmi_range = (15, 50)
        self.height_diff_threshold = 1.0  # cm
        
        # MAP (Mean Arterial Pressure) configuration
        self.map_bins = [0, 70, 93, 100, 106, 120, 200]
        self.map_labels = [
            'Low (<70)', 'Normal (70-93)', 'Elevated (93-100)',
            'Stage 1 (100-106)', 'Stage 2 (106-120)', 'Severe (120+)'
        ]
        
        # Columns to drop after MAP calculation
        self.bp_columns_to_drop = [
            'latest_systolic', 'm5_systolic',
            'latest_diastolic', 'm5_diastolic'
        ]
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor (placeholder for future stateful operations)
        
        Args:
            df: Training dataframe
            
        Returns:
            self
        """
        logger.info("Fitting preprocessor...")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing transformations
        
        Args:
            df: Input dataframe with raw patient data
            
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        logger.info(f"Starting preprocessing with {len(df)} patients")
        
        # 1. Split temporal features
        df = self._split_temporal_features(df)
        
        # 2. Validate height consistency
        df = self._validate_height_consistency(df)
        
        # 3. Validate weight ranges
        df = self._validate_weight_ranges(df)
        
        # 4. Validate BMI ranges
        df = self._validate_bmi_ranges(df)
        
        # 5. Calculate MAP (Mean Arterial Pressure)
        df = self._calculate_map(df)
        
        # 6. Remove MAP outliers
        df = self._remove_map_outliers(df)
        
        # 7. Create MAP groups
        df = self._create_map_groups(df)
        
        # 8. Drop blood pressure columns after MAP calculation
        df = self._drop_bp_columns(df)
        
        # 9. Handle age column
        df = self._process_age(df)
        
        # 10. Create age groups
        df = self._create_age_groups(df)
        
        # 11. Process marital status
        df = self._process_marital_status(df)
        
        # 12. Drop unnecessary columns
        df = self._drop_columns(df)
        
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        
        return df
        
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        return self.fit(df).transform(df)
    
    def _split_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split temporal features (bmi, height, weight, BP) into latest and m5 (minus 5 months)
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with split temporal features
        """
        temporal_config = {
            'bmi': ['latest_bmi', 'm5_bmi'],
            'height': ['latest_height', 'm5_height'],
            'weight': ['latest_weight', 'm5_weight'],
            'systolic_bp': ['latest_systolic', 'm5_systolic'],
            'diastolic_bp': ['latest_diastolic', 'm5_diastolic']
        }
        
        for original_col, new_cols in temporal_config.items():
            if original_col in df.columns:
                df = self._split_and_convert_temporal_feature(df, original_col, new_cols)
        
        # Drop original BP columns after splitting
        if 'systolic_bp' in df.columns:
            df = df.drop('systolic_bp', axis=1)
        if 'diastolic_bp' in df.columns:
            df = df.drop('diastolic_bp', axis=1)
        
        # df = df.drop('PatientID', axis=1)
        
        return df
    
    def _split_and_convert_temporal_feature(
        self, 
        df: pd.DataFrame, 
        col_name: str, 
        new_col_names: List[str]
    ) -> pd.DataFrame:
        """
        Generic method to split and convert a temporal feature
        
        Args:
            df: Input dataframe
            col_name: Original column name
            new_col_names: List of two new column names [latest, m5]
            
        Returns:
            Dataframe with split and converted feature
        """
        logger.info(f"Processing temporal feature: {col_name}")
        initial_count = len(df)
        
        # Split the column
        df[new_col_names] = df[col_name].str.split(';', expand=True)
        
        # Strip whitespace
        df[new_col_names[0]] = df[new_col_names[0]].str.strip()
        df[new_col_names[1]] = df[new_col_names[1]].str.strip()
        
        # Convert to numeric
        df[new_col_names[0]] = pd.to_numeric(df[new_col_names[0]], errors='coerce')
        df[new_col_names[1]] = pd.to_numeric(df[new_col_names[1]], errors='coerce')
        
        # Drop rows with missing values
        df = df.dropna(subset=new_col_names)
        
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logger.warning(
                f"Dropped {dropped_count} patients ({dropped_count/initial_count*100:.2f}%) "
                f"due to missing {col_name}"
            )
        
        return df
    
    def _process_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename age column and handle missing values
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with processed age
        """
        # Rename age_years to age if it exists
        if 'age_years' in df.columns:
            df = df.rename(columns={'age_years': 'age'})
        
        if 'age' in df.columns:
            # ADDED: Convert to numeric (handles string inputs)
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            
            initial_count = len(df)
            missing_count = df['age'].isnull().sum()
            
            if missing_count > 0:
                logger.info(f"Patients with missing age: {missing_count} ({missing_count/initial_count*100:.2f}%)")
                df = df.dropna(subset=['age'])
                logger.info(f"Dropped {missing_count} patients due to missing age")
        
        return df
    
    def _create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical and encoded age groups
        
        Args:
            df: Input dataframe with 'age' column
            
        Returns:
            Dataframe with age_groups and age_groups_encoded columns
        """
        if 'age' not in df.columns:
            logger.warning("'age' column not found. Skipping age group creation.")
            return df
        
        logger.info("Creating age groups...")
        
        # Create age groups
        df['age_groups'] = pd.cut(
            df['age'], 
            bins=self.age_bins, 
            labels=self.age_labels, 
            right=False
        )
        
        # Create encoded version
        df['age_groups_encoded'] = df['age_groups'].cat.codes
    
        # Log distribution
        age_dist = df['age_groups'].value_counts().sort_index()
        logger.info("Age Groups Distribution:")
        for group, count in age_dist.items():
            percentage = (count / len(df) * 100)
            logger.info(f"  {group}: {count} patients ({percentage:.2f}%)")
        
        # DROP the categorical column, keep only encoded
        df = df.drop('age_groups', axis=1)
        df = df.drop('age', axis=1)

        df = df.rename(columns={'age_groups_encoded': 'age_groups'})
        
        return df
    
    def _process_marital_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process marital status: fill missing values and encode as binary
        Single group (0): Single, Separated, Unknown, Widowed, Divorced
        Married group (1): Married, Partnered
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with encoded marital_status
        """
        if 'marital_status' not in df.columns:
            logger.warning("'marital_status' column not found. Skipping.")
            return df
        
        logger.info("Processing marital status...")
        
        # Fill missing values with 'Unknown'
        df['marital_status'] = df['marital_status'].fillna('Unknown')
        
        # Create mapping
        marital_mapping = {
            'single': 0,
            'separated': 0,
            'unknown': 0,
            'widowed': 0,
            'divorced': 0,
            'married': 1,
            'partnered': 1
        }
        
        # Encode marital status
        df['marital_status_encoded'] = (
            df['marital_status']
            .str.strip()
            .str.lower()
            .map(marital_mapping)
        )
        
        # Drop original and rename
        df = df.drop('marital_status', axis=1)
        df = df.rename(columns={'marital_status_encoded': 'marital_status'})
        
        # Log distribution
        distribution = df['marital_status'].value_counts()
        logger.info(f"Marital status distribution:\n  Single/Others (0): {distribution.get(0, 0)}\n  Married (1): {distribution.get(1, 0)}")
        
        return df
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop unnecessary columns
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with columns dropped
        """
        existing_cols_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        
        if existing_cols_to_drop:
            logger.info(f"Dropping {len(existing_cols_to_drop)} columns: {existing_cols_to_drop}")
            df = df.drop(existing_cols_to_drop, axis=1)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature names after preprocessing
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            List of feature column names
        """
        return df.columns.tolist()
    
    def _validate_height_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that height difference between latest and m5 is <= 1cm
        Adults' height shouldn't change significantly in 5 months
        
        Args:
            df: Input dataframe
            
        Returns:
            Filtered dataframe with consistent heights
        """
        if 'latest_height' not in df.columns or 'm5_height' not in df.columns:
            logger.warning("Height columns not found. Skipping height validation.")
            return df
        
        logger.info("Validating height consistency...")
        initial_count = len(df)
        
        # Calculate height difference
        df['height_difference'] = abs(df['latest_height'] - df['m5_height'])
        
        # Filter valid heights
        df = df[
            (df['height_difference'] >= 0) & 
            (df['height_difference'] <= self.height_diff_threshold)
        ].copy()
        
        dropped = initial_count - len(df)
        logger.info(
            f"Height validation: Dropped {dropped} patients ({dropped/initial_count*100:.2f}%) "
            f"with height difference > {self.height_diff_threshold}cm"
        )
        logger.info(
            f"Height difference range: {df['height_difference'].min():.4f} - "
            f"{df['height_difference'].max():.4f} cm"
        )
        
        return df
    
    def _validate_weight_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate weight ranges (30-250 kg) for both latest and m5
        
        Args:
            df: Input dataframe
            
        Returns:
            Filtered dataframe with valid weights
        """
        if 'latest_weight' not in df.columns or 'm5_weight' not in df.columns:
            logger.warning("Weight columns not found. Skipping weight validation.")
            return df
        
        logger.info(f"Validating weight ranges ({self.weight_range[0]}-{self.weight_range[1]} kg)...")
        initial_count = len(df)
        
        df = df[
            (df['latest_weight'] >= self.weight_range[0]) & 
            (df['latest_weight'] <= self.weight_range[1]) &
            (df['m5_weight'] >= self.weight_range[0]) & 
            (df['m5_weight'] <= self.weight_range[1])
        ]
        
        dropped = initial_count - len(df)
        logger.info(
            f"Weight validation: Dropped {dropped} patients ({dropped/initial_count*100:.2f}%) "
            f"with weights outside valid range"
        )
        
        return df
    
    def _validate_bmi_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate BMI ranges (15-50) for both latest and m5
        
        Args:
            df: Input dataframe
            
        Returns:
            Filtered dataframe with valid BMI values
        """
        if 'latest_bmi' not in df.columns or 'm5_bmi' not in df.columns:
            logger.warning("BMI columns not found. Skipping BMI validation.")
            return df
        
        logger.info(f"Validating BMI ranges ({self.bmi_range[0]}-{self.bmi_range[1]})...")
        initial_count = len(df)
        
        df = df[
            (df['latest_bmi'] >= self.bmi_range[0]) & 
            (df['latest_bmi'] <= self.bmi_range[1]) &
            (df['m5_bmi'] >= self.bmi_range[0]) & 
            (df['m5_bmi'] <= self.bmi_range[1])
        ]
        
        dropped = initial_count - len(df)
        logger.info(
            f"BMI validation: Dropped {dropped} patients ({dropped/initial_count*100:.2f}%) "
            f"with BMI outside valid range"
        )
        
        return df
    
    def _calculate_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Mean Arterial Pressure (MAP) for both latest and m5
        Formula: MAP = DP + 1/3(SP - DP)
        
        Args:
            df: Input dataframe with systolic and diastolic BP
            
        Returns:
            Dataframe with MAP columns added
        """
        required_cols = ['latest_systolic', 'latest_diastolic', 'm5_systolic', 'm5_diastolic']
        
        if not all(col in df.columns for col in required_cols):
            logger.warning("Blood pressure columns not found. Skipping MAP calculation.")
            return df
        
        logger.info("Calculating Mean Arterial Pressure (MAP)...")
        
        # Calculate latest MAP
        df['latest_MAP'] = (
            df['latest_diastolic'] + 
            (1/3) * (df['latest_systolic'] - df['latest_diastolic'])
        )
        
        # Calculate m5 MAP
        df['m5_MAP'] = (
            df['m5_diastolic'] + 
            (1/3) * (df['m5_systolic'] - df['m5_diastolic'])
        )
        
        logger.info("MAP Statistics:")
        logger.info(f"Latest MAP - Mean: {df['latest_MAP'].mean():.2f}, "
                   f"Std: {df['latest_MAP'].std():.2f}")
        logger.info(f"M5 MAP - Mean: {df['m5_MAP'].mean():.2f}, "
                   f"Std: {df['m5_MAP'].std():.2f}")
        
        return df
    
    def _remove_map_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from MAP using IQR method (1.5 * IQR)
        
        Args:
            df: Input dataframe with MAP columns
            
        Returns:
            Dataframe with MAP outliers removed
        """
        if 'latest_MAP' not in df.columns or 'm5_MAP' not in df.columns:
            logger.warning("MAP columns not found. Skipping outlier removal.")
            return df
        
        logger.info("Removing MAP outliers using IQR method...")
        initial_count = len(df)
        
        # Calculate IQR for latest_MAP
        Q1_latest = df['latest_MAP'].quantile(0.25)
        Q3_latest = df['latest_MAP'].quantile(0.75)
        IQR_latest = Q3_latest - Q1_latest
        
        # Calculate IQR for m5_MAP
        Q1_m5 = df['m5_MAP'].quantile(0.25)
        Q3_m5 = df['m5_MAP'].quantile(0.75)
        IQR_m5 = Q3_m5 - Q1_m5
        
        # Filter outliers
        df = df[
            (df['latest_MAP'].between(
                Q1_latest - 1.5*IQR_latest, 
                Q3_latest + 1.5*IQR_latest
            )) &
            (df['m5_MAP'].between(
                Q1_m5 - 1.5*IQR_m5, 
                Q3_m5 + 1.5*IQR_m5
            ))
        ].copy()
        
        removed = initial_count - len(df)
        logger.info(
            f"Removed {removed} MAP outliers ({removed/initial_count*100:.2f}%). "
            f"Retained: {len(df)} patients"
        )
        
        return df
    
    def _create_map_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical MAP groups based on clinical categories
        Categories: Low, Normal, Elevated, Stage 1 HTN, Stage 2 HTN, Severe
        
        Args:
            df: Input dataframe with MAP columns
            
        Returns:
            Dataframe with MAP group columns (encoded)
        """
        if 'latest_MAP' not in df.columns or 'm5_MAP' not in df.columns:
            logger.warning("MAP columns not found. Skipping MAP group creation.")
            return df
        
        logger.info("Creating MAP groups based on clinical categories...")
        
        # Create latest_MAP_groups
        df['latest_MAP_groups_temp'] = pd.cut(
            df['latest_MAP'],
            bins=self.map_bins,
            labels=self.map_labels,
            right=False
        )
        df['latest_MAP_groups'] = df['latest_MAP_groups_temp'].cat.codes
        
        # Log distribution
        logger.info("Latest MAP Groups Distribution:")
        for group, count in df['latest_MAP_groups_temp'].value_counts().sort_index().items():
            percentage = (count / len(df) * 100)
            logger.info(f"  {group}: {count} patients ({percentage:.2f}%)")
        
        # Create m5_MAP_groups
        df['m5_MAP_groups_temp'] = pd.cut(
            df['m5_MAP'],
            bins=self.map_bins,
            labels=self.map_labels,
            right=False
        )
        df['m5_MAP_groups'] = df['m5_MAP_groups_temp'].cat.codes
        
        # Log distribution
        logger.info("M5 MAP Groups Distribution:")
        for group, count in df['m5_MAP_groups_temp'].value_counts().sort_index().items():
            percentage = (count / len(df) * 100)
            logger.info(f"  {group}: {count} patients ({percentage:.2f}%)")
        
        # Drop temporary categorical columns (keep only encoded)
        df = df.drop(['latest_MAP_groups_temp', 'm5_MAP_groups_temp'], axis=1)
        
        # Drop original MAP values (keep only groups)
        df = df.drop(['latest_MAP', 'm5_MAP'], axis=1)
        
        # Also drop height_difference if it exists
        if 'height_difference' in df.columns:
            df = df.drop('height_difference', axis=1)
        
        return df
    
    def _drop_bp_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop blood pressure columns after MAP calculation
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with BP columns dropped
        """
        existing_bp_cols = [col for col in self.bp_columns_to_drop if col in df.columns]
        
        if existing_bp_cols:
            logger.info(f"Dropping BP columns: {existing_bp_cols}")
            df = df.drop(existing_bp_cols, axis=1)
        
        return df
