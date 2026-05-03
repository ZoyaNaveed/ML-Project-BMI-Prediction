import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OOP.services.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample patient data for testing"""
        np.random.seed(42)
        n_patients = 100
        
        data = {
            'patient_id': range(1, n_patients + 1),
            'age_years': np.random.randint(20, 80, n_patients),
            'marital_status': np.random.choice(
                ['Single', 'Married', 'Divorced', 'Widowed', None], 
                n_patients
            ),
            # Temporal features with semicolon separation
            'bmi': [f"{np.random.uniform(18, 35):.1f};{np.random.uniform(18, 35):.1f}" 
                    for _ in range(n_patients)],
            'height': [f"{np.random.uniform(150, 190):.1f};{np.random.uniform(150, 190):.1f}" 
                       for _ in range(n_patients)],
            'weight': [f"{np.random.uniform(50, 100):.1f};{np.random.uniform(50, 100):.1f}" 
                       for _ in range(n_patients)],
            'systolic_bp': [f"{np.random.uniform(110, 140):.0f};{np.random.uniform(110, 140):.0f}" 
                            for _ in range(n_patients)],
            'diastolic_bp': [f"{np.random.uniform(70, 90):.0f};{np.random.uniform(70, 90):.0f}" 
                             for _ in range(n_patients)],
            # Columns to be dropped
            'date_of_birth': pd.date_range('1950-01-01', periods=n_patients),
            'date_of_icd_code': pd.date_range('2020-01-01', periods=n_patients),
            'weight_category': np.random.choice(['Normal', 'Overweight'], n_patients),
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return DataPreprocessor()
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor.weight_range == (30, 250)
        assert preprocessor.bmi_range == (15, 50)
        assert preprocessor.height_diff_threshold == 1.0
        assert len(preprocessor.map_bins) == 7
    
    def test_split_temporal_features(self, preprocessor, sample_data):
        """Test splitting of temporal features"""
        df = preprocessor._split_temporal_features(sample_data.copy())
        
        # Check that new columns exist
        assert 'latest_bmi' in df.columns
        assert 'm5_bmi' in df.columns
        assert 'latest_height' in df.columns
        assert 'm5_height' in df.columns
        
        # Check that values are numeric
        assert pd.api.types.is_numeric_dtype(df['latest_bmi'])
        assert pd.api.types.is_numeric_dtype(df['m5_bmi'])
        
        # Check that original BP columns are dropped
        assert 'systolic_bp' not in df.columns
        assert 'diastolic_bp' not in df.columns
    
    def test_height_validation(self, preprocessor):
        """Test height consistency validation"""
        data = pd.DataFrame({
            'latest_height': [170.0, 170.0, 170.0],
            'm5_height': [170.0, 170.5, 172.0],  # Last one should be dropped
        })
        
        df = preprocessor._validate_height_consistency(data)
        
        # Should keep first two, drop last one
        assert len(df) == 2
        assert 'height_difference' in df.columns
    
    def test_weight_validation(self, preprocessor):
        """Test weight range validation"""
        data = pd.DataFrame({
            'latest_weight': [70.0, 25.0, 300.0],  # Second and third invalid
            'm5_weight': [70.0, 70.0, 70.0],
        })
        
        df = preprocessor._validate_weight_ranges(data)
        
        # Should keep only first row
        assert len(df) == 1
    
    def test_bmi_validation(self, preprocessor):
        """Test BMI range validation"""
        data = pd.DataFrame({
            'latest_bmi': [25.0, 10.0, 60.0],  # Second and third invalid
            'm5_bmi': [25.0, 25.0, 25.0],
        })
        
        df = preprocessor._validate_bmi_ranges(data)
        
        # Should keep only first row
        assert len(df) == 1
    
    def test_map_calculation(self, preprocessor):
        """Test Mean Arterial Pressure calculation"""
        data = pd.DataFrame({
            'latest_systolic': [120.0, 130.0],
            'latest_diastolic': [80.0, 85.0],
            'm5_systolic': [118.0, 128.0],
            'm5_diastolic': [78.0, 83.0],
        })
        
        df = preprocessor._calculate_map(data)
        
        # Check MAP columns exist
        assert 'latest_MAP' in df.columns
        assert 'm5_MAP' in df.columns
        
        # Verify calculation for first row: 80 + 1/3(120 - 80) = 93.33
        expected_map = 80.0 + (1/3) * (120.0 - 80.0)
        assert abs(df['latest_MAP'].iloc[0] - expected_map) < 0.01
    
    def test_age_processing(self, preprocessor):
        """Test age processing and group creation"""
        data = pd.DataFrame({
            'age_years': [25, 35, 55, 75, None],
        })
        
        df = preprocessor._process_age(data.copy())
        
        # Check age column renamed
        assert 'age' in df.columns
        assert 'age_years' not in df.columns
        
        # Check NaN dropped
        assert len(df) == 4
        
        # Test age groups
        df = preprocessor._create_age_groups(df)
        assert 'age_groups' in df.columns
        assert pd.api.types.is_numeric_dtype(df['age_groups'])
    
    def test_marital_status_processing(self, preprocessor):
        """Test marital status encoding"""
        data = pd.DataFrame({
            'marital_status': ['Single', 'Married', 'Divorced', None, 'Partnered']
        })
        
        df = preprocessor._process_marital_status(data)
        
        # Check encoding
        assert df['marital_status'].iloc[0] == 0  # Single
        assert df['marital_status'].iloc[1] == 1  # Married
        assert df['marital_status'].iloc[2] == 0  # Divorced
        assert df['marital_status'].iloc[3] == 0  # Unknown (was None)
        assert df['marital_status'].iloc[4] == 1  # Partnered
    
    def test_map_groups_creation(self, preprocessor):
        """Test MAP groups creation"""
        data = pd.DataFrame({
            'latest_MAP': [65, 85, 95, 103, 110, 125],
            'm5_MAP': [65, 85, 95, 103, 110, 125],
        })
        
        df = preprocessor._create_map_groups(data)
        
        # Check groups exist and are numeric
        assert 'latest_MAP_groups' in df.columns
        assert 'm5_MAP_groups' in df.columns
        assert pd.api.types.is_numeric_dtype(df['latest_MAP_groups'])
        
        # Check original MAP columns dropped
        assert 'latest_MAP' not in df.columns
        assert 'm5_MAP' not in df.columns
    
    def test_full_pipeline(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline"""
        initial_shape = sample_data.shape
        
        df_processed = preprocessor.fit_transform(sample_data)
        
        # Check that processing completed
        assert len(df_processed) > 0
        assert len(df_processed) <= len(sample_data)
        
        # Check key features exist
        expected_features = [
            'age', 'marital_status', 'latest_bmi', 'm5_bmi',
            'latest_weight', 'm5_weight', 'latest_height', 'm5_height'
        ]
        
        for feature in expected_features:
            if feature in ['age', 'marital_status']:
                continue  # These might be transformed
            assert feature in df_processed.columns, f"Missing feature: {feature}"
        
        # Check that unwanted columns are dropped
        unwanted_cols = ['date_of_birth', 'weight_category', 'systolic_bp']
        for col in unwanted_cols:
            assert col not in df_processed.columns, f"Column should be dropped: {col}"
        
        # Check no NaN in critical columns (after preprocessing)
        critical_cols = [col for col in df_processed.columns if 'bmi' in col or 'weight' in col]
        for col in critical_cols:
            assert df_processed[col].notna().all(), f"NaN found in {col}"
    
    def test_fit_transform_equivalence(self, preprocessor, sample_data):
        """Test that fit_transform equals fit().transform()"""
        df1 = preprocessor.fit_transform(sample_data.copy())
        
        preprocessor2 = DataPreprocessor()
        df2 = preprocessor2.fit(sample_data.copy()).transform(sample_data.copy())
        
        # Should produce same results
        assert len(df1) == len(df2)
        assert set(df1.columns) == set(df2.columns)
    
    def test_edge_cases(self, preprocessor):
        """Test edge cases and error handling"""
        # Empty dataframe
        empty_df = pd.DataFrame()
        result = preprocessor.transform(empty_df)
        assert len(result) == 0
        
        # Missing columns
        minimal_df = pd.DataFrame({'patient_id': [1, 2, 3]})
        result = preprocessor.transform(minimal_df)
        assert len(result) > 0  # Should not crash
    
    def test_get_feature_names(self, preprocessor, sample_data):
        """Test feature name extraction"""
        df_processed = preprocessor.fit_transform(sample_data)
        feature_names = preprocessor.get_feature_names(df_processed)
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])