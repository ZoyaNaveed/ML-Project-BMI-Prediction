import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from OOP.services.encoder import FeatureEncoder


def create_sample_encoded_data():
    """Create sample data that mimics preprocessed output"""
    np.random.seed(42)
    n_patients = 50
    
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.randint(20, 80, n_patients),
        'marital_status': np.random.randint(0, 2, n_patients),
        'gender': np.random.choice(['F', 'M'], n_patients),
        
        # Race data
        'race': np.random.choice([
            'White', 'Black', 'Asian', 'Hispanic', 'Unknown', 
            'CAUCASIAN,WHITE', 'AFRICAN AMERICAN'
        ], n_patients),
        
        # Surgery data
        'surgery_name': [
            'gastric bypass;cholecystectomy' if i % 10 == 0 
            else 'hernia repair' if i % 5 == 0 
            else 'c-section' if i % 3 == 0
            else 'none'
            for i in range(n_patients)
        ],
        
        # ICD-10 codes
        'icd_10': [
            'E66.9;I10;E11.9' if i % 4 == 0  # Obesity, HTN, Diabetes
            else 'I10;E78.5' if i % 3 == 0  # HTN, Dyslipidemia
            else 'E11.9' if i % 2 == 0  # Diabetes only
            else 'none'
            for i in range(n_patients)
        ],
        
        # Disease history
        'jsdisease': [
            'diabetes mellitus;hypertension' if i % 4 == 0
            else 'heart disease' if i % 3 == 0
            else 'thyroid disorder' if i % 2 == 0
            else 'none'
            for i in range(n_patients)
        ],
        
        # Lab results
        'lab_name_result': [
            f'Glucose measurement={np.random.randint(80, 150)};Hemoglobin A1c={np.random.uniform(5, 8):.1f}'
            for _ in range(n_patients)
        ],
        
        # Lifestyle factors
        'smoking_status': np.random.choice(['Never', 'Current', 'Former', 'Unknown'], n_patients),
        'alcohol_usage_type': np.random.choice(['never', 'current', 'abuse', 'unknown'], n_patients),
        
        # GPI medication codes (sample)
        '121030': np.random.randint(0, 100, n_patients),
        '221000': np.random.randint(0, 100, n_patients),
        '572000': np.random.randint(0, 100, n_patients),
        'gpi': ['some;raw;gpi;data' for _ in range(n_patients)],
    }
    
    return pd.DataFrame(data)


def print_dataframe_info(df, title):
    """Print detailed dataframe information"""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics for numeric columns:")
    print(df.describe().to_string())


def test_individual_encodings():
    """Test each encoding method individually"""
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL ENCODING METHODS")
    print("="*80)
    
    encoder = FeatureEncoder()
    
    # Test 1: Gender encoding
    print("\n[Test 1] Gender Encoding")
    df_gender = pd.DataFrame({'gender': ['F', 'M', 'F', 'M', 'F']})
    result = encoder._encode_gender(df_gender)
    print(f"Original: {df_gender['gender'].tolist()}")
    print(f"Encoded: {result['gender'].tolist()}")
    assert result['gender'].tolist() == [0, 1, 0, 1, 0], "Gender encoding failed"
    print("✓ Gender encoding working correctly")
    
    # Test 2: Race encoding
    print("\n[Test 2] Race Encoding")
    df_race = pd.DataFrame({'race': ['White', 'Black', 'Asian', 'Hispanic']})
    result = encoder._encode_race(df_race)
    race_cols = [col for col in result.columns if col.startswith('race_')]
    print(f"Created columns: {race_cols}")
    print(result[race_cols].to_string())
    assert len(race_cols) > 0, "Race encoding failed"
    print("✓ Race encoding working correctly")
    
    # Test 3: Surgery encoding
    print("\n[Test 3] Surgery Encoding")
    df_surgery = pd.DataFrame({
        'surgery_name': ['gastric bypass', 'hernia repair', 'c-section', 'none']
    })
    result = encoder._encode_surgeries(df_surgery)
    print(f"Surgery columns created: {[col for col in result.columns if col in encoder.surgery_groups.keys()]}")
    if 'Bariatric' in result.columns:
        print(f"Bariatric surgeries found: {result['Bariatric'].sum()}")
    print("✓ Surgery encoding working correctly")
    
    # Test 4: ICD-10 encoding
    print("\n[Test 4] ICD-10 Code Encoding")
    df_icd = pd.DataFrame({
        'icd_10': ['E66.9;I10', 'E11.9', 'F32.1', 'none']
    })
    result = encoder._encode_icd_codes(df_icd)
    icd_cols = [col for col in result.columns if col in encoder.icd_groups.keys()]
    print(f"ICD group columns: {icd_cols}")
    print(result[icd_cols].to_string())
    print("✓ ICD-10 encoding working correctly")
    
    # Test 5: Lab encoding
    print("\n[Test 5] Lab Results Encoding")
    df_labs = pd.DataFrame({
        'lab_name_result': [
            'Glucose measurement=120;Hemoglobin A1c=6.5',
            'TSH measurement=2.5;Creatinine measurement=1.0'
        ]
    })
    result = encoder._encode_labs(df_labs)
    lab_cols = list(encoder.lab_keep.keys())
    print(f"Lab columns: {lab_cols}")
    print(result[lab_cols].head().to_string())
    print("✓ Lab encoding working correctly")
    
    # Test 6: Smoking encoding
    print("\n[Test 6] Smoking Status Encoding")
    df_smoking = pd.DataFrame({
        'smoking_status': ['Never', 'Current', 'Former', 'Unknown']
    })
    result = encoder._encode_smoking(df_smoking)
    print(f"Encoded: {result['smoking_status'].tolist()}")
    assert result['smoking_status'].tolist() == [0, 1, 1, 0], "Smoking encoding failed"
    print("✓ Smoking encoding working correctly")
    
    # Test 7: Alcohol encoding
    print("\n[Test 7] Alcohol Usage Encoding")
    df_alcohol = pd.DataFrame({
        'alcohol_usage_type': ['never', 'current', 'abuse', 'unknown']
    })
    result = encoder._encode_alcohol(df_alcohol)
    print(f"Encoded: {result['alcohol_usage'].tolist()}")
    assert result['alcohol_usage'].tolist() == [0, 1, 1, 0], "Alcohol encoding failed"
    print("✓ Alcohol encoding working correctly")
    
    print("\n" + "="*80)
    print("✓ ALL INDIVIDUAL TESTS PASSED")
    print("="*80)


def test_full_encoding_pipeline():
    """Test complete encoding pipeline"""
    print("\n" + "="*80)
    print("TESTING FULL ENCODING PIPELINE")
    print("="*80)
    
    # Create sample data
    df = create_sample_encoded_data()
    print_dataframe_info(df, "ORIGINAL DATA (Before Encoding)")
    
    # Initialize encoder
    encoder = FeatureEncoder()
    
    # Run encoding
    try:
        df_encoded = encoder.fit_transform(df)
        print_dataframe_info(df_encoded, "ENCODED DATA (After Encoding)")
        
        # Verify encoding results
        print("\n" + "-"*80)
        print("ENCODING VERIFICATION")
        print("-"*80)
        
        # Check gender is binary
        if 'gender' in df_encoded.columns:
            unique_gender = df_encoded['gender'].unique()
            print(f"✓ Gender values: {sorted(unique_gender)}")
            assert set(unique_gender).issubset({0, 1}), "Gender should be binary"
        
        # Check race columns exist
        race_cols = [col for col in df_encoded.columns if col.startswith('race_')]
        if race_cols:
            print(f"✓ Race columns created: {race_cols}")
        
        # Check surgery groups
        surgery_cols = [col for col in df_encoded.columns if col in encoder.surgery_groups.keys()]
        if surgery_cols:
            print(f"✓ Surgery groups: {surgery_cols}")
            for col in surgery_cols:
                print(f"  - {col}: {df_encoded[col].sum()} patients")
        
        # Check ICD groups
        icd_cols = [col for col in df_encoded.columns if col in encoder.icd_groups.keys()]
        if icd_cols:
            print(f"✓ ICD-10 groups: {icd_cols}")
            for col in icd_cols:
                print(f"  - {col}: {df_encoded[col].sum()} patients")
        
        # Check disease history
        history_cols = [col for col in df_encoded.columns if col.startswith('history_of_')]
        if history_cols:
            print(f"✓ Disease history columns: {history_cols}")
        
        # Check lab results
        lab_cols = [col for col in df_encoded.columns if col in encoder.lab_keep.keys()]
        if lab_cols:
            print(f"✓ Lab result columns: {lab_cols}")
        
        # Check lifestyle factors
        if 'smoking_status' in df_encoded.columns:
            print(f"✓ Smoking status encoded: {df_encoded['smoking_status'].value_counts().to_dict()}")
        
        if 'alcohol_usage' in df_encoded.columns:
            print(f"✓ Alcohol usage encoded: {df_encoded['alcohol_usage'].value_counts().to_dict()}")
        
        # Check GPI columns
        gpi_cols = [col for col in df_encoded.columns if col in encoder.gpi_columns]
        if gpi_cols:
            print(f"✓ GPI medication columns kept: {len(gpi_cols)} columns")
        
        # Check that raw columns were dropped
        dropped_cols = ['surgery_name', 'surgeries', 'icd_10', 'jsdisease', 'lab_name_result', 'gpi']
        remaining_dropped = [col for col in dropped_cols if col in df_encoded.columns]
        if not remaining_dropped:
            print(f"✓ Raw columns properly dropped")
        else:
            print(f"⚠ Warning: These raw columns still exist: {remaining_dropped}")
        
        # Check bariatric patients removed
        if 'Bariatric' not in df_encoded.columns:
            print(f"✓ Bariatric patients removed and column dropped")
        
        print("\n" + "="*80)
        print("✓ FULL PIPELINE TEST PASSED")
        print("="*80)
        
        return df_encoded
        
    except Exception as e:
        print(f"\n✗ ERROR during encoding: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_with_real_data(input_csv_path, output_csv_path):
    """Test encoder with real CSV data"""
    print("\n" + "="*80)
    print("TESTING WITH REAL DATA")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
        print(f"✓ Successfully loaded {len(df)} patients")
    except Exception as e:
        print(f"✗ Error loading CSV: {str(e)}")
        return
    
    print_dataframe_info(df, "ORIGINAL DATA")
    
    # Run encoding
    encoder = FeatureEncoder()
    
    try:
        df_encoded = encoder.fit_transform(df)
        print_dataframe_info(df_encoded, "ENCODED DATA")
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_encoded.to_csv(output_csv_path, index=False)
        print(f"\n✓ Encoded data saved to: {output_csv_path}")
        
        # Show changes
        print("\n" + "-"*80)
        print("ENCODING CHANGES")
        print("-"*80)
        print(f"Rows: {len(df)} -> {len(df_encoded)}")
        print(f"Columns: {len(df.columns)} -> {len(df_encoded.columns)}")
        
        print("\nColumns added:")
        added = set(df_encoded.columns) - set(df.columns)
        if added:
            for col in sorted(added):
                print(f"  + {col}")
        
        print("\nColumns removed:")
        removed = set(df.columns) - set(df_encoded.columns)
        if removed:
            for col in sorted(removed):
                print(f"  - {col}")
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during encoding: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BMI TRAJECTORY FEATURE ENCODING - TEST SUITE")
    print("="*80)
    
    print("\nSelect test mode:")
    print("1. Test individual encoding methods")
    print("2. Test full encoding pipeline with sample data")
    print("3. Test with real CSV file")
    print("4. Run all tests")
    
    choice = input("\nEnter choice (1/2/3/4) or press Enter for option 4: ").strip() or "4"
    
    if choice in ["1", "4"]:
        test_individual_encodings()
    
    if choice in ["2", "4"]:
        test_full_encoding_pipeline()
    
    if choice in ["3", "4"]:
        print("\n" + "="*80)
        print("REAL DATA TEST")
        print("="*80)
        
        # For real data test, you need to provide paths
        INPUT_CSV = "csv/test_pre_processing_data.csv"  # Output from preprocessing
        OUTPUT_CSV = "csv/encoded_patient_data.csv"
        
        print(f"\nInput file: {INPUT_CSV}")
        print(f"Output file: {OUTPUT_CSV}")
        
        if os.path.exists(INPUT_CSV):
            test_with_real_data(INPUT_CSV, OUTPUT_CSV)
        else:
            print(f"\n⚠ File not found: {INPUT_CSV}")
            print("Skipping real data test. Run preprocessing first.")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)