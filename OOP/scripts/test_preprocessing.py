import pandas as pd
import sys
import os

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from OOP.services.preprocessor import DataPreprocessor


def test_with_real_csv(input_csv_path, output_csv_path):
    """
    Test preprocessing with real CSV data
    
    Args:
        input_csv_path: Path to your raw CSV file
        output_csv_path: Path where processed CSV will be saved
    """
    print("\n" + "="*80)
    print("TESTING WITH REAL DATA")
    print("="*80)
    
    # Load raw data
    print(f"\nLoading data from: {input_csv_path}")
    try:
        df_raw = pd.read_csv(input_csv_path)
        print(f"Successfully loaded {len(df_raw)} patients")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return
    
    # Display original data info
    print("\n" + "-"*80)
    print("ORIGINAL DATA SUMMARY")
    print("-"*80)
    print(f"Shape: {df_raw.shape}")
    print(f"Columns ({len(df_raw.columns)}): {df_raw.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df_raw.head().to_string())
    print("\nMissing values per column:")
    missing = df_raw.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
    
    # Keep patient ID separate if it exists
    patient_id_col = 'practice_patient_id'
    patient_ids = None
    
    if patient_id_col in df_raw.columns:
        patient_ids = df_raw[patient_id_col].copy()
        print(f"\nPreserved {len(patient_ids)} patient IDs for comparison")
    else:
        print(f"\nWarning: '{patient_id_col}' column not found")
        print(f"Available columns: {df_raw.columns.tolist()}")
    
    # Preprocess data
    print("\n" + "-"*80)
    print("RUNNING PREPROCESSING PIPELINE...")
    print("-"*80)
    
    preprocessor = DataPreprocessor()
    
    try:
        # Create a copy with index tracking
        df_with_index = df_raw.copy()
        df_with_index['_original_index'] = range(len(df_with_index))
        
        # Run preprocessing
        df_processed = preprocessor.fit_transform(df_with_index)
        
        # Map patient IDs back using original indices
        if patient_ids is not None and '_original_index' in df_processed.columns:
            original_indices = df_processed['_original_index'].astype(int).values
            df_processed.insert(0, patient_id_col, patient_ids.iloc[original_indices].values)
            df_processed = df_processed.drop('_original_index', axis=1)
        
        print(f"Preprocessing completed successfully!")
        print(f"  Input: {len(df_raw)} patients")
        print(f"  Output: {len(df_processed)} patients")
        print(f"  Dropped: {len(df_raw) - len(df_processed)} patients ({(len(df_raw) - len(df_processed))/len(df_raw)*100:.2f}%)")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Display processed data info
    print("\n" + "-"*80)
    print("PROCESSED DATA SUMMARY")
    print("-"*80)
    print(f"Shape: {df_processed.shape}")
    print(f"Columns ({len(df_processed.columns)}): {df_processed.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df_processed.head().to_string())
    print("\nMissing values per column:")
    missing_processed = df_processed.isnull().sum()
    print(missing_processed[missing_processed > 0] if missing_processed.sum() > 0 else "No missing values")
    
    # Show changes
    print("\n" + "-"*80)
    print("PREPROCESSING CHANGES")
    print("-"*80)
    
    print("\nColumns removed:")
    removed = set(df_raw.columns) - set(df_processed.columns)
    if removed:
        for col in sorted(removed):
            print(f"  - {col}")
    else:
        print("  None")
    
    print("\nColumns added:")
    added = set(df_processed.columns) - set(df_raw.columns) - {patient_id_col}
    if added:
        for col in sorted(added):
            print(f"  + {col}")
    else:
        print("  None")
    
    # Save processed data
    print("\n" + "-"*80)
    print("SAVING PROCESSED DATA")
    print("-"*80)
    
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        df_processed.to_csv(output_csv_path, index=False)
        print(f"Processed data saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")
    
    # Create comparison file
    if patient_ids is not None:
        comparison_path = output_csv_path.replace('.csv', '_comparison.csv')
        try:
            df_comparison = df_raw.copy()
            df_comparison['preprocessing_status'] = 'Dropped'
            
            if patient_id_col in df_processed.columns:
                kept_ids = df_processed[patient_id_col].values
                df_comparison.loc[df_comparison[patient_id_col].isin(kept_ids), 'preprocessing_status'] = 'Kept'
            
            df_comparison.to_csv(comparison_path, index=False)
            print(f"Comparison file saved to: {comparison_path}")
            print(f"  (Original data with 'Kept' or 'Dropped' status)")
        except Exception as e:
            print(f"Warning: Could not create comparison file: {str(e)}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(df_processed.shape)
    return df_processed


if __name__ == "__main__":
    
    # ========================================================================
    # MANUALLY SET YOUR FILE PATHS HERE
    # ========================================================================
    
    INPUT_CSV = "combined_patients.csv"  # CHANGE THIS to your raw CSV path
    OUTPUT_CSV = "csv/test_pre_processing_data.csv"  # CHANGE THIS to desired output path
    
    # ========================================================================
    
    print("\n" + "="*80)
    print("BMI TRAJECTORY PREPROCESSING - REAL DATA TEST")
    print("="*80)
    print(f"\nInput file: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_CSV}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"\nError: File not found at {INPUT_CSV}")
        print("\nPlease update the INPUT_CSV path in the script.")
        exit(1)
    
    # Run test
    test_with_real_csv(INPUT_CSV, OUTPUT_CSV)