import pandas as pd
import sys
import os

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)



from OOP.services.preprocessor import DataPreprocessor
from OOP.services.encoder import FeatureEncoder


def run_full_pipeline(input_csv, output_csv):
    """
    Run preprocessing and encoding pipelines on raw data
    
    Args:
        input_csv: Path to raw CSV file
        output_csv: Path to save final processed CSV
    """
    print("="*80)
    print("BMI TRAJECTORY - FULL DATA PIPELINE")
    print("="*80)
    
    # 1. Load raw data
    print(f"\n[1/4] Loading raw data from: {input_csv}")
    try:
        df_raw = pd.read_csv(input_csv)
        print(f"✓ Loaded {len(df_raw)} patients with {len(df_raw.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return
    
    # 2. Run preprocessing pipeline
    print(f"\n[2/4] Running preprocessing pipeline...")
    preprocessor = DataPreprocessor()
    try:
        df_preprocessed = preprocessor.fit_transform(df_raw)
        print(f"✓ Preprocessing complete: {len(df_preprocessed)} patients, {len(df_preprocessed.columns)} columns")
        print(f"  Dropped {len(df_raw) - len(df_preprocessed)} patients during preprocessing")
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Run encoding pipeline
    print(f"\n[3/4] Running encoding pipeline...")
    encoder = FeatureEncoder()
    try:
        df_final = encoder.fit_transform(df_preprocessed)
        print(f"✓ Encoding complete: {len(df_final)} patients, {len(df_final.columns)} columns")
        print(f"  Dropped {len(df_preprocessed) - len(df_final)} patients during encoding")
    except Exception as e:
        print(f"✗ Error in encoding: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Save final data
    print(f"\n[4/4] Saving final data to: {output_csv}")
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_final.to_csv(output_csv, index=False)
        print(f"✓ Saved successfully!")
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Input:  {len(df_raw)} patients, {len(df_raw.columns)} columns")
    print(f"Output: {len(df_final)} patients, {len(df_final.columns)} columns")
    print(f"Total dropped: {len(df_raw) - len(df_final)} patients ({(len(df_raw) - len(df_final))/len(df_raw)*100:.2f}%)")
    
    print("\nFinal columns:")
    for i, col in enumerate(df_final.columns, 1):
        print(f"  {i}. {col}")
    
    print("\nFirst 3 rows of final data:")
    print(df_final.head(3).to_string())
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Final data saved to: {output_csv}")


if __name__ == "__main__":
    # ========================================================================
    # SET YOUR FILE PATHS HERE
    # ========================================================================
    
    INPUT_FILE = "combined_patients.csv"        # Your raw CSV
    OUTPUT_FILE = "csv/final_data.csv"   # Where to save final result
    
    # ========================================================================
    
    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Check if input exists
    if not os.path.exists(INPUT_FILE):
        print(f"\n✗ Error: Input file not found at {INPUT_FILE}")
        print("\nPlease update INPUT_FILE path in the script.")
        exit(1)
    
    # Run the full pipeline
    run_full_pipeline(INPUT_FILE, OUTPUT_FILE)