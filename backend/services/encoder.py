import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Handles all encoding transformations for patient features.
    Encodes surgeries, diagnoses, labs, medications, and lifestyle factors.
    """
    
    def __init__(self):
        """Initialize encoder with all encoding configurations"""
        
        # Surgery groups
        self.surgery_groups = {
            "Bariatric": [
                "gastric bypass", "gastric sleeve", "bariatric", "lap band", "lapband",
                "sleeve gastrectomy", "duodenal switch", "gastric banding", "roux-en-y",
                "vertical sleeve", "stomach stapled", "bariatric gastric sleeve"
            ],
            "Gallbladder": [
                "gallbladder", "cholecystectomy", "gall bladder", "gallstone"
            ],
            "Thyroid": [
                "thyroidectomy", "thyroid surgery", "thyroid removal", "thyroid",
                "parathyroidectomy", "parathyroid"
            ],
            "Reproductive": [
                "c-section", "cesarean", "tubal ligation", "hysterectomy", "oophorectomy",
                "csection", "c section", "salpingectomy", "btl", "bilateral tubal",
                "tubal", "ovarian", "uterine", "fallopian"
            ],
            "Gastrointestinal": [
                "partial gastrectomy", "gastrectomy", "colon resection", "colectomy",
                "bowel resection", "hemicolectomy", "sigmoidectomy", "colostomy",
                "ileostomy", "whipple", "pancreatic", "esophageal", "stomach surgery",
                "intestinal", "rectal", "bowel obstruction"
            ],
            "Nissen_Fundoplication": [
                "nissen fundoplication", "fundoplication"
            ],
            "Kidney": [
                "nephrectomy", "kidney removal", "kidney surgery", "renal transplant",
                "kidney transplant"
            ],
            "Cardiac": [
                "cabg", "coronary artery bypass", "bypass", "cardiac bypass",
                "heart surgery", "open heart", "aortic valve", "mitral valve",
                "heart stent", "cardiac stent", "coronary stent", "angioplasty",
                "pacemaker", "defibrillator", "aicd"
            ],
            "Hernia": [
                "hernia", "herniorrhaphy", "umbilical hernia", "inguinal hernia",
                "hiatal hernia", "ventral hernia"
            ],
        }
        
        # ICD-10 code groups
        self.icd_groups = {
            "Obesity": ["E66"],
            "Diabetes": ["E11", "E08", "E10", "E13"],
            "Dyslipidemia": ["E78"],
            "Hypertension": ["I10", "I11", "I12", "I13", "I16"],
            "Hypothyroid": ["E03", "E89", "E05"],
            "PCOS": ["E28"],
            "Depression": ["F32", "F33"],
            "Schizophrenia": ["F20", "F29", "F34", "F39", "F43"],
            "Bipolar": ["F30", "F31"],
            "SleepApnea": ["G47", "R06"],
        }
        
        # Build reverse map for ICD codes
        self.prefix_to_groups = {}
        for grp, codes in self.icd_groups.items():
            for code in codes:
                self.prefix_to_groups.setdefault(code.upper(), set()).add(grp)
        
        # Disease history groups
        self.disease_groups = {
            "Diabetes_and_Obesity": ["diabetes", "dm", "diabetes mellitus", "obesity"],
            "Hypertension": ["hypertension", "high blood pressure", "htn"],
            "HeartDisease_or_Stroke": [
                "heart disease", "coronary artery", "cad", "mi", 
                "myocardial infarction", "stroke", "cva", "cerebrovascular accident"
            ],
            "Cancer": ["cancer", "carcinoma", "tumor", "malignancy"],
            "KidneyDisease": ["renal", "kidney disease", "ckd"],
            "Thyroid": ["thyroid"],
        }
        
        # Lab tests to keep
        self.lab_keep = {
            "Glucose": ["Glucose measurement", "Glucose Level", "Fasting blood sugar", 
                       "Random blood glucose", "Serum glucose"],
            "HbA1c": ["Estimated average blood glucose determination based on hemoglobin A1c", 
                     "Hemoglobin A1c", "Glycated hemoglobin"],
            "LDL": ["Low density lipoprotein (LDL) cholesterol measurement"],
            "HDL": ["High density lipoprotein (HDL) cholesterol measurement"],
            "Triglycerides": ["Very low density lipoprotein (VLDL) cholesterol measurement"],
            "TotalChol": ["Serum total cholesterol to high density lipoprotein cholesterol ratio"],
            "TSH": ["Thyroid stimulating hormone (TSH) measurement"],
            "Creatinine": ["Serum or plasma creatinine measurement (mass/volume)"],
        }
        
        # GPI medication codes
        self.gpi_columns = [
            '121030', '121099', '214040', '221000', '251500', '271040', '272000',
            '272800', '276070', '331000', '332000', '412000', '415000', '572000',
            '580300', '581600', '582000', '590700', '591000', '591520', '591530',
            '591570', '592000', '592500', '595000', '725000', '726000'
        ]
        
        # Race standardization mapping
        self.race_mapping = {
            'DECLINED': 'Unknown', 'UNCERTAIN': 'Unknown', 'PROHIBITED': 'Unknown', 'UNK': 'Unknown',
            'BLACK': 'Black', 'AFRICAN AMERICAN': 'Black', 'HAITIAN': 'Black', 'JAMAICAN': 'Black',
            'WHITE': 'White', 'CAUCASIAN': 'White', 'POLISH': 'White', 'ENGLISH': 'White',
            'ASIAN': 'Asian', 'CHINESE': 'Asian', 'VIETNAMESE': 'Asian', 'INDIAN': 'Asian',
            'HISPANIC': 'Hispanic_or_Mixed', 'MEXICAN': 'Hispanic_or_Mixed', 'MIXED': 'Hispanic_or_Mixed',
        }
        
        self.columns_to_drop = []
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEncoder':
        """Fit the encoder (placeholder for future stateful operations)"""
        logger.info("Fitting encoder...")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all encoding transformations"""
        df = df.copy()
        
        logger.info(f"Starting encoding with {len(df)} patients")
        
        # 1. Encode gender
        df = self._encode_gender(df)
        
        # 2. Standardize and encode race
        df = self._encode_race(df)
        
        # 3. Encode surgeries
        df = self._encode_surgeries(df)
        
        # 4. Remove bariatric surgery patients
        df = self._remove_bariatric_patients(df)
        
        # 5. Encode ICD-10 codes
        df = self._encode_icd_codes(df)
        
        # 6. Encode disease history
        df = self._encode_disease_history(df)
        
        # 7. Encode lab results
        df = self._encode_labs(df)
        
        # 8. Encode smoking status
        df = self._encode_smoking(df)
        
        # 9. Encode alcohol usage
        df = self._encode_alcohol(df)
        
        # 10. Keep only specified GPI columns (medications)
        df = self._create_gpi_columns(df)
        
        logger.info(f"Encoding complete. Final shape: {df.shape}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    # ========================================================================
    # GENDER ENCODING
    # ========================================================================
    
    def _encode_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode gender as binary (F=0, M=1)"""
        if 'gender' not in df.columns:
            logger.warning("'gender' column not found")
            return df
        
        logger.info("Encoding gender...")
        
        gender_encoded = df['gender'].map({'F': 0, 'M': 1})
        df = df.drop('gender', axis=1)
        df['gender'] = gender_encoded
        
        logger.info(f"Gender encoding: Female (0): {(df['gender']==0).sum()}, Male (1): {(df['gender']==1).sum()}")
        
        return df
    
    # ========================================================================
    # RACE ENCODING
    # ========================================================================
    
    def _standardize_race(self, race_str):
        """Standardize race categories"""
        if pd.isna(race_str):
            return 'Unknown'
        
        race_str = str(race_str).upper().strip()
        
        if any(term in race_str for term in ['DECLINED', 'UNCERTAIN', 'PROHIBITED', 'UNK']):
            return 'Unknown'
        elif any(term in race_str for term in ['BLACK', 'AFRICAN AMERICAN', 'HAITIAN', 'JAMAICAN']):
            return 'Black'
        elif any(term in race_str for term in ['WHITE', 'CAUCASIAN', 'POLISH', 'ENGLISH', 'FRENCH', 'IRISH', 'ITALIAN']):
            return 'White'
        elif any(term in race_str for term in ['ASIAN', 'CHINESE', 'VIETNAMESE', 'KOREAN', 'FILIPINO', 'INDIAN']):
            return 'Asian'
        else:
            return 'Hispanic_or_Mixed'
    
    def _encode_race(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and one-hot encode race"""
        if 'race' not in df.columns:
            logger.warning("'race' column not found")
            return df
        
        logger.info("Standardizing and encoding race...")
        
        df['race_standardized'] = df['race'].apply(self._standardize_race)
        race_dummies = pd.get_dummies(df['race_standardized'], prefix='race')
        df = pd.concat([df, race_dummies], axis=1)
        df = df.drop(['race', 'race_standardized'], axis=1)
        
        logger.info(f"Created race columns: {[col for col in df.columns if col.startswith('race_')]}")
        
        return df
    
    # ========================================================================
    # SURGERY ENCODING
    # ========================================================================
    
    def _parse_surgeries(self, x):
        """Parse semicolon-separated surgery strings"""
        if pd.isna(x):
            return []
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "no surgeries"}:
            return []
        return [p.strip() for p in s.split(';') if p.strip()]
    
    def _extract_surgery_groups(self, surg_list):
        """Extract surgery groups from list of surgeries"""
        values = [str(v).strip().lower() for v in surg_list]
        results = {}
        
        for group, keywords in self.surgery_groups.items():
            results[group] = int(any(any(k in v for k in keywords) for v in values))
        
        return results
    
    def _encode_surgeries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode surgery types into binary groups"""
        surgery_col = None
        for col in ['surgery_name', 'surgeries']:
            if col in df.columns:
                surgery_col = col
                break
        
        if not surgery_col:
            logger.warning("No surgery column found")
            return df
        
        logger.info(f"Encoding surgeries from '{surgery_col}'...")
        
        # Rename to standard name
        if surgery_col != 'surgeries':
            df = df.rename(columns={surgery_col: 'surgeries'})
        
        df["surgery_list"] = df["surgeries"].apply(self._parse_surgeries)
        surg_df = df["surgery_list"].apply(self._extract_surgery_groups).apply(pd.Series)
        df = pd.concat([df, surg_df], axis=1)
        
        # Log distribution
        for group in self.surgery_groups.keys():
            count = df[group].sum()
            logger.info(f"  {group}: {count} patients ({count/len(df)*100:.2f}%)")
        
        return df
    
    def _remove_bariatric_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove patients who had bariatric surgery"""
        if 'Bariatric' not in df.columns:
            logger.warning("'Bariatric' column not found, skipping removal")
            return df
        
        initial_count = len(df)
        bariatric_count = df['Bariatric'].sum()
        
        df = df[df['Bariatric'] == 0]
        
        logger.info(f"Removed {bariatric_count} patients with bariatric surgery ({bariatric_count/initial_count*100:.2f}%)")
        
        # Drop surgery-related columns
        df = df.drop(['Bariatric', 'surgeries', 'surgery_list'], axis=1, errors='ignore')
        
        return df
    
    # ========================================================================
    # ICD-10 CODE ENCODING
    # ========================================================================
    
    def _parse_icd_list(self, x):
        """Parse semicolon-separated ICD codes"""
        if pd.isna(x):
            return []
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none"}:
            return []
        return [p.strip() for p in s.split(';') if p.strip()]
    
    def _extract_icd_prefixes(self, icd_list):
        """Extract first 3 characters from ICD codes"""
        prefixes = []
        for code in icd_list:
            sc = str(code).upper().strip().replace('.', '')
            sc = re.sub(r'[^A-Z0-9]', '', sc)
            if len(sc) >= 3:
                prefixes.append(sc[:3])
        return prefixes
    
    def _map_icd_to_groups(self, prefixes):
        """Map ICD prefixes to disease groups"""
        if not prefixes:
            return {g: 0 for g in self.icd_groups.keys()}
        
        present = set()
        for p in set(prefixes):
            if p in self.prefix_to_groups:
                present |= self.prefix_to_groups[p]
        
        return {g: int(g in present) for g in self.icd_groups.keys()}
    
    def _encode_icd_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode ICD-10 diagnosis codes into disease groups"""
        if 'icd_10' not in df.columns:
            logger.warning("'icd_10' column not found")
            return df
        
        logger.info("Encoding ICD-10 codes...")
        
        icd_lists = df["icd_10"].apply(self._parse_icd_list)
        df["icd_prefixes"] = icd_lists.apply(self._extract_icd_prefixes)
        group_df = df["icd_prefixes"].apply(self._map_icd_to_groups).apply(pd.Series).fillna(0).astype(int)
        df = pd.concat([df, group_df], axis=1)
        df = df.drop(['icd_10', 'icd_prefixes'], axis=1)
        
        # Log distribution
        for group in self.icd_groups.keys():
            count = df[group].sum()
            logger.info(f"  {group}: {count} patients ({count/len(df)*100:.2f}%)")
        
        return df
    
    # ========================================================================
    # DISEASE HISTORY ENCODING
    # ========================================================================
    
    def _parse_diseases(self, x):
        """Parse semicolon-separated disease strings"""
        if pd.isna(x):
            return []
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none"}:
            return []
        return [p.strip() for p in s.split(';') if p.strip()]
    
    def _normalize_text(self, s):
        """Normalize text for matching"""
        s = str(s).lower().strip()
        return re.sub(r"[^a-z0-9\s]", "", s)
    
    def _extract_disease_groups(self, disease_list):
        """Extract disease history groups"""
        values = [self._normalize_text(v) for v in disease_list]
        results = {}
        
        for group, keywords in self.disease_groups.items():
            col_name = f"history_of_{group}"
            results[col_name] = int(any(any(k in v for k in keywords) for v in values))
        
        return results
    
    def _encode_disease_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode disease history from jsdisease column"""
        if 'jsdisease' not in df.columns:
            logger.warning("'jsdisease' column not found")
            return df
        
        logger.info("Encoding disease history...")
        
        df["disease_list"] = df["jsdisease"].apply(self._parse_diseases)
        disease_df = df["disease_list"].apply(self._extract_disease_groups).apply(pd.Series).fillna(0).astype(int)
        df = pd.concat([df, disease_df], axis=1)
        df = df.drop(['jsdisease', 'disease_list'], axis=1)
        
        # Log distribution
        for col in [c for c in df.columns if c.startswith('history_of_')]:
            count = df[col].sum()
            logger.info(f"  {col}: {count} patients ({count/len(df)*100:.2f}%)")
        
        return df
    
    # ========================================================================
    # LAB RESULTS ENCODING
    # ========================================================================
    
    def _parse_labs(self, x):
        """Parse semicolon-separated lab key=value pairs"""
        if pd.isna(x):
            return {}
        
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none"}:
            return {}
        
        lab_dict = {}
        pairs = [p.strip() for p in s.split(';') if p.strip()]
        
        for pair in pairs:
            if '=' in pair:
                parts = pair.split('=', 1)
                key = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ''
                
                if key in lab_dict:
                    try:
                        old_val = float(lab_dict[key])
                        new_val = float(value)
                        lab_dict[key] = (old_val + new_val) / 2
                    except:
                        lab_dict[key] = value
                else:
                    lab_dict[key] = value
        
        return lab_dict
    
    def _normalize_lab_value(self, val):
        """Normalize lab values to float"""
        if pd.isna(val):
            return 0.0
        s = str(val).strip().lower()
        
        if s in {"negative", "neg"}:
            return 0.0
        if s in {"positive", "pos"}:
            return 1.0
        
        try:
            return float(s)
        except:
            return 0.0
    
    def _extract_lab_values(self, lab_dict):
        """Extract specific lab values"""
        result = {}
        
        for lab_name, lab_aliases in self.lab_keep.items():
            found_value = None
            
            for alias in lab_aliases:
                if alias in lab_dict:
                    found_value = lab_dict[alias]
                    break
            
            result[lab_name] = self._normalize_lab_value(found_value) if found_value is not None else 0.0
        
        return result
    
    def _encode_labs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode laboratory test results"""
        # Check for different possible column names
        lab_col = None
        for col in ['lab_name_result', 'labs']:
            if col in df.columns:
                lab_col = col
                break
        
        if not lab_col:
            logger.warning("No lab column found")
            return df
        
        logger.info(f"Encoding lab results from '{lab_col}'...")
        
        # Rename to standard name
        if lab_col != 'labs':
            df = df.rename(columns={lab_col: 'labs'})
        
        df["labs_dict"] = df["labs"].apply(self._parse_labs)
        labs_df = df["labs_dict"].apply(self._extract_lab_values).apply(pd.Series)
        df = pd.concat([df, labs_df], axis=1)
        df = df.drop(['labs', 'labs_dict'], axis=1)
        
        logger.info(f"Created lab columns: {list(self.lab_keep.keys())}")
        
        return df
    
    # ========================================================================
    # LIFESTYLE FACTORS ENCODING
    # ========================================================================
    
    def _encode_smoking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode smoking status (0=non-smoker, 1=current smoker)"""
        if 'smoking_status' not in df.columns:
            logger.warning("'smoking_status' column not found")
            return df
        
        logger.info("Encoding smoking status...")
        
        df['smoking_status_clean'] = df['smoking_status'].fillna('nan').str.strip().str.lower()
        
        smoking_map = {
            'never': 0,
            'former': 1,
            'unknown': 0,
            'nan': 0,
            'current': 1
        }
        
        df['smoking_status_encoded'] = df['smoking_status_clean'].map(smoking_map).fillna(0).astype(int)
        df = df.drop(['smoking_status', 'smoking_status_clean'], axis=1)
        df = df.rename(columns={'smoking_status_encoded': 'smoking_status'})
        
        smokers = df['smoking_status'].sum()
        logger.info(f"Smoking: Current/Former (1): {smokers} ({smokers/len(df)*100:.2f}%)")
        
        return df
    
    def _encode_alcohol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode alcohol usage (0=non-user, 1=user/abuse)"""
        if 'alcohol_usage_type' not in df.columns:
            logger.warning("'alcohol_usage_type' column not found")
            return df
        
        logger.info("Encoding alcohol usage...")
        
        df['alcohol_usage_clean'] = df['alcohol_usage_type'].fillna('never').astype(str).str.strip().str.lower()
        
        alcohol_map = {
            'never': 0,
            'never; never': 0,
            'unknown': 0,
            'nan': 0,
            'none': 0,
            'abuse': 1,
            'sel': 1,
            'current': 1
        }
        
        df['alcohol_usage_encoded'] = df['alcohol_usage_clean'].map(alcohol_map).fillna(0).astype(int)
        df = df.drop(['alcohol_usage_type', 'alcohol_usage_clean'], axis=1)
        df = df.rename(columns={'alcohol_usage_encoded': 'alcohol_usage'})
        
        users = df['alcohol_usage'].sum()
        logger.info(f"Alcohol: Users (1): {users} ({users/len(df)*100:.2f}%)")
        
        return df
    
    # ========================================================================
    # MEDICATION (GPI) ENCODING
    # ========================================================================
    
    def _create_gpi_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure GPI medication columns exist and fill missing values with 0
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with GPI columns filled
        """
        logger.info("Processing GPI medication columns...")
        
        # For each expected GPI column
        for gpi in self.gpi_columns:
            if gpi in df.columns:
                # Convert to numeric and fill NaN with 0
                df[gpi] = pd.to_numeric(df[gpi], errors='coerce').fillna(0)
            else:
                # Create column with 0 if it doesn't exist
                df[gpi] = 0
        
        logger.info(f"Processed {len(self.gpi_columns)} GPI columns")
        
        # Log distribution
        for gpi in self.gpi_columns:
            non_zero = (df[gpi] > 0).sum()
            if non_zero > 0:
                avg_value = df[df[gpi] > 0][gpi].mean()
                logger.info(f"  GPI {gpi}: {non_zero} patients, avg value: {avg_value:.2f}")
        
        # Drop the raw 'gpi' text column if it exists
        if 'gpi' in df.columns:
            df = df.drop('gpi', axis=1)
        
        return df   
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Get list of feature names after encoding"""
        return df.columns.tolist()