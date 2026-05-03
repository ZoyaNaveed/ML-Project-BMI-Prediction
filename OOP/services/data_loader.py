import pandas as pd
import math
from datetime import datetime
from dateutil.parser import parse as dateparse
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class MongoDataExtractor:
    """
    Extracts and transforms patient data from MongoDB for BMI trajectory prediction.
    Handles complex temporal windows, vitals processing, and feature aggregation.
    """
    
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        """
        Initialize MongoDB data extractor
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
            collection_name: Collection name
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Static reference date
        self.today_static = datetime(2025, 9, 25)
        
        # Lab categories mapping
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
        
        # Build lab synonyms map
        self.lab_synonyms = {}
        for cat, names in self.lab_keep.items():
            for nm in set([cat] + names):
                self.lab_synonyms[self._normalize_name(nm)] = cat
        
        # MongoDB projection
        self.projection = {
            "_id": 0,
            "PatientID": 1,
            "demographics": 1,
            "diagnosis": 1,
            "family_history": 1,
            "lab_results": 1,
            "vitals": 1,
            "social_history": 1,
            "surgery_history": 1,
            "medications": 1,
            "latest_encounter_date": 1,
            "practice": 1,
            "Practice": 1,
            "organization": 1,
        }
    
    def connect(self):
        """Establish MongoDB connection"""
        logger.info(f"Connecting to MongoDB: {self.db_name}/{self.collection_name}")
        self.client = MongoClient(self.mongo_uri)
        self.collection = self.client[self.db_name][self.collection_name]
        # Test connection
        self.client.server_info()
        logger.info("Connected successfully")
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @staticmethod
    def _to_datetime(maybe_date) -> Optional[datetime]:
        """Convert various date formats to datetime"""
        if maybe_date is None:
            return None
        if isinstance(maybe_date, datetime):
            return maybe_date
        if isinstance(maybe_date, str):
            for f in (lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")), dateparse):
                try:
                    return f(maybe_date)
                except:
                    pass
            return None
        if isinstance(maybe_date, dict) and "$date" in maybe_date:
            s = maybe_date["$date"]
            for f in (lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")), dateparse):
                try:
                    return f(s)
                except:
                    pass
        return None
    
    @staticmethod
    def _numify(x) -> Optional[float]:
        """Convert various numeric formats to float"""
        if x is None:
            return None
        if isinstance(x, (int, float)):
            try:
                v = float(x)
                return v if math.isfinite(v) else None
            except:
                return None
        if isinstance(x, str):
            try:
                v = float(x)
                return v if math.isfinite(v) else None
            except:
                return None
        if isinstance(x, dict):
            for k in ("$numberDouble", "$numberInt", "$numberLong", "value"):
                if k in x:
                    try:
                        v = float(x[k])
                        return v if math.isfinite(v) else None
                    except:
                        return None
        return None
    
    @staticmethod
    def _stringify(x) -> str:
        """Convert value to string"""
        if x is None:
            return ""
        n = MongoDataExtractor._numify(x)
        if n is not None:
            return str(int(n)) if float(n).is_integer() else str(n)
        return str(x).strip()
    
    @staticmethod
    def _in_window(dt, start_dt, end_dt) -> bool:
        """Check if datetime is within window"""
        return (dt is not None) and (start_dt <= dt <= end_dt)
    
    @staticmethod
    def _uniq_join(values) -> str:
        """Join unique non-empty values with semicolon"""
        if not values:
            return ""
        uniq, seen = [], set()
        for v in values:
            if v is None:
                continue
            s = str(v).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        return "; ".join(uniq)
    
    @staticmethod
    def _normalize_name(s) -> str:
        """Normalize name for matching"""
        return " ".join(str(s or "").strip().split()).lower()
    
    @staticmethod
    def _norm_gender(g) -> str:
        """Normalize gender to M/F"""
        if not g:
            return ""
        s = str(g).strip().lower()
        if s in ("m", "male"):
            return "M"
        if s in ("f", "female"):
            return "F"
        return ""
    
    def _lab_category(self, name) -> Optional[str]:
        """Get lab category from name"""
        if not name:
            return None
        return self.lab_synonyms.get(self._normalize_name(name))
    
    @staticmethod
    def _years_between(d1, d2) -> Optional[int]:
        """Calculate years between two dates"""
        if not d1 or not d2:
            return None
        return relativedelta(d2, d1).years
    
    @staticmethod
    def _select_closest_prior(items, anchor_dt, months_back, get_dt):
        """Select item closest to anchor within time window"""
        if not items or not anchor_dt:
            return None
        start_dt = anchor_dt - relativedelta(months=months_back)
        candidates = []
        for it in items:
            dt = get_dt(it)
            if isinstance(dt, datetime) and start_dt <= dt <= anchor_dt:
                candidates.append(it)
        if not candidates:
            return None
        candidates.sort(key=lambda it: (anchor_dt - get_dt(it)).total_seconds())
        return candidates[0]
    
    @staticmethod
    def _get_bp_value(v, keys) -> Optional[float]:
        """Extract blood pressure value from dict"""
        for k in keys:
            if k in v:
                n = MongoDataExtractor._numify(v.get(k))
                if n is not None:
                    return n
        return None
    
    @staticmethod
    def _pick_practice(doc, demo) -> str:
        """Extract practice name from document"""
        candidates = [
            doc.get("practice"),
            doc.get("Practice"),
            doc.get("practice_name"),
            doc.get("practiceName"),
            doc.get("organization"),
            doc.get("org"),
            (demo or {}).get("practice"),
            (demo or {}).get("organization"),
        ]
        for c in candidates:
            if c:
                return str(c).strip()
        return ""
    
    # ========================================================================
    # DATA EXTRACTION METHODS
    # ========================================================================
    
    def _extract_demographics(self, doc) -> Dict[str, Any]:
        """Extract demographic information"""
        demo = {}
        if isinstance(doc.get("demographics"), list) and doc["demographics"]:
            for d in doc["demographics"]:
                if any(k in d for k in ("date_of_birth", "gender", "race", "marital_status")):
                    demo = d
                    break
            if not demo:
                demo = doc["demographics"][0]
        
        dob = self._to_datetime(demo.get("date_of_birth"))
        age_years = self._years_between(dob, self.today_static)
        
        race = demo.get("race")
        if not race:
            rm = demo.get("race_mapping")
            if isinstance(rm, list):
                race = "; ".join([str(x) for x in rm if x])
        
        return {
            'date_of_birth': dob.date().isoformat() if dob else "",
            'age_years': str(age_years) if age_years is not None else "",
            'gender': self._norm_gender(demo.get("gender")),
            'race': race or "",
            'marital_status': demo.get("marital_status") or "",
            'practice': self._pick_practice(doc, demo)
        }
    
    def _extract_vitals(self, doc, A_start, A_end, B_start, B_end) -> Optional[Dict[str, str]]:
        """Extract vitals within time windows"""
        vitals_raw = doc.get("vitals") or []
        vit_prepped = []
        
        for v in vitals_raw:
            dt = self._to_datetime(v.get("date"))
            if not isinstance(dt, datetime):
                continue
            
            hr_raw = self._numify(v.get("height"))
            wr_raw = self._numify(v.get("weight"))
            if hr_raw is None or wr_raw is None:
                continue
            
            # Detect unit system (inches/lbs vs cm/kg)
            inches_mode = hr_raw < 100
            
            # Convert height
            h_cm = hr_raw * 2.54 if inches_mode else hr_raw
            if not math.isfinite(h_cm) or h_cm <= 0 or h_cm > 300:
                continue
            
            # Convert weight
            w_kg = wr_raw * 0.453592 if inches_mode else wr_raw
            if not (w_kg and math.isfinite(w_kg) and 0 < w_kg <= 350):
                continue
            
            bmi_val = w_kg / ((h_cm / 100.0) ** 2)
            sys_val = self._get_bp_value(v, ["systolic", "systolic_bp", "bp_systolic"])
            dia_val = self._get_bp_value(v, ["diastolic", "diastolic_bp", "bp_diastolic"])
            
            vit_prepped.append({
                "_dt": dt,
                "height_cm": round(h_cm, 2),
                "weight_kg": round(w_kg, 2),
                "bmi": round(bmi_val, 3),
                "weight_category": v.get("weight_category") or "",
                "systolic_bp": sys_val,
                "diastolic_bp": dia_val,
            })
        
        # Select closest vitals in each window
        vit_A = self._select_closest_prior(
            [x for x in vit_prepped if self._in_window(x["_dt"], A_start, A_end)],
            anchor_dt=A_end, months_back=2, get_dt=lambda x: x["_dt"]
        )
        vit_B = self._select_closest_prior(
            [x for x in vit_prepped if self._in_window(x["_dt"], B_start, B_end)],
            anchor_dt=B_end, months_back=2, get_dt=lambda x: x["_dt"]
        )
        
        if not vit_A or not vit_B:
            return None
        
        return {
            'height': "; ".join([str(vit_A["height_cm"]), str(vit_B["height_cm"])]),
            'weight': "; ".join([str(vit_A["weight_kg"]), str(vit_B["weight_kg"])]),
            'bmi': "; ".join([str(vit_A["bmi"]), str(vit_B["bmi"])]),
            'weight_category': self._uniq_join([vit_A["weight_category"], vit_B["weight_category"]]),
            'systolic_bp': "; ".join([self._stringify(vit_A["systolic_bp"]), self._stringify(vit_B["systolic_bp"])]).strip(),
            'diastolic_bp': "; ".join([self._stringify(vit_A["diastolic_bp"]), self._stringify(vit_B["diastolic_bp"])]).strip(),
            'date_of_height_value': "; ".join([vit_A["_dt"].date().isoformat(), vit_B["_dt"].date().isoformat()]),
            'date_of_weight_value': "; ".join([vit_A["_dt"].date().isoformat(), vit_B["_dt"].date().isoformat()]),
        }
    
    


    def _extract_vitals(self, doc, A_start, A_end, B_start, B_end) -> Optional[Dict[str, str]]:
        """Extract vitals within time windows — only height/weight/BMI from both A and B"""
        vitals_raw = doc.get("vitals") or []
        vit_prepped = []

        for v in vitals_raw:
            dt = self._to_datetime(v.get("date"))
            if not isinstance(dt, datetime):
                continue

            hr_raw = self._numify(v.get("height"))
            wr_raw = self._numify(v.get("weight"))
            if hr_raw is None or wr_raw is None:
                continue

            inches_mode = hr_raw < 100  # detect unit system
            h_cm = hr_raw * 2.54 if inches_mode else hr_raw
            if not math.isfinite(h_cm) or h_cm <= 0 or h_cm > 300:
                continue

            w_kg = wr_raw * 0.453592 if inches_mode else wr_raw
            if not (w_kg and math.isfinite(w_kg) and 0 < w_kg <= 350):
                continue

            bmi_val = w_kg / ((h_cm / 100.0) ** 2)
            sys_val = self._get_bp_value(v, ["systolic", "systolic_bp", "bp_systolic"])
            dia_val = self._get_bp_value(v, ["diastolic", "diastolic_bp", "bp_diastolic"])

            vit_prepped.append({
                "_dt": dt,
                "height_cm": round(h_cm, 2),
                "weight_kg": round(w_kg, 2),
                "bmi": round(bmi_val, 3),
                "weight_category": v.get("weight_category") or "",
                "systolic_bp": sys_val,
                "diastolic_bp": dia_val,
            })

        # Select closest vitals in each window
        vit_A = self._select_closest_prior(
            [x for x in vit_prepped if self._in_window(x["_dt"], A_start, A_end)],
            anchor_dt=A_end, months_back=2, get_dt=lambda x: x["_dt"]
        )
        vit_B = self._select_closest_prior(
            [x for x in vit_prepped if self._in_window(x["_dt"], B_start, B_end)],
            anchor_dt=B_end, months_back=12, get_dt=lambda x: x["_dt"]
        )

        if not vit_A or not vit_B:
            return None

        return {
            'height': "; ".join([str(vit_A["height_cm"]), str(vit_B["height_cm"])]),
            'weight': "; ".join([str(vit_A["weight_kg"]), str(vit_B["weight_kg"])]),
            'bmi': "; ".join([str(vit_A["bmi"]), str(vit_B["bmi"])]),
            'weight_category': self._uniq_join([vit_A["weight_category"], vit_B["weight_category"]]),
            'systolic_bp': "; ".join([
                self._stringify(vit_B["systolic_bp"])
            ]).strip(),
            'diastolic_bp': "; ".join([
                self._stringify(vit_B["diastolic_bp"])
            ]).strip(),
            'date_of_height_value': "; ".join([
                vit_A["_dt"].date().isoformat(),
                vit_B["_dt"].date().isoformat()
            ]),
            'date_of_weight_value': "; ".join([
                vit_A["_dt"].date().isoformat(),
                vit_B["_dt"].date().isoformat()
            ]),
            'vitals_B_date': vit_B["_dt"].date().isoformat()
        }


    def _extract_labs(self, doc, B_start, B_end) -> Optional[Dict[str, str]]:
        """Extract lab results (only from Window B, up to vitals_B_date)"""
        labs_all = doc.get("lab_results") or []
        labs_prepped = []

        for lr in labs_all:
            dt = self._to_datetime(lr.get("date"))
            if not isinstance(dt, datetime):
                continue
            if not self._in_window(dt, B_start, B_end):
                continue

            nm = self._stringify(lr.get("api_test_name"))
            if not self._lab_category(nm):
                continue

            raw_res = lr.get("result") or lr.get("value") or lr.get("result_value")
            res = self._stringify(raw_res)
            if nm and res:
                labs_prepped.append({"_dt": dt, "name": nm, "result": res})

        if not labs_prepped:
            return None

        return {
            'api_test_name': "; ".join([x["name"] for x in labs_prepped]),
            'lab_name_result': "; ".join([f"{x['name']}={x['result']}" for x in labs_prepped]),
            'date_of_api_test_name': "; ".join([x["_dt"].date().isoformat() for x in labs_prepped]),
        }


    def _extract_diagnosis(self, doc, B_start, B_end) -> Dict[str, str]:
        """Extract ICD-10 diagnosis codes (only from Window B)"""
        diags = doc.get("diagnosis") or []
        icd_list = []

        for d in diags:
            dt = self._to_datetime(d.get("date"))
            code = (d.get("icd_10") or "").strip()
            if not code or not isinstance(dt, datetime):
                continue
            if self._in_window(dt, B_start, B_end):
                icd_list.append((dt, code))

        icd_list.sort(key=lambda x: x[0])
        return {
            'icd_10': "; ".join([code for _, code in icd_list]),
            'date_of_icd_code': "; ".join([dt.date().isoformat() for dt, _ in icd_list]),
        }


    def _extract_social_history(self, doc, B_start, B_end) -> Dict[str, str]:
        """Extract social history (only from Window B)"""
        soc = doc.get("social_history") or []
        soc_prepped = []

        for s in soc:
            dt = self._to_datetime(s.get("date"))
            if not isinstance(dt, datetime):
                continue
            if not self._in_window(dt, B_start, B_end):
                continue
            soc_prepped.append({
                "_dt": dt,
                "smoking_status": s.get("smoking_status"),
                "alcohol_usage_type": s.get("alcohol_usage_type")
            })

        smoking = [self._stringify(x["smoking_status"]) for x in soc_prepped if x.get("smoking_status")]
        alcohol = [self._stringify(x["alcohol_usage_type"]) for x in soc_prepped if x.get("alcohol_usage_type")]

        return {
            'smoking_status': "; ".join(smoking),
            'alcohol_usage_type': "; ".join(alcohol),
        }


    def _extract_medications(self, doc, B_start, B_end) -> Dict[str, str]:
        """Extract medications (only from Window B)"""
        meds_all = doc.get("medications") or []
        meds_prepped = []

        for m in meds_all:
            dt = self._to_datetime(m.get("date"))
            if not isinstance(dt, datetime):
                continue
            if not self._in_window(dt, B_start, B_end):
                continue

            status = (m.get("status") or "").strip().lower()
            if status != "active":
                continue

            gpi_raw = str(m.get("gpi") or "").strip()
            if len(gpi_raw) < 6:
                continue

            gpi6 = gpi_raw[:6]
            meds_prepped.append({
                "_dt": dt,
                "gpi": gpi6,
                "status": status.capitalize()
            })

        return {
            'gpi': "; ".join([x["gpi"] for x in meds_prepped]),
            'date_of_gpi': "; ".join([x["_dt"].date().isoformat() for x in meds_prepped]),
            'status_of_gpi': "; ".join([x["status"] for x in meds_prepped]),
        }


    def _process_patient(self, doc) -> Optional[Dict[str, Any]]:
        """Process single patient document"""
        pid = doc.get("PatientID")
        led_dt = self._to_datetime(doc.get("latest_encounter_date"))
        if not led_dt:
            return None

        A_start = led_dt - relativedelta(months=1)
        A_end = led_dt
        B_start = led_dt - relativedelta(months=24)
        B_end = led_dt - relativedelta(months=12)

        demographics = self._extract_demographics(doc)
        vitals = self._extract_vitals(doc, A_start, A_end, B_start, B_end)
        if not vitals:
            return None

        B_end_actual = datetime.fromisoformat(vitals['vitals_B_date'])

        labs = self._extract_labs(doc, B_start, B_end_actual)
        diagnosis = self._extract_diagnosis(doc, B_start, B_end_actual)
        family_history = self._extract_family_history(doc, B_start, B_end_actual)
        social_history = self._extract_social_history(doc, B_start, B_end_actual)
        surgery_history = self._extract_surgery_history(doc)
        medications = self._extract_medications(doc, B_start, B_end_actual)

        if not labs:
            return None

        practice = demographics['practice']
        patient_practice_id = f"{pid}__{practice}" if practice else f"{pid}__"

        return {
            'patient_practice_id': patient_practice_id,
            'PatientID': pid,
            'date_of_birth': demographics['date_of_birth'],
            'age_years': demographics['age_years'],
            'gender': demographics['gender'],
            'race': demographics['race'],
            'marital_status': demographics['marital_status'],
            'icd_10': diagnosis['icd_10'],
            'date_of_icd_code': diagnosis['date_of_icd_code'],
            'jsdisease': family_history,
            'api_test_name': labs['api_test_name'],
            'date_of_api_test_name': labs['date_of_api_test_name'],
            'lab_name_result': labs['lab_name_result'],
            'height': vitals['height'],
            'date_of_height_value': vitals['date_of_height_value'],
            'weight': vitals['weight'],
            'date_of_weight_value': vitals['date_of_weight_value'],
            'bmi': vitals['bmi'],
            'weight_category': vitals['weight_category'],
            'systolic_bp': vitals['systolic_bp'],
            'diastolic_bp': vitals['diastolic_bp'],
            'smoking_status': social_history['smoking_status'],
            'alcohol_usage_type': social_history['alcohol_usage_type'],
            'surgery_name': surgery_history['surgery_name'],
            'date_of_surgery_name': surgery_history['date_of_surgery_name'],
            'gpi': medications['gpi'],
            'date_of_gpi': medications['date_of_gpi'],
            'status_of_gpi': medications['status_of_gpi'],
        }
     
    def extract_all_patients(self) -> pd.DataFrame:
        """Extract all patients from MongoDB"""
        logger.info("Starting patient data extraction...")
        
        cursor = self.collection.find({}, self.projection)
        rows = []
        
        processed = 0
        skipped = 0
        
        for doc in cursor:
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"Processed {processed} patients, extracted {len(rows)} records")
            
            row = self._process_patient(doc)
            if row:
                rows.append(row)
            else:
                skipped += 1
        
        logger.info(f"Extraction complete: {processed} processed, {len(rows)} extracted, {skipped} skipped")
        
        # Define column order
        columns = [
            "patient_practice_id", "PatientID", "date_of_birth", "age_years", "gender", "race", "marital_status",
            "icd_10", "date_of_icd_code", "jsdisease", "api_test_name", "date_of_api_test_name", "lab_name_result",
            "height", "date_of_height_value", "weight", "date_of_weight_value", "bmi", "weight_category",
            "systolic_bp", "diastolic_bp", "smoking_status", "alcohol_usage_type",
            "surgery_name", "date_of_surgery_name", "gpi", "date_of_gpi", "status_of_gpi",
        ]
        
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()