# -*- coding: utf-8 -*-
"""
Export patient-by-GPI6 medication dosages from MongoDB for a specified subset of patients.
 
- Filters: meds with status == "Active"
- Time window per patient: (LED - 18 months, LED - 6 months), exclusive of both boundaries
- Dose contribution = dose_g * frequency * overlap_days
- Aggregates by GPI-6 and writes a wide CSV
- Processes only patients listed in an input CSV (e.g. 400k patients)
"""
 
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional
from dateutil import parser as dtparser
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
import pandas as pd
import os

# ---- Config -----------------------------------------------------------------
 
MONGODB_URI = (
    "mongodb://bcu25:bcu25%40226mongo@172.16.101.226:27017/"
)
DB_NAME = "Bootcamp_2025"
COLLECTION_NAME = "Bmi_trajectory_v2"
 
# CSV of patients to include
TARGET_PATIENTS_CSV = r"5-7 Months\patients_longwindow.csv"   # <--- place your 400k patient CSV here
OUTPUT_CSV = "gpi6_dosages_18to6_months_filtered.csv"
 
RANDOM_SAMPLE = False  # unused here but kept for compatibility
 
BASE_QUERY = {
    "latest_encounter_date": {"$exists": True, "$ne": None},
    "medications": {"$exists": True},
}
 
PROJECTION = {
    "_id": 1,
    "PatientID": 1,
    "Practice": 1,
    "latest_encounter_date": 1,
    "medications": 1,
}
 
# ---- Helpers ----------------------------------------------------------------
 
def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return dtparser.isoparse(str(value))
    except Exception:
        return None
 
 
def exclusive_overlap_days(med_start: datetime, med_end: datetime, led: datetime) -> int:
    """Days of overlap between [med_start, med_end] and the exclusive window (LED-18mo, LED-6mo)."""
    if not (med_start and med_end and led) or med_end < med_start:
        return 0
    ws = (led - relativedelta(months=18)).date()
    we = (led - relativedelta(months=6)).date()
    inner_start = ws + timedelta(days=1)
    inner_end = we - timedelta(days=1)
    if inner_end < inner_start:
        return 0
    s, e = med_start.date(), med_end.date()
    start, end = max(s, inner_start), min(e, inner_end)
    if end < start:
        return 0
    return (end - start).days
 
 
def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None
 
# ---- Fetchers ---------------------------------------------------------------
 
def fetch_patients(coll) -> Iterable[Dict[str, Any]]:
    """Stream all patient docs from Mongo."""
    cursor = coll.find(BASE_QUERY, PROJECTION, no_cursor_timeout=True)
    try:
        for d in cursor:
            yield d
    finally:
        cursor.close()
 
# ---- Core aggregation -------------------------------------------------------
 
def compute_patient_gpi6_totals(doc: Dict[str, Any]) -> Dict[str, float]:
    """For a single patient doc, return {gpi6: total_dose_g} within the exclusive window."""
    led = parse_dt(doc.get("latest_encounter_date"))
    if not led:
        return {}
    meds = doc.get("medications") or []
    totals: Dict[str, float] = {}
    for m in meds:
        if (m.get("status") or "").strip().lower() != "active":
            continue
        start = parse_dt(m.get("date"))
        end = parse_dt(m.get("end_date"))
        if end is None and start is not None:
            sigp = m.get("sig_parsed") or {}
            days_from_sig = sigp.get("days")
            if days_from_sig is not None:
                try:
                    end = start + timedelta(days=int(days_from_sig))
                except Exception:
                    pass
        if start is None or end is None:
            continue
        overlap_days = exclusive_overlap_days(start, end, led)
        if overlap_days <= 0:
            continue
        sigp = m.get("sig_parsed") or {}
        dose_g = safe_float(sigp.get("dose_g", m.get("dose_g")))
        freq = safe_float(sigp.get("frequency", m.get("frequency")))
        if dose_g is None or freq is None:
            continue
        gpi = (m.get("gpi") or "").strip()
        if len(gpi) < 6:
            continue
        gpi6 = gpi[:6]
        total = dose_g * freq * overlap_days
        if total <= 0:
            continue
        totals[gpi6] = totals.get(gpi6, 0.0) + total
    return totals
 
 
# ---- Main -------------------------------------------------------------------
 
def main():

    if not os.path.exists(TARGET_PATIENTS_CSV):
        raise FileNotFoundError(f"Target patient CSV not found: {TARGET_PATIENTS_CSV}")

    target_df = pd.read_csv(TARGET_PATIENTS_CSV)
    if "patient_practice_id" not in target_df.columns:
        raise ValueError("CSV must contain a column: patient_practice_id")

    target_df["patient_uid"] = target_df["patient_practice_id"].astype(str)
    target_ids = set(target_df["patient_uid"])
    print(f"Loaded {len(target_ids):,} target patients to include.")





#     # 1️⃣ Load target patients
#     if not os.path.exists(TARGET_PATIENTS_CSV):
#         raise FileNotFoundError(f"Target patient CSV not found: {TARGET_PATIENTS_CSV}")

#     if "patient_practice_id" not in target_df.columns:
#     raise ValueError("CSV must contain a column: patient_practice_id")

# target_df["patient_uid"] = target_df["patient_practice_id"].astype(str)
# target_ids = set(target_df["patient_uid"])


#     # target_df = pd.read_csv(TARGET_PATIENTS_CSV)
#     # if not {"Practice", "PatientID"}.issubset(target_df.columns):
#     #     raise ValueError("CSV must contain columns: Practice and PatientID")
 
#     # target_df["patient_uid"] = target_df["Practice"].astype(str) + "::" + target_df["PatientID"].astype(str)
#     # target_ids = set(target_df["patient_uid"].astype(str))
#     # print(f"Loaded {len(target_ids):,} target patients to include.")
 
    # 2️⃣ Connect to Mongo
    client = MongoClient(
        MONGODB_URI,
        socketTimeoutMS=None,
        connectTimeoutMS=600000000,
        serverSelectionTimeoutMS=600000000,
    )
    coll = client[DB_NAME][COLLECTION_NAME]
 
    # 3️⃣ Process patients
    patient_rows, all_gpi6 = [], set()
    processed = 0
    matched = 0
 
    for doc in fetch_patients(coll):

        practice = (doc.get("Practice") or "").strip()
        pid = str(doc.get("PatientID") or "").strip()
        if not practice or not pid:
            continue
        patient_uid = f"{pid}__{practice}"
        if patient_uid not in target_ids:
            continue  # skip if not in your 400k list



        # practice = (doc.get("Practice") or "").strip()
        # pid = str(doc.get("PatientID") or "").strip()
        # if not practice or not pid:
        #     continue
        # patient_uid = f"{practice}::{pid}"

        matched += 1
        totals = compute_patient_gpi6_totals(doc)
        row = {"patient_uid": patient_uid}
        row.update(totals)
        patient_rows.append(row)
        all_gpi6.update(totals.keys())
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed} patients... (matched {matched})")
 
    print(f"Finished looping. {matched:,} of your target patients were found in Mongo.")
 
    if not patient_rows:
        print("No rows computed. Check filters/time window or patient list.")
        return
 
    df = pd.DataFrame(patient_rows).fillna(0.0)
    gpi6_sorted = sorted(all_gpi6)
    for g in gpi6_sorted:
        if g not in df.columns:
            df[g] = 0.0
    df = df[["patient_uid"] + gpi6_sorted].round(6)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} patients and {len(gpi6_sorted)} GPI-6 columns to {OUTPUT_CSV}")
 
 
if __name__ == "__main__":
    main()