# =============================================================================
# ETL Pipeline — Phase 0B: Parse Unstructured Clinical Notes → Structured CSVs
# Project : Early Disease Prediction Using Healthcare Data Warehouse
# Script  : etl_pipeline/parse_unstructured.py
# =============================================================================

import os
import re
import sys
import json
from typing import cast

import numpy as np
import pandas as pd

from note_styles_toddler import QCHAT_SOCIAL

# ---------------------------------------------------------------------------
# 0.  Path Resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
UNSTRUCTURED_DIR = os.path.join(ROOT_DIR, "datasets", "unstructured")
CLEANED_DIR = os.path.join(ROOT_DIR, "datasets", "cleaned")

os.makedirs(CLEANED_DIR, exist_ok=True)

IN_AUTISM = os.path.join(UNSTRUCTURED_DIR, "autism_clinical_notes.txt")
IN_TODDLER = os.path.join(UNSTRUCTURED_DIR, "toddler_social_media_posts.txt")
IN_DIABETES = os.path.join(UNSTRUCTURED_DIR, "diabetes_clinical_notes.txt")
IN_STROKE = os.path.join(UNSTRUCTURED_DIR, "stroke_clinical_notes.txt")

OUT_AUTISM = os.path.join(CLEANED_DIR, "autism_cleaned.csv")
OUT_TODDLER = os.path.join(CLEANED_DIR, "toddler_autism_cleaned.csv")
OUT_DIABETES = os.path.join(CLEANED_DIR, "diabetes_cleaned.csv")
OUT_STROKE = os.path.join(CLEANED_DIR, "stroke_cleaned.csv")

def banner(title: str, width: int = 72, char: str = "=") -> None:
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)

def sub_step(message: str) -> None:
    print(f"    ▸  {message}")

def assert_no_missing(df: pd.DataFrame, dataset_name: str) -> None:
    remaining = int(df.isnull().sum().sum())
    if remaining > 0:
        bad_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"[{dataset_name}] Extraction incomplete — "
            f"{remaining} NaN(s) remain in: {bad_cols}"
        )
    print("    ✔  Zero missing values confirmed.")

def split_records(text: str) -> list[str]:
    records = re.split(r"---\s*END OF RECORD\s*#\d+\s*---", text)
    return [r.strip() for r in records if r.strip()]

# ---------------------------------------------------------------------------
# 1.  PARSE AUTISM CLINICAL NOTES
# ---------------------------------------------------------------------------
def parse_autism_record(record: str) -> dict:
    data: dict = {}
    record_lower = record.lower()

    age_m = re.search(r'(\d+)\s*(?:yo\b|years?\s+old|-?yr-old|y/o|-year-old)|(?:age|aged)\s*(\d+)', record_lower)
    if age_m: data['Age'] = int(age_m.group(1) or age_m.group(2))

    if re.search(r'\b(male|boy|m)\b', record_lower): data['Sex'] = 1
    elif re.search(r'\b(female|girl|f)\b', record_lower): data['Sex'] = 0

    pos_j = re.search(r'(history of neonatal jaundice|jaundice: positive|hx neonatal jaundice noted|born with jaundice|neonatal icterus documented|jaundice\(\+\)|neonatal jaundice present|jaundice hx: yes)', record_lower)
    neg_j = re.search(r'(no history of neonatal|jaundice: negative|denies neonatal jaundice|no neonatal icterus|jaundice\(-\)|no jaundice reported|jaundice hx: no|unremarkable for jaundice)', record_lower)
    if neg_j: data['Jaundice'] = 0
    elif pos_j: data['Jaundice'] = 1
    else: data['Jaundice'] = 1 if 'jaundice' in record_lower and not any(x in record_lower for x in ['no ', 'negative', '(-)']) else 0

    pos_f = re.search(r'(positive family history|family hx of autism spectrum disorder reported|fam hx asd: positive|diagnosed with asd|first-degree relative|family autism history: yes|asd runs in the family|fam asd\(\+\))', record_lower)
    neg_f = re.search(r'(no family history|family hx of asd: negative|fam hx asd: unremarkable|no known family members|denies family history|family hx: no asd|family autism history: no|no asd in family|fam asd\(-\))', record_lower)
    if neg_f: data['Family_ASD'] = 0
    elif pos_f: data['Family_ASD'] = 1
    else: data['Family_ASD'] = 0

    pos_c = re.search(r'(classification: yes|positive for asd|asd screen positive|positive \(score|asd likely|asd positive|asd screen pos)', record_lower)
    neg_c = re.search(r'(classification: no|negative for asd|asd screen negative|negative \(score|asd unlikely|asd negative|asd screen neg)', record_lower)
    if pos_c: data['Class'] = 1
    elif neg_c: data['Class'] = 0
    else: data['Class'] = 0

    aq_vals = {f"A{i}": 0 for i in range(1, 11)}
    endorsed_m = re.search(r'(?:endorsed items:|patient endorsed the following:|patient endorsed)(.*?)(?:\.|all other|remaining items)', record_lower)
    
    if endorsed_m:
        for i in range(1, 11):
            if f"a{i}" in endorsed_m.group(1): aq_vals[f"A{i}"] = 1
    elif re.search(r'(no items endorsed|did not endorse any of the aq-10)', record_lower):
        pass
    else:
        for i in range(1, 11):
            if re.search(rf'a{i}\(\+\)', record_lower): aq_vals[f"A{i}"] = 1
            elif re.search(rf'a{i}\(\-\)', record_lower): aq_vals[f"A{i}"] = 0
            elif re.search(rf'a{i}:\s*(yes|\+)', record_lower): aq_vals[f"A{i}"] = 1
            elif re.search(rf'a{i}:\s*(no|\-)', record_lower): aq_vals[f"A{i}"] = 0
            elif re.search(rf'a{i}\s*=\s*1', record_lower): aq_vals[f"A{i}"] = 1
            elif re.search(rf'a{i}\s*=\s*0', record_lower): aq_vals[f"A{i}"] = 0
            else:
                m = re.search(rf'item a{i}:.*?—\s*(endorsed|not endorsed)', record_lower)
                if m and m.group(1) == 'endorsed': aq_vals[f"A{i}"] = 1

    for k, v in aq_vals.items(): data[k] = v
    data['AQ_Score'] = sum(aq_vals.values())
    return data

def parse_autism_notes(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f: text = f.read()
    records = split_records(text)
    parsed = [parse_autism_record(rec) for rec in records]
    df = pd.DataFrame(parsed)
    expected_cols = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Age", "Sex", "Jaundice", "Family_ASD", "Class", "AQ_Score"]
    df = df[expected_cols]
    int8_cols = [f"A{i}" for i in range(1, 11)] + ["Sex", "Jaundice", "Family_ASD", "Class", "AQ_Score"]
    df[int8_cols] = df[int8_cols].astype(np.int8)
    df["Age"] = df["Age"].astype(np.int16)
    return df

# ---------------------------------------------------------------------------
# 2.  PARSE TODDLER AUTISM SOCIAL MEDIA POSTS
# ---------------------------------------------------------------------------
def normalize_social(text: str) -> str:
    """Reverse typos to help matching."""
    return text.replace("bc", "because").replace("w/", "with").replace(" u ", " you ")

def parse_toddler_record(record: str) -> dict:
    data = {}
    record_lower = normalize_social(record.lower())

    age_m = re.search(r'(\d+)\s*(?:month)', record_lower)
    if age_m: data['Age_Mons'] = int(age_m.group(1))
    
    if re.search(r'\b(son)\b', record_lower): data['Sex'] = 'm'
    elif re.search(r'\b(daughter)\b', record_lower): data['Sex'] = 'f'
    else: data['Sex'] = 'm'

    if re.search(r'(had jaundice|jaundice as a baby)', record_lower): data['Jaundice'] = 'yes'
    else: data['Jaundice'] = 'no'
    
    if re.search(r'(autism runs in our family|runs in the family)', record_lower): data['Family_mem_with_ASD'] = 'yes'
    else: data['Family_mem_with_ASD'] = 'no'

    # Ethnicity and Who completed
    eth_m = re.search(r'\(([^,]+),\s*([^)]+)\)', record_lower)
    if eth_m:
        data['Who completed the test'] = eth_m.group(1).strip()
        data['Ethnicity'] = eth_m.group(2).strip()
    else:
        data['Who completed the test'] = 'family member'
        data['Ethnicity'] = 'White European'

    # Q-Chat parsing
    for i in range(1, 11):
        item = f"A{i}"
        found = False
        # check positive traits (1)
        for phrase in QCHAT_SOCIAL[item][1]:
            if normalize_social(phrase.lower()) in record_lower:
                data[item] = 1
                found = True
                break
        if not found:
            # check negative traits (0)
            for phrase in QCHAT_SOCIAL[item][0]:
                if normalize_social(phrase.lower()) in record_lower:
                    data[item] = 0
                    found = True
                    break
        if not found:
            data[item] = 0 # Default if not matched
            
    data['Qchat-10-Score'] = sum(data[f"A{i}"] for i in range(1, 11))
    
    # Class mapping: in the original CSV, Qchat > 3 or so usually means Yes.
    # To precisely match the original logic without training a model,
    # we'll use a simple threshold or just set it based on score > 3.
    data['Class/ASD Traits '] = "Yes" if data['Qchat-10-Score'] > 3 else "No"
    
    return data

def parse_toddler_notes(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f: text = f.read()
    records = split_records(text)
    parsed = [parse_toddler_record(rec) for rec in records]
    df = pd.DataFrame(parsed)
    expected_cols = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Age_Mons", "Qchat-10-Score", "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who completed the test", "Class/ASD Traits "]
    df = df[expected_cols]
    int8_cols = [f"A{i}" for i in range(1, 11)] + ["Age_Mons", "Qchat-10-Score"]
    df[int8_cols] = df[int8_cols].astype(np.int8)
    return df

# ---------------------------------------------------------------------------
# 3.  PARSE DIABETES CLINICAL NOTES & IoT LOGS
# ---------------------------------------------------------------------------
DIABETES_SYMPTOMS = {
    "Polyuria": ["excessive urination", "polyuria", "frequent urination", "polyurea", "urinary freq"],
    "Polydipsia": ["excessive thirst", "polydipsia", "increased thirst", "constant thirst", "polydpsia"],
    "sudden weight loss": ["sudden weight loss", "unexplained wt loss", "rapid weight decrease", "unintentional weight loss"],
    "weakness": ["weakness", "generalized fatigue", "feeling weak and tired", "fatigue and weakness"],
    "Polyphagia": ["excessive hunger", "polyphagia", "increased appetite", "constant hunger", "polyphaiga"],
    "Genital thrush": ["genital thrush", "recurrent genital yeast", "genital candidiasis", "thrush infections"],
    "visual blurring": ["visual blurring", "blurred vision", "blurry vision", "visual disturbance"],
    "Itching": ["itching", "persistent pruritus", "skin itching", "generalized itching"],
    "Irritability": ["irritability", "mood changes", "irritable mood", "increased irritability"],
    "delayed healing": ["delayed healing", "slow wound healing", "wounds heal slowly", "poor wound healing"],
    "partial paresis": ["partial paresis", "muscle weakness in extremities", "limb weakness", "partial paralysis"],
    "muscle stiffness": ["muscle stiffness", "joint stiffness", "stiff muscles and joints", "musculoskeletal stiffness"],
    "Alopecia": ["alopecia", "hair loss", "hair thinning", "losing hair"],
    "Obesity": ["obesity", "clinically obese", "bmi > 30", "obese"],
}

def parse_diabetes_iot(record: str) -> dict:
    data = {}
    if "{" in record and "}" in record:
        # JSON format
        try:
            match = re.search(r'(\{.*\})', record, re.DOTALL)
            if match:
                payload = json.loads(match.group(1))
                data['Age'] = payload['patient']['age']
                data['Gender'] = 1 if payload['patient']['gender'] == 'M' else 0
                data['Class'] = 1 if payload['telemetry']['risk_classification'] == 'HIGH' else 0
                
                flags = payload['telemetry']['symptoms_flagged']
                for col in DIABETES_SYMPTOMS:
                    data[col] = 1 if col.upper().replace(" ", "_") in flags else 0
        except:
            pass
    elif "SYS_LOG" in record:
        # Syslog format
        age_m = re.search(r'PT_AGE=(\d+)', record)
        if age_m: data['Age'] = int(age_m.group(1))
        
        g_m = re.search(r'GENDER=(M|F)', record)
        if g_m: data['Gender'] = 1 if g_m.group(1) == 'M' else 0
        
        c_m = re.search(r'RISK_CLASS=(HIGH|LOW)', record)
        if c_m: data['Class'] = 1 if c_m.group(1) == 'HIGH' else 0
        
        s_m = re.search(r'SYMPT_FLAG=([A-Z_,]+)', record)
        flags = s_m.group(1).split(",") if s_m and s_m.group(1) != "NONE" else []
        for col in DIABETES_SYMPTOMS:
            data[col] = 1 if col.upper().replace(" ", "_") in flags else 0
            
    return data

def parse_diabetes_record(record: str) -> dict:
    if "SYS_LOG" in record or ("{" in record and "telemetry" in record):
        return parse_diabetes_iot(record)

    # Standard NLP Parsing
    data = {}
    record_lower = record.lower()

    age_m = re.search(r'(\d+)\s*(?:yo\b|years?\s+old|-?yr-old|y/o|-year-old)|(?:age|aged)\s*(\d+)', record_lower)
    if age_m: data['Age'] = int(age_m.group(1) or age_m.group(2))

    if re.search(r'\b(male|boy|m)\b', record_lower): data['Gender'] = 1
    elif re.search(r'\b(female|girl|f)\b', record_lower): data['Gender'] = 0

    if re.search(r'(positive|high risk)', record_lower): data['Class'] = 1
    elif re.search(r'(negative|low risk)', record_lower): data['Class'] = 0

    reports_match = re.search(r'(?:reports the following|complains of|presents with|positive findings:|reports)(.*?)(?:denies|assessment|plan|all other)', record_lower, re.DOTALL)
    denies_match = re.search(r'(?:denies|specifically denies)(.*?)(?:assessment|plan)', record_lower, re.DOTALL)
    
    reports_text = reports_match.group(1) if reports_match else ""
    denies_text = denies_match.group(1) if denies_match else ""

    all_denied = bool(re.search(r'(denies all screened symptoms|all screened symptoms denied|all other screened symptoms denied)', record_lower))

    for sym_col, synonyms in DIABETES_SYMPTOMS.items():
        data[sym_col] = 0
        found_compact = False
        for syn in synonyms:
            syn_safe = re.escape(syn)
            if re.search(rf'{syn_safe}\(\+\)', record_lower):
                data[sym_col] = 1; found_compact = True; break
            elif re.search(rf'{syn_safe}\(\-\)', record_lower):
                data[sym_col] = 0; found_compact = True; break
        
        if found_compact: continue

        if all_denied:
            if any(syn in reports_text for syn in synonyms): data[sym_col] = 1
            else: data[sym_col] = 0
        else:
            if any(syn in reports_text for syn in synonyms): data[sym_col] = 1
            elif any(syn in denies_text for syn in synonyms): data[sym_col] = 0
            else:
                if any(syn in record_lower for syn in synonyms) and not denies_match:
                    data[sym_col] = 1
    return data

def parse_diabetes_notes(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f: text = f.read()
    records = split_records(text)
    parsed = [parse_diabetes_record(rec) for rec in records]
    df = pd.DataFrame(parsed)
    expected_cols = ["Age", "Gender"] + list(DIABETES_SYMPTOMS.keys()) + ["Class"]
    df = df[expected_cols]
    binary_cols = ["Gender"] + list(DIABETES_SYMPTOMS.keys()) + ["Class"]
    df[binary_cols] = df[binary_cols].astype(np.int8)
    df["Age"] = df["Age"].astype(np.int16)
    return df

# ---------------------------------------------------------------------------
# 4.  PARSE STROKE CLINICAL NOTES
# ---------------------------------------------------------------------------
def parse_stroke_record(record: str) -> dict:
    data = {}
    record_lower = record.lower()

    if re.search(r'\b(male|boy)\b', record_lower) or re.search(r'\b(m)\b(,| \bmarried)', record_lower): data['gender'] = 1
    elif re.search(r'\b(female|girl)\b', record_lower) or re.search(r'\b(f)\b(,| \bmarried)', record_lower): data['gender'] = 0
    else: data['gender'] = -1

    age_m = re.search(r'([\d.]+)\s*(?:years?\s+of\s+age|yo|years?\s+old|yr-old|y/o|-year-old)|(?:age|aged)\s*([\d.]+)', record_lower)
    if age_m: data['age'] = float(age_m.group(1) or age_m.group(2))

    if re.search(r'(currently or has been previously married|has been married|married: y\b|\bmarried\b)', record_lower) and not re.search(r'(never been married|never married|single/never married|married: n\b)', record_lower):
        data['ever_married'] = 1
    elif re.search(r'(never been married|never married|single/never married|married: n\b)', record_lower):
        data['ever_married'] = 0
    else: data['ever_married'] = 0

    work_m = re.search(r'(?:employment status|work|employment):\s*(private|self-employed|govt_job|children|never_worked)', record_lower, re.IGNORECASE)
    if work_m: data['work_type'] = work_m.group(1).title() if work_m.group(1).lower() != 'govt_job' else 'Govt_job'
    else:
        work_m2 = re.search(r'works in\s+(the\s+)?(private|self-employed|govt_job|children|never_worked)', record_lower, re.IGNORECASE)
        if work_m2: data['work_type'] = work_m2.group(2).title() if work_m2.group(2).lower() != 'govt_job' else 'Govt_job'

    res_m = re.search(r'(?:resides in an?|lives in an?|residence:)\s*(urban|rural)', record_lower, re.IGNORECASE)
    if res_m: data['Residence_type'] = 1 if res_m.group(1).lower() == 'urban' else 0
    else: data['Residence_type'] = 1

    if re.search(r'(htn\(\+\)|documented hypertension|history significant for hypertension)', record_lower) or (re.search(r'significant for.*hypertension', record_lower)): data['hypertension'] = 1
    elif re.search(r'(htn\(-\)|no hypertension|htn\(-|no significant comorbidities)', record_lower): data['hypertension'] = 0
    else: data['hypertension'] = 0

    if re.search(r'(heart dz\(\+\)|history of heart disease|significant for.*heart disease)', record_lower): data['heart_disease'] = 1
    elif re.search(r'(heart dz\(-\)|no heart disease|heart dz\(-|no significant comorbidities)', record_lower): data['heart_disease'] = 0
    else: data['heart_disease'] = 0

    gluc_m = re.search(r'(?:glucose level of|glucose level:|glucose:|glucose \(avg\):|glucose)\s*([\d.]+)', record_lower)
    if gluc_m: data['avg_glucose_level'] = float(gluc_m.group(1))

    bmi_m = re.search(r'(?:bmi:?|bmi of)\s*([\d.]+)', record_lower)
    if bmi_m: data['bmi'] = float(bmi_m.group(1))
    else: data['bmi'] = np.nan

    smok_m = re.search(r'(?:smoking(?: history|):|tobacco use:)\s*(formerly smoked|never smoked|smokes|unknown)', record_lower)
    if smok_m:
        s = smok_m.group(1).lower()
        if s == 'formerly smoked': data['smoking_status'] = 'formerly smoked'
        elif s == 'never smoked': data['smoking_status'] = 'never smoked'
        elif s == 'smokes': data['smoking_status'] = 'smokes'
        elif s == 'unknown': data['smoking_status'] = 'Unknown'

    if re.search(r'(outcome: stroke|stroke occurred|stroke confirmed|acute stroke)', record_lower) and not re.search(r'(outcome: no stroke|no stroke|no cerebrovascular event)', record_lower): data['Class'] = 1
    elif re.search(r'(outcome: no stroke|no stroke|no cerebrovascular event)', record_lower): data['Class'] = 0
    else: data['Class'] = 0

    return data

def parse_stroke_notes(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f: text = f.read()
    records = split_records(text)
    parsed = [parse_stroke_record(rec) for rec in records]
    df = pd.DataFrame(parsed)

    other_mask = df["gender"] == -1
    if other_mask.sum() > 0: df = df[~other_mask].reset_index(drop=True)

    AGE_BINS = [0.0, 18.0, 40.0, 60.0, 80.0, float("inf")]
    AGE_LABELS = ["0-18", "19-40", "41-60", "61-80", "81+"]

    if df["bmi"].isnull().sum() > 0:
        df["age_bracket"] = pd.cut(cast(pd.Series, df["age"]), bins=AGE_BINS, labels=AGE_LABELS, right=True)
        df["bmi"] = pd.Series(df.groupby("age_bracket", observed=True)["bmi"].transform(lambda grp: grp.fillna(grp.median())), index=df.index)
        if df["bmi"].isnull().sum() > 0: df["bmi"] = df["bmi"].fillna(float(df["bmi"].median()))
        df = df.drop(columns=["age_bracket"])

    expected_cols = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "Class"]
    df = df[expected_cols]
    int8_cols = ["gender", "hypertension", "heart_disease", "ever_married", "Residence_type", "Class"]
    for col in int8_cols: df[col] = df[col].astype(np.int8)
    df["age"] = df["age"].astype(np.float32)
    df["avg_glucose_level"] = df["avg_glucose_level"].astype(np.float32)
    df["bmi"] = df["bmi"].astype(np.float32)
    return df

# ===========================================================================
# 5.  MAIN EXECUTION
# ===========================================================================
def main() -> None:
    banner("PHASE 0B — PARSING TRULY UNSTRUCTURED DATA (Clinical/Social/IoT)")
    
    for filepath in [IN_AUTISM, IN_TODDLER, IN_DIABETES, IN_STROKE]:
        if not os.path.isfile(filepath):
            print(f"  [ERROR] Missing unstructured file: {filepath}")
            sys.exit(1)

    banner("STEP 1 — PARSING: AUTISM CLINICAL NOTES", char="-")
    df_autism = parse_autism_notes(IN_AUTISM)
    sub_step(f"Extracted: {df_autism.shape[0]:,} records × {df_autism.shape[1]} cols")
    assert_no_missing(df_autism, "Autism")

    banner("STEP 2 — PARSING: TODDLER SOCIAL MEDIA POSTS", char="-")
    df_toddler = parse_toddler_notes(IN_TODDLER)
    sub_step(f"Extracted: {df_toddler.shape[0]:,} records × {df_toddler.shape[1]} cols")
    assert_no_missing(df_toddler, "Toddler")

    banner("STEP 3 — PARSING: DIABETES CLINICAL NOTES + IoT LOGS", char="-")
    df_diabetes = parse_diabetes_notes(IN_DIABETES)
    sub_step(f"Extracted: {df_diabetes.shape[0]:,} records × {df_diabetes.shape[1]} cols")
    assert_no_missing(df_diabetes, "Diabetes")

    banner("STEP 4 — PARSING: STROKE CLINICAL NOTES", char="-")
    df_stroke = parse_stroke_notes(IN_STROKE)
    sub_step(f"Extracted: {df_stroke.shape[0]:,} records × {df_stroke.shape[1]} cols")
    assert_no_missing(df_stroke, "Stroke")

    banner("STEP 5 — MERGING TODDLER AUTISM INTO MAIN AUTISM DATASET", char="-")
    df_toddler_mapped = df_toddler.copy()
    df_toddler_mapped['Age'] = (df_toddler_mapped['Age_Mons'] / 12.0).round().astype(np.int16)
    df_toddler_mapped['Sex'] = df_toddler_mapped['Sex'].map({'m': 1, 'f': 0}).astype(np.int8)
    df_toddler_mapped['Jaundice'] = df_toddler_mapped['Jaundice'].map({'yes': 1, 'no': 0}).astype(np.int8)
    df_toddler_mapped['Family_ASD'] = df_toddler_mapped['Family_mem_with_ASD'].map({'yes': 1, 'no': 0}).astype(np.int8)
    df_toddler_mapped['Class'] = df_toddler_mapped['Class/ASD Traits '].map({'Yes': 1, 'No': 0}).astype(np.int8)
    df_toddler_mapped['AQ_Score'] = df_toddler_mapped['Qchat-10-Score'].astype(np.int8)
    
    expected_cols = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Age", "Sex", "Jaundice", "Family_ASD", "Class", "AQ_Score"]
    df_toddler_mapped = df_toddler_mapped[expected_cols]
    
    df_autism_combined = pd.concat([df_autism, df_toddler_mapped], ignore_index=True)
    sub_step(f"Combined Autism size: {df_autism_combined.shape[0]:,} records")

    banner("STEP 6 — SAVING STRUCTURED CSVs", char="-")
    for fp, df, fn in [(OUT_AUTISM, df_autism_combined, "autism_cleaned.csv"),
                       (OUT_DIABETES, df_diabetes, "diabetes_cleaned.csv"),
                       (OUT_STROKE, df_stroke, "stroke_cleaned.csv")]:
        df.to_csv(fp, index=False)
        print(f"  [SAVED] {fn:<30} {df.shape[0]:>6,} rows")

    print("\n  Extraction successfully handled NLP variations, typos, and IoT JSON/Syslog formats.")

if __name__ == "__main__":
    main()
