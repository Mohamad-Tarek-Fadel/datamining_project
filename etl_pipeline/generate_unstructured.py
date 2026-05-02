# =============================================================================
# ETL Pipeline — Phase 0: Generate REALISTIC Unstructured Clinical Notes
# Project : Early Disease Prediction Using Healthcare Data Warehouse
# Script  : etl_pipeline/generate_unstructured.py
# =============================================================================

import os
import random
import sys

import pandas as pd

from note_styles import (
    DOCTORS, NOISE_AUTISM, NOISE_DIABETES, NOISE_STROKE,
    HEADERS_AUTISM, HEADERS_DIABETES, HEADERS_STROKE,
    SYMPTOM_SYNONYMS, AQ_ITEMS_FULL, AQ_DESCRIPTIONS,
    age_sex_phrase, jaundice_phrase, family_asd_phrase,
    aq10_block, asd_class_phrase, diabetes_class_phrase,
    stroke_class_phrase, diabetes_symptoms_block,
    stroke_demographics_block, stroke_comorbid_block,
    stroke_measurements_block, inject_typos,
)

from note_styles_toddler import (
    toddler_social_media_post, diabetes_style_iot_log
)

# ---------------------------------------------------------------------------
# 0.  Path Resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
UNSTRUCTURED_DIR = os.path.join(ROOT_DIR, "datasets", "unstructured")
os.makedirs(UNSTRUCTURED_DIR, exist_ok=True)

RAW_AUTISM = os.path.join(DATASETS_DIR, "Autism_Screening_Data_Combined.csv")
RAW_TODDLER = os.path.join(DATASETS_DIR, "Toddler Autism dataset July 2018.csv")
RAW_DIABETES = os.path.join(DATASETS_DIR, "diabetes_data_upload.csv")
RAW_STROKE = os.path.join(DATASETS_DIR, "healthcare-dataset-stroke-data.csv")

OUT_AUTISM = os.path.join(UNSTRUCTURED_DIR, "autism_clinical_notes.txt")
OUT_TODDLER = os.path.join(UNSTRUCTURED_DIR, "toddler_social_media_posts.txt")
OUT_DIABETES = os.path.join(UNSTRUCTURED_DIR, "diabetes_clinical_notes.txt")
OUT_STROKE = os.path.join(UNSTRUCTURED_DIR, "stroke_clinical_notes.txt")

random.seed(42)

def banner(title: str, width: int = 72, char: str = "=") -> None:
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)

def sub_step(msg: str) -> None:
    print(f"    ▸  {msg}")

# ===========================================================================
# 1.  AUTISM NOTE GENERATION
# ===========================================================================
def _autism_vals(row, df):
    jaundice_col = "Jauundice" if "Jauundice" in df.columns else "Jaundice"
    vals = {f"A{i}": int(row[f"A{i}"]) for i in range(1, 11)}
    return {
        "age": row["Age"],
        "sex": str(row["Sex"]).strip(),
        "jaundice": str(row[jaundice_col]).strip().lower() == "yes",
        "family_asd": str(row["Family_ASD"]).strip().lower() == "yes",
        "class_val": str(row["Class"]).strip().upper() == "YES",
        "aq_vals": vals,
        "aq_score": sum(vals.values()),
    }

def autism_style_soap(v, rid) -> str:
    doc = random.choice(DOCTORS)
    return f"{random.choice(HEADERS_AUTISM)}\nRecord #{rid:05d} | Clinician: {doc}\n\nS: {age_sex_phrase(v['age'], v['sex'])} brought in for ASD screening.\n   {jaundice_phrase(v['jaundice'])}\n   {family_asd_phrase(v['family_asd'])}\n   {random.choice(NOISE_AUTISM)}\n\nO: {aq10_block(v['aq_vals'])}\n   Clinical threshold: score >= 6.\n\nA: {asd_class_phrase(v['class_val'])}\n\nP: {'Refer for comprehensive diagnostic evaluation.' if v['class_val'] else 'No further evaluation needed. Routine follow-up.'}"

def autism_style_narrative(v, rid) -> str:
    sex_full = "male" if v['sex'].lower() in ('m', 'male') else "female"
    pronoun = "He" if sex_full == "male" else "She"
    child = "boy" if sex_full == "male" else "girl"
    jaund = "has a history of neonatal jaundice" if v['jaundice'] else "has no history of neonatal jaundice"
    fam = "has a positive family history of ASD" if v['family_asd'] else "has no family history of ASD"
    endorsed = [f"A{i}" for i in range(1, 11) if v['aq_vals'][f"A{i}"] == 1]
    not_end = [f"A{i}" for i in range(1, 11) if v['aq_vals'][f"A{i}"] == 0]
    
    if endorsed:
        end_desc = ", ".join(f"{k} ({AQ_DESCRIPTIONS[k]})" for k in endorsed)
        end_text = f"{pronoun} endorsed items: {end_desc}."
    else: end_text = f"{pronoun} did not endorse any AQ-10 items."
    
    ne_text = f"Items not endorsed: {', '.join(not_end)}." if not_end else "All items were endorsed."
    dx = "positive" if v['class_val'] else "negative"
    
    return f"{random.choice(HEADERS_AUTISM)}\nRecord #{rid:05d}\n\nPatient is a {v['age']}-year-old {child} who presents for developmental screening.\n{pronoun} {jaund} and {fam}. {random.choice(NOISE_AUTISM)}\n\nThe AQ-10 screening tool was administered. {end_text} {ne_text}\nTotal score was {v['aq_score']} out of 10 (clinical cutoff >= 6).\n\nImpression: ASD screen {dx}. {'Referral recommended for comprehensive evaluation.' if v['class_val'] else 'No further action at this time.'}"

def autism_style_abbreviated(v, rid) -> str:
    sex_abbr = "M" if v['sex'].lower() in ('m', 'male') else "F"
    aq_compact = " ".join(f"A{i}({'+'if v['aq_vals'][f'A{i}']==1 else '-'})" for i in range(1, 11))
    return f"{random.choice(HEADERS_AUTISM)}\nRecord #{rid:05d}\n\nPt: {v['age']}yo {sex_abbr}\nCC: ASD screening\nHx: Jaundice({'+' if v['jaundice'] else '-'}), Fam ASD({'+' if v['family_asd'] else '-'})\n\nAQ-10: {aq_compact} = {v['aq_score']}/10\nThreshold: >=6\n\nDx: ASD screen {'POS' if v['class_val'] else 'NEG'}\nPlan: {'Refer for full eval' if v['class_val'] else 'No f/u needed'}"

def autism_style_bullet(v, rid) -> str:
    sex_full = "male" if v['sex'].lower() in ('m', 'male') else "female"
    lines = [f"{random.choice(HEADERS_AUTISM)}\nRecord #{rid:05d}\n\n- Patient: {sex_full}, age {v['age']}", f"- Neonatal jaundice: {'Yes' if v['jaundice'] else 'No'}", f"- Family history of ASD: {'Yes' if v['family_asd'] else 'No'}", f"- {random.choice(NOISE_AUTISM)}\n\n- AQ-10 Screening:"]
    for i in range(1, 11): lines.append(f"    A{i}: {'Yes' if v['aq_vals'][f'A{i}'] == 1 else 'No'}")
    lines.append(f"- AQ-10 Score: {v['aq_score']}/10 (threshold >= 6)")
    lines.append(f"- {random.choice(NOISE_AUTISM)}\n\n- {asd_class_phrase(v['class_val'])}")
    return "\n".join(lines)

def autism_style_mixed(v, rid) -> str:
    sex_full = "male" if v['sex'].lower() in ('m', 'male') else "female"
    return f"{random.choice(HEADERS_AUTISM)}\nRecord #{rid:05d} — {random.choice(DOCTORS)}\n\nThis {v['age']}-year-old {sex_full} was referred for autism screening.\n{jaundice_phrase(v['jaundice'])} {family_asd_phrase(v['family_asd'])}\n{random.choice(NOISE_AUTISM)}\n\n--- AQ-10 Data ---\n{aq10_block(v['aq_vals'])}\nReferral threshold: score >= 6\n---\n\n{asd_class_phrase(v['class_val'])}\n{'He' if sex_full=='male' else 'She'} {'will be referred for a comprehensive diagnostic evaluation.' if v['class_val'] else 'does not require further ASD-specific evaluation at this time.'}"

AUTISM_STYLES = [autism_style_soap, autism_style_narrative, autism_style_abbreviated, autism_style_bullet, autism_style_mixed]

def generate_autism_notes(df: pd.DataFrame) -> str:
    notes = []
    for idx, row in df.iterrows():
        rid = idx + 1
        note = random.choice(AUTISM_STYLES)(_autism_vals(row, df), rid)
        if random.random() < 0.05: note = inject_typos(note)
        notes.append(note + f"\n\n--- END OF RECORD #{rid:05d} ---\n")
    return "\n".join(notes)

# ===========================================================================
# 2.  TODDLER AUTISM — SOCIAL MEDIA POSTS
# ===========================================================================
def generate_toddler_notes(df: pd.DataFrame) -> str:
    notes = []
    for idx, row in df.iterrows():
        rid = idx + 1
        note = toddler_social_media_post(row, rid)
        notes.append(note + f"\n\n--- END OF RECORD #{rid:05d} ---\n")
    return "\n".join(notes)

# ===========================================================================
# 3.  DIABETES NOTE GENERATION (MIXED WITH IoT LOGS)
# ===========================================================================
SYMPTOM_COLS = ["Polyuria", "Polydipsia", "sudden weight loss", "weakness", "Polyphagia", "Genital thrush", "visual blurring", "Itching", "Irritability", "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"]

def diabetes_style_formal(row, rid) -> str:
    gender, age = str(row["Gender"]).strip(), row["Age"]
    return f"{random.choice(HEADERS_DIABETES)}\nRecord #{rid:05d}\n\nPATIENT INFORMATION:\n  {age_sex_phrase(age, gender)} patient presented for diabetes risk evaluation.\n  {random.choice(NOISE_DIABETES)}\n\nPRESENTING COMPLAINTS AND SYMPTOM REVIEW:\n{diabetes_symptoms_block(row, SYMPTOM_COLS)}\n\nCLINICAL ASSESSMENT:\n  {diabetes_class_phrase(str(row['class']).strip() == 'Positive')}"

def diabetes_style_abbreviated(row, rid) -> str:
    gender, age = str(row["Gender"]).strip(), row["Age"]
    sym_parts = [f"{SYMPTOM_SYNONYMS[c][0]}({'+' if str(row[c]).strip().lower()=='yes' else '-'})" for c in SYMPTOM_COLS]
    return f"{random.choice(HEADERS_DIABETES)}\nRecord #{rid:05d}\n\nPt: {age}yo {'M' if gender=='Male' else 'F'}\nCC: Diabetes risk screening\n\nROS: {', '.join(sym_parts)}\n\n{diabetes_class_phrase(str(row['class']).strip() == 'Positive')}"

DIABETES_STYLES = [diabetes_style_formal, diabetes_style_abbreviated, lambda r, rid: diabetes_style_iot_log(r, rid, SYMPTOM_COLS)]

def generate_diabetes_notes(df: pd.DataFrame) -> str:
    notes = []
    for idx, row in df.iterrows():
        rid = idx + 1
        note = random.choice(DIABETES_STYLES)(row, rid)
        if random.random() < 0.05: note = inject_typos(note)
        notes.append(note + f"\n\n--- END OF RECORD #{rid:05d} ---\n")
    return "\n".join(notes)

# ===========================================================================
# 4.  STROKE NOTE GENERATION
# ===========================================================================
def stroke_style_formal(row, rid) -> str:
    return f"{random.choice(HEADERS_STROKE)}\nRecord #{rid:05d}  |  Original Patient ID: {row['id']}\n\nPATIENT DEMOGRAPHICS:\n{stroke_demographics_block(row)}\n\nMEDICAL HISTORY:\n{stroke_comorbid_block(row)}\n\nCLINICAL MEASUREMENTS:\n{stroke_measurements_block(row)}\n  {random.choice(NOISE_STROKE)}\n\nCEREBROVASCULAR EVENT STATUS:\n  {stroke_class_phrase(int(row['stroke']) == 1)}"

def stroke_style_abbreviated(row, rid) -> str:
    return f"{random.choice(HEADERS_STROKE)}\nRecord #{rid:05d} | PID: {row['id']}\n\nPt: {row['age']}yo {'M' if str(row['gender']).strip()=='Male' else 'F'}, married: {'Y' if str(row['ever_married']).strip()=='Yes' else 'N'}\nWork: {str(row['work_type']).strip()} | Residence: {str(row['Residence_type']).strip().lower()}\nPMH: HTN({'+'if int(row['hypertension']) else'-'}) Heart dz({'+'if int(row['heart_disease']) else'-'})\nGlucose: {row['avg_glucose_level']} mg/dL | BMI: {row['bmi'] if pd.notna(row['bmi']) else 'N/A'} kg/m² | Smoking: {str(row['smoking_status']).strip()}\n\n{stroke_class_phrase(int(row['stroke']) == 1)}"

STROKE_STYLES = [stroke_style_formal, stroke_style_abbreviated]

def generate_stroke_notes(df: pd.DataFrame) -> str:
    notes = []
    for idx, row in df.iterrows():
        rid = idx + 1
        note = random.choice(STROKE_STYLES)(row, rid)
        if random.random() < 0.05: note = inject_typos(note)
        notes.append(note + f"\n\n--- END OF RECORD #{rid:05d} ---\n")
    return "\n".join(notes)

# ===========================================================================
# 5.  MAIN EXECUTION
# ===========================================================================
def main() -> None:
    banner("PHASE 0 — GENERATING REALISTIC UNSTRUCTURED CLINICAL NOTES")
    banner("STEP 1 — LOADING RAW CSVs", char="-")
    na_vals = ["?", "", " ", "N/A", "n/a", "NA", "na", "null", "NULL", "None"]
    
    df_autism = pd.read_csv(RAW_AUTISM, na_values=na_vals, keep_default_na=True)
    df_toddler = pd.read_csv(RAW_TODDLER, na_values=na_vals, keep_default_na=True)
    df_diabetes = pd.read_csv(RAW_DIABETES, na_values=na_vals, keep_default_na=True)
    df_stroke = pd.read_csv(RAW_STROKE, na_values=na_vals, keep_default_na=True)

    sub_step(f"Loaded autism   — {df_autism.shape[0]:,} records")
    sub_step(f"Loaded toddler  — {df_toddler.shape[0]:,} records")
    sub_step(f"Loaded diabetes — {df_diabetes.shape[0]:,} records")
    sub_step(f"Loaded stroke   — {df_stroke.shape[0]:,} records")

    banner("STEP 2 — GENERATING UNSTRUCTURED TEXT (Clinical/Social/IoT)", char="-")

    sub_step("Generating autism clinical notes...")
    with open(OUT_AUTISM, "w", encoding="utf-8") as f: f.write(generate_autism_notes(df_autism))

    sub_step("Generating toddler social media posts...")
    with open(OUT_TODDLER, "w", encoding="utf-8") as f: f.write(generate_toddler_notes(df_toddler))

    sub_step("Generating diabetes mixed notes (Clinical + IoT)...")
    with open(OUT_DIABETES, "w", encoding="utf-8") as f: f.write(generate_diabetes_notes(df_diabetes))

    sub_step("Generating stroke clinical notes...")
    with open(OUT_STROKE, "w", encoding="utf-8") as f: f.write(generate_stroke_notes(df_stroke))

    banner("GENERATION COMPLETE", char="-")
    print(f"  Total clinical records generated: {len(df_autism) + len(df_toddler) + len(df_diabetes) + len(df_stroke):,}")
    print(f"  Output directory: {UNSTRUCTURED_DIR}\n")

if __name__ == "__main__":
    main()
