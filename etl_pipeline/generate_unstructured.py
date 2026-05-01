# =============================================================================
# ETL Pipeline — Phase 0: Generate Unstructured Clinical Notes
# Project : Early Disease Prediction Using Healthcare Data Warehouse
# Script  : etl_pipeline/generate_unstructured.py
#
# PURPOSE:
#   Convert the three structured CSV datasets into realistic free-text
#   clinical notes that simulate real-world unstructured Electronic Health
#   Record (EHR) data.  This demonstrates the Bronze Layer of the Medallion
#   Architecture where data arrives as raw, unstructured text.
#
# OUTPUT:
#   datasets/unstructured/
#     ├── autism_clinical_notes.txt
#     ├── diabetes_clinical_notes.txt
#     └── stroke_clinical_notes.txt
#
# HOW TO RUN:
#   python etl_pipeline/generate_unstructured.py
# =============================================================================

import os
import random
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# 0.  Path Resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
UNSTRUCTURED_DIR = os.path.join(ROOT_DIR, "datasets", "unstructured")

os.makedirs(UNSTRUCTURED_DIR, exist_ok=True)

# Raw CSV paths
RAW_AUTISM = os.path.join(DATASETS_DIR, "Autism_Screening_Data_Combined.csv")
RAW_DIABETES = os.path.join(DATASETS_DIR, "diabetes_data_upload.csv")
RAW_STROKE = os.path.join(DATASETS_DIR, "healthcare-dataset-stroke-data.csv")

# Output paths
OUT_AUTISM = os.path.join(UNSTRUCTURED_DIR, "autism_clinical_notes.txt")
OUT_DIABETES = os.path.join(UNSTRUCTURED_DIR, "diabetes_clinical_notes.txt")
OUT_STROKE = os.path.join(UNSTRUCTURED_DIR, "stroke_clinical_notes.txt")


# ---------------------------------------------------------------------------
# 1.  Utility Helpers
# ---------------------------------------------------------------------------

def banner(title: str, width: int = 72, char: str = "=") -> None:
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def sub_step(message: str) -> None:
    print(f"    ▸  {message}")


# Seed for reproducibility
random.seed(42)

# Natural language templates for variation
AUTISM_TEMPLATES = [
    "AUTISM SCREENING — CLINICAL ASSESSMENT NOTE",
    "BEHAVIORAL SCREENING REPORT — AQ-10 EVALUATION",
    "DEVELOPMENTAL SCREENING DOCUMENTATION",
]

DIABETES_TEMPLATES = [
    "DIABETES RISK ASSESSMENT — CLINICAL INTAKE NOTE",
    "ENDOCRINOLOGY CONSULTATION — SYMPTOM EVALUATION",
    "PRIMARY CARE ASSESSMENT — METABOLIC SCREENING",
]

STROKE_TEMPLATES = [
    "CEREBROVASCULAR RISK ASSESSMENT — DISCHARGE SUMMARY",
    "NEUROLOGY DEPARTMENT — STROKE RISK EVALUATION",
    "CLINICAL CASE SUMMARY — CARDIOVASCULAR PROFILE",
]

AQ_ITEM_DESCRIPTIONS = {
    "A1": "I often notice small sounds when others do not",
    "A2": "I usually concentrate more on the whole picture rather than the small details",
    "A3": "I find it easy to do more than one thing at once",
    "A4": "If there is an interruption I can switch back to what I was doing very quickly",
    "A5": "I find it easy to read between the lines when someone is talking to me",
    "A6": "I know how to tell if someone listening to me is getting bored",
    "A7": "When I'm reading a story I find it difficult to work out the characters' intentions",
    "A8": "I like to collect information about categories of things",
    "A9": "I find it easy to work out what someone is thinking or feeling just by looking at their face",
    "A10": "I find it difficult to work out people's intentions",
}

SYMPTOM_DESCRIPTIONS = {
    "Polyuria": "excessive urination (polyuria)",
    "Polydipsia": "excessive thirst (polydipsia)",
    "sudden weight loss": "sudden unexplained weight loss",
    "weakness": "generalized weakness and fatigue",
    "Polyphagia": "excessive hunger (polyphagia)",
    "Genital thrush": "recurrent genital thrush infections",
    "visual blurring": "episodes of visual blurring",
    "Itching": "persistent itching of the skin",
    "Irritability": "notable irritability and mood changes",
    "delayed healing": "delayed wound healing",
    "partial paresis": "partial paresis of the extremities",
    "muscle stiffness": "muscle stiffness and joint pain",
    "Alopecia": "hair loss (alopecia)",
    "Obesity": "clinical obesity",
}


# ---------------------------------------------------------------------------
# 2.  Generate Autism Clinical Notes
# ---------------------------------------------------------------------------

def generate_autism_notes(df: pd.DataFrame) -> str:
    """Generate unstructured clinical notes from the autism screening CSV."""
    notes = []
    for idx, row in df.iterrows():
        record_id = idx + 1  # type: ignore[operator]
        template = random.choice(AUTISM_TEMPLATES)
        sex_str = "male" if str(row["Sex"]).lower() in ("m", "male") else "female"
        age = row["Age"]

        # Build the AQ-10 item narrative
        aq_items = []
        for i in range(1, 11):
            col = f"A{i}"
            val = int(row[col])
            desc = AQ_ITEM_DESCRIPTIONS[col]
            response = "endorsed (YES)" if val == 1 else "not endorsed (NO)"
            aq_items.append(f"      Item {col}: \"{desc}\" — {response}")

        aq_score = sum(int(row[f"A{i}"]) for i in range(1, 11))

        jaundice_col = "Jauundice" if "Jauundice" in df.columns else "Jaundice"
        jaundice = str(row[jaundice_col]).strip().lower()
        jaundice_text = (
            "History of neonatal jaundice is noted."
            if jaundice == "yes"
            else "No history of neonatal jaundice."
        )

        family_asd = str(row["Family_ASD"]).strip().lower()
        family_text = (
            "Positive family history of Autism Spectrum Disorder (ASD) is reported."
            if family_asd == "yes"
            else "No family history of Autism Spectrum Disorder (ASD)."
        )

        class_val = str(row["Class"]).strip().upper()
        diagnosis = (
            "ASD POSITIVE — referral recommended for comprehensive diagnostic evaluation"
            if class_val == "YES"
            else "ASD NEGATIVE — no further ASD-specific evaluation indicated at this time"
        )

        note = f"""{'=' * 70}
{template}
Record #{record_id:05d}
{'=' * 70}

PATIENT DEMOGRAPHICS:
  The patient is a {age}-year-old {sex_str} presenting for routine
  developmental and behavioral screening.

CLINICAL HISTORY:
  {jaundice_text}
  {family_text}

AQ-10 BEHAVIORAL SCREENING QUESTIONNAIRE:
  The Autism-Spectrum Quotient (AQ-10) screening tool was administered.
  Individual item responses are as follows:

{chr(10).join(aq_items)}

  TOTAL AQ-10 SCORE: {aq_score} out of 10
  Clinical referral threshold: score >= 6

SCREENING OUTCOME:
  Classification: {class_val}
  Assessment: {diagnosis}.

--- END OF RECORD #{record_id:05d} ---
"""
        notes.append(note)

    return "\n".join(notes)


# ---------------------------------------------------------------------------
# 3.  Generate Diabetes Clinical Notes
# ---------------------------------------------------------------------------

def generate_diabetes_notes(df: pd.DataFrame) -> str:
    """Generate unstructured clinical notes from the diabetes risk CSV."""
    notes = []
    for idx, row in df.iterrows():
        record_id = idx + 1  # type: ignore[operator]
        template = random.choice(DIABETES_TEMPLATES)
        gender = str(row["Gender"]).strip()
        age = row["Age"]
        pronoun = "He" if gender == "Male" else "She"

        # Build symptom narrative
        present_symptoms = []
        absent_symptoms = []
        for symptom_col, description in SYMPTOM_DESCRIPTIONS.items():
            val = str(row[symptom_col]).strip().lower()
            if val == "yes":
                present_symptoms.append(description)
            else:
                absent_symptoms.append(description)

        if present_symptoms:
            present_text = (
                f"  {pronoun} reports the following symptoms:\n"
                + "\n".join(f"    - {s}" for s in present_symptoms)
            )
        else:
            present_text = f"  {pronoun} denies all screened symptoms."

        if absent_symptoms:
            absent_text = (
                f"  {pronoun} denies:\n"
                + "\n".join(f"    - {s}" for s in absent_symptoms)
            )
        else:
            absent_text = ""

        class_val = str(row["class"]).strip()
        diagnosis = (
            "POSITIVE — Patient meets criteria for early-stage diabetes risk. "
            "Further laboratory workup including fasting blood glucose and HbA1c "
            "is strongly recommended."
            if class_val == "Positive"
            else "NEGATIVE — Patient does not currently meet criteria for diabetes "
            "risk based on symptom screening. Routine follow-up advised."
        )

        note = f"""{'=' * 70}
{template}
Record #{record_id:05d}
{'=' * 70}

PATIENT INFORMATION:
  A {age}-year-old {gender.lower()} patient presented for diabetes risk
  evaluation at the outpatient endocrinology clinic.

PRESENTING COMPLAINTS AND SYMPTOM REVIEW:
{present_text}

{absent_text}

  Total symptoms reported: {len(present_symptoms)} out of {len(SYMPTOM_DESCRIPTIONS)} screened.

CLINICAL ASSESSMENT:
  Based on the symptom profile and clinical evaluation, the patient's
  diabetes risk classification is determined as follows:

  Risk Classification: {class_val}
  {diagnosis}

--- END OF RECORD #{record_id:05d} ---
"""
        notes.append(note)

    return "\n".join(notes)


# ---------------------------------------------------------------------------
# 4.  Generate Stroke Clinical Notes
# ---------------------------------------------------------------------------

def generate_stroke_notes(df: pd.DataFrame) -> str:
    """Generate unstructured clinical notes from the stroke prediction CSV."""
    notes = []
    for idx, row in df.iterrows():
        record_id = idx + 1  # type: ignore[operator]
        template = random.choice(STROKE_TEMPLATES)
        patient_id = row["id"]
        gender = str(row["gender"]).strip()
        age = row["age"]
        pronoun = "He" if gender == "Male" else "She"

        # Marriage status
        married = str(row["ever_married"]).strip()
        married_text = (
            f"{pronoun} is currently or has been previously married."
            if married == "Yes"
            else f"{pronoun} has never been married."
        )

        # Work type
        work = str(row["work_type"]).strip()
        work_text = f"Employment status: {work} sector."

        # Residence
        residence = str(row["Residence_type"]).strip()
        residence_text = f"The patient resides in an {residence.lower()} area."

        # Comorbidities
        hypertension = int(row["hypertension"])
        heart_disease = int(row["heart_disease"])
        comorbid_parts = []
        if hypertension == 1:
            comorbid_parts.append("documented hypertension")
        if heart_disease == 1:
            comorbid_parts.append("history of heart disease")
        if not comorbid_parts:
            comorbid_text = "  No significant comorbidities (no hypertension, no heart disease)."
        else:
            comorbid_text = f"  Significant comorbidities include: {', '.join(comorbid_parts)}."

        # Clinical measurements
        glucose = row["avg_glucose_level"]
        bmi = row["bmi"]
        bmi_text = (
            f"BMI: {bmi} kg/m²" if pd.notna(bmi) else "BMI: Not recorded (missing value)"
        )

        # Smoking
        smoking = str(row["smoking_status"]).strip()
        smoking_text = f"Smoking history: {smoking}."

        # Stroke outcome
        stroke = int(row["stroke"])
        stroke_text = (
            "STROKE OCCURRED — The patient experienced a cerebrovascular event. "
            "Immediate neurological intervention was initiated."
            if stroke == 1
            else "NO STROKE — No cerebrovascular event recorded during the "
            "observation period."
        )

        note = f"""{'=' * 70}
{template}
Record #{record_id:05d}  |  Original Patient ID: {patient_id}
{'=' * 70}

PATIENT DEMOGRAPHICS:
  {gender} patient, {age} years of age.
  {married_text}
  {work_text}
  {residence_text}

MEDICAL HISTORY:
{comorbid_text}

CLINICAL MEASUREMENTS:
  Average glucose level: {glucose} mg/dL
  {bmi_text}
  {smoking_text}

CEREBROVASCULAR EVENT STATUS:
  Outcome: {"Stroke" if stroke == 1 else "No Stroke"}
  {stroke_text}

--- END OF RECORD #{record_id:05d} ---
"""
        notes.append(note)

    return "\n".join(notes)


# ===========================================================================
# 5.  MAIN EXECUTION
# ===========================================================================

def main() -> None:
    banner("PHASE 0 — GENERATING UNSTRUCTURED CLINICAL NOTES")

    # ── Load raw CSVs ────────────────────────────────────────────────────────
    banner("STEP 1 — LOADING RAW CSVs", char="-")

    na_values = ["?", "", " ", "N/A", "n/a", "NA", "na", "null", "NULL", "None"]

    try:
        df_autism = pd.read_csv(RAW_AUTISM, na_values=na_values, keep_default_na=True)
        df_diabetes = pd.read_csv(RAW_DIABETES, na_values=na_values, keep_default_na=True)
        df_stroke = pd.read_csv(RAW_STROKE, na_values=na_values, keep_default_na=True)
    except FileNotFoundError as exc:
        print(f"\n  [ERROR] Could not find a raw dataset file.\n  {exc}")
        sys.exit(1)

    sub_step(f"Loaded autism   — {df_autism.shape[0]:,} records")
    sub_step(f"Loaded diabetes — {df_diabetes.shape[0]:,} records")
    sub_step(f"Loaded stroke   — {df_stroke.shape[0]:,} records")

    # ── Generate unstructured text ───────────────────────────────────────────
    banner("STEP 2 — GENERATING UNSTRUCTURED TEXT", char="-")

    sub_step("Generating autism clinical notes...")
    autism_text = generate_autism_notes(df_autism)
    with open(OUT_AUTISM, "w", encoding="utf-8") as f:
        f.write(autism_text)
    size_kb = os.path.getsize(OUT_AUTISM) / 1024
    sub_step(f"  Saved: {OUT_AUTISM} ({size_kb:.1f} KB)")

    sub_step("Generating diabetes clinical notes...")
    diabetes_text = generate_diabetes_notes(df_diabetes)
    with open(OUT_DIABETES, "w", encoding="utf-8") as f:
        f.write(diabetes_text)
    size_kb = os.path.getsize(OUT_DIABETES) / 1024
    sub_step(f"  Saved: {OUT_DIABETES} ({size_kb:.1f} KB)")

    sub_step("Generating stroke clinical notes...")
    stroke_text = generate_stroke_notes(df_stroke)
    with open(OUT_STROKE, "w", encoding="utf-8") as f:
        f.write(stroke_text)
    size_kb = os.path.getsize(OUT_STROKE) / 1024
    sub_step(f"  Saved: {OUT_STROKE} ({size_kb:.1f} KB)")

    # ── Summary ──────────────────────────────────────────────────────────────
    banner("GENERATION COMPLETE — SUMMARY", char="-")

    print(f"\n  {'File':<40} {'Records':>8} {'Size (KB)':>10}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 10}")
    for name, path, count in [
        ("autism_clinical_notes.txt", OUT_AUTISM, len(df_autism)),
        ("diabetes_clinical_notes.txt", OUT_DIABETES, len(df_diabetes)),
        ("stroke_clinical_notes.txt", OUT_STROKE, len(df_stroke)),
    ]:
        size = os.path.getsize(path) / 1024
        print(f"  {name:<40} {count:>8,} {size:>10.1f}")

    total_records = len(df_autism) + len(df_diabetes) + len(df_stroke)
    print(f"\n  Total clinical records generated: {total_records:,}")
    print(f"  Output directory: {UNSTRUCTURED_DIR}")
    print()
    banner("Next Step: etl_pipeline/parse_unstructured.py", char="-")
    print("  Parse the unstructured text back into structured DataFrames")
    print()


if __name__ == "__main__":
    main()
