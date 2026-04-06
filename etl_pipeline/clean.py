# =============================================================================
# ETL Pipeline — Phase 1A: Data Cleaning & Standardisation
# Project : Early Disease Prediction Using Healthcare Data Warehouse
# Script  : etl_pipeline/clean.py
#
# Datasets cleaned:
#   1. Autism_Screening_Data_Combined.csv  →  autism_cleaned.csv
#   2. diabetes_data_upload.csv            →  diabetes_cleaned.csv
#   3. healthcare-dataset-stroke-data.csv  →  stroke_cleaned.csv
#
# Output  : datasets/cleaned/
#
# Pylance fixes applied:
#   • f-string without placeholders removed (L90 equivalent)
#   • Series.map(dict) replaced with Series.replace(dict) for type safety
#   • assert_binary_column() call sites use cast(pd.Series, ...) to resolve
#     the Series|DataFrame union that Pylance infers for df[variable_col]
#   • groupby().transform() result wrapped in pd.Series() to resolve NDArray
#   • rename(inplace=True) replaced with assignment form to fix overload error
#   • assert_no_missing() call sites use cast(pd.DataFrame, ...) where needed
#   • Whitespace stripping done column-by-column (avoids DataFrame apply ambiguity)
# =============================================================================

import os
import sys
from typing import cast

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0.  Path resolution — works regardless of working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
CLEANED_DIR = os.path.join(ROOT_DIR, "datasets", "cleaned")

# Ensure output directory exists
os.makedirs(CLEANED_DIR, exist_ok=True)

# Raw file paths
RAW_AUTISM = os.path.join(DATASETS_DIR, "Autism_Screening_Data_Combined.csv")
RAW_DIABETES = os.path.join(DATASETS_DIR, "diabetes_data_upload.csv")
RAW_STROKE = os.path.join(DATASETS_DIR, "healthcare-dataset-stroke-data.csv")

# Cleaned output paths
OUT_AUTISM = os.path.join(CLEANED_DIR, "autism_cleaned.csv")
OUT_DIABETES = os.path.join(CLEANED_DIR, "diabetes_cleaned.csv")
OUT_STROKE = os.path.join(CLEANED_DIR, "stroke_cleaned.csv")


# ---------------------------------------------------------------------------
# 1.  Utility helpers
# ---------------------------------------------------------------------------


def banner(title: str, width: int = 72, char: str = "=") -> None:
    """Print a prominent section banner to stdout."""
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def sub_step(message: str) -> None:
    """Print an indented sub-step message."""
    print(f"    \u25b8  {message}")


def print_change_summary(
    label: str,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    missing_before: int,
    missing_after: int,
) -> None:
    """Print a before/after comparison table for one dataset."""
    print(f"\n  {'Metric':<30} {'Before':>12} {'After':>12}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12}")
    print(f"  {'Rows':<30} {before_shape[0]:>12,} {after_shape[0]:>12,}")
    print(f"  {'Columns':<30} {before_shape[1]:>12,} {after_shape[1]:>12,}")
    print(f"  {'Total Missing Cells':<30} {missing_before:>12,} {missing_after:>12,}")

    row_delta = after_shape[0] - before_shape[0]
    col_delta = after_shape[1] - before_shape[1]
    if row_delta != 0:
        print(f"\n  \u26a0  Row delta : {row_delta:+,}  (rows removed/added)")
    if col_delta != 0:
        print(f"  \u26a0  Col delta : {col_delta:+,}  (columns added/removed)")


def assert_no_missing(df: pd.DataFrame, dataset_name: str) -> None:
    """Hard assertion — raise if any NaN remains after cleaning."""
    remaining = int(df.isnull().sum().sum())
    if remaining > 0:
        bad_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"[{dataset_name}] Cleaning incomplete — "
            f"{remaining} NaN(s) remain in: {bad_cols}"
        )
    # Plain string — no f-prefix needed (fix: was flagged as f-string without placeholders)
    print("  \u2714  Zero missing values confirmed.")


def assert_binary_column(series: pd.Series, col: str, dataset: str) -> None:
    """Confirm a column contains only 0 and 1 after encoding."""
    unique_vals = set(series.dropna().unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"[{dataset}] Column '{col}' should be binary {{0,1}} "
            f"but contains: {unique_vals}"
        )


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from every object column in-place.
    Iterating column-by-column avoids the DataFrame.apply() return-type
    ambiguity that Pylance cannot resolve statically.
    """
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# ---------------------------------------------------------------------------
# 2.  Load raw datasets
# ---------------------------------------------------------------------------
banner("STEP 1 — LOADING RAW DATASETS")

UNIVERSAL_NA = ["N/A", "n/a", "NA", "na", "null", "NULL", "None", "?", "", " "]

try:
    df_autism = pd.read_csv(RAW_AUTISM, na_values=UNIVERSAL_NA, keep_default_na=True)
    df_diabetes = pd.read_csv(
        RAW_DIABETES, na_values=UNIVERSAL_NA, keep_default_na=True
    )
    df_stroke = pd.read_csv(RAW_STROKE, na_values=UNIVERSAL_NA, keep_default_na=True)
except FileNotFoundError as exc:
    print(f"\n  [ERROR] Could not find a raw dataset file.\n  {exc}")
    sys.exit(1)

print(
    f"  [OK]  autism   \u2014 {df_autism.shape[0]:,} rows \u00d7 {df_autism.shape[1]} cols"
)
print(
    f"  [OK]  diabetes \u2014 {df_diabetes.shape[0]:,} rows \u00d7 {df_diabetes.shape[1]} cols"
)
print(
    f"  [OK]  stroke   \u2014 {df_stroke.shape[0]:,} rows \u00d7 {df_stroke.shape[1]} cols"
)


# ===========================================================================
# 3.  AUTISM SCREENING — Cleaning
# ===========================================================================
banner("STEP 2 — CLEANING: AUTISM SCREENING")

df_a: pd.DataFrame = df_autism.copy()
before_shape_a = df_a.shape
before_missing_a = int(df_a.isnull().sum().sum())

# ── 3.1  Strip whitespace from all string columns ──────────────────────────
df_a = strip_string_columns(df_a)
sub_step("Stripped leading/trailing whitespace from all object columns.")

# ── 3.2  Fix the 'Jauundice' column-name typo ──────────────────────────────
if "Jauundice" in df_a.columns:
    df_a = df_a.rename(columns={"Jauundice": "Jaundice"})
    sub_step("Renamed column  'Jauundice'  \u2192  'Jaundice'.")
else:
    sub_step("Column 'Jaundice' already correctly named \u2014 no rename needed.")

# ── 3.3  Standardise 'Sex' values (m/f → Male/Female) then encode ──────────
# Fix: use .replace(dict) instead of .map(dict) — identical semantics for
# exhaustive mappings and resolves the Pylance arg-type error on map().
df_a["Sex"] = df_a["Sex"].str.lower().replace({"m": "Male", "f": "Female"})

unexpected_sex = int(df_a["Sex"].isnull().sum())
if unexpected_sex > 0:
    print(
        f"  \u26a0  Warning: {unexpected_sex} unexpected value(s) in 'Sex' \u2014 "
        "they will be NaN and may need manual review."
    )
sub_step(
    f"Normalised 'Sex': m/f \u2192 Male/Female  "
    f"(unique after: {sorted(df_a['Sex'].dropna().unique())})"
)

# Encode Sex: Male → 1, Female → 0
df_a["Sex"] = df_a["Sex"].replace({"Male": 1, "Female": 0})
# cast: df_a["Sex"] is always a Series; cast() narrows the type for Pylance
assert_binary_column(cast(pd.Series, df_a["Sex"]), "Sex", "Autism")
sub_step("Encoded  'Sex'  \u2192  Male=1, Female=0.")

# ── 3.4  Encode binary yes/no columns → 1/0 (case-insensitive) ─────────────
BINARY_COLS_A = ["Jaundice", "Family_ASD"]

for _col in BINARY_COLS_A:
    df_a[_col] = df_a[_col].str.lower().replace({"yes": 1, "no": 0})
    # cast resolves Series|DataFrame union Pylance infers for df[variable]
    assert_binary_column(cast(pd.Series, df_a[_col]), _col, "Autism")
    sub_step(f"Encoded  '{_col}'  \u2192  yes=1, no=0.")

# ── 3.5  Encode target column 'Class' (YES/NO → 1/0) ───────────────────────
df_a["Class"] = df_a["Class"].str.upper().replace({"YES": 1, "NO": 0})
assert_binary_column(cast(pd.Series, df_a["Class"]), "Class", "Autism")
sub_step("Encoded  target 'Class'  \u2192  YES=1, NO=0.")

# ── 3.6  Verify AQ-10 items are already integer 0/1 ────────────────────────
aq_cols = [f"A{i}" for i in range(1, 11)]
for _col in aq_cols:
    assert_binary_column(cast(pd.Series, df_a[_col]), _col, "Autism")
sub_step(
    f"Verified AQ-10 columns {aq_cols[0]}\u2013{aq_cols[-1]} are already 0/1 integers."
)

# ── 3.7  Engineer AQ Total Score (standard clinical sum) ───────────────────
df_a["AQ_Score"] = df_a[aq_cols].sum(axis=1)
sub_step(
    f"Engineered feature 'AQ_Score' = sum(A1..A10)  "
    f"(range: {int(df_a['AQ_Score'].min())}\u2013{int(df_a['AQ_Score'].max())})."
)

# ── 3.8  Enforce correct dtypes ─────────────────────────────────────────────
int8_cols_a = aq_cols + ["Jaundice", "Family_ASD", "Class", "Sex", "AQ_Score"]
df_a[int8_cols_a] = df_a[int8_cols_a].astype(np.int8)
df_a["Age"] = df_a["Age"].astype(np.int16)
sub_step("Enforced compact dtypes  (int8 for binary cols, int16 for Age).")

# ── 3.9  Final validation ────────────────────────────────────────────────────
assert_no_missing(cast(pd.DataFrame, df_a), "Autism")
print_change_summary(
    "Autism",
    before_shape_a,
    df_a.shape,
    before_missing_a,
    int(df_a.isnull().sum().sum()),
)

print(f"\n  Final column order : {df_a.columns.tolist()}")
print(
    f"  Target distribution:\n"
    f"    ASD Positive (1) : {int((df_a['Class'] == 1).sum()):,}  "
    f"({(df_a['Class'] == 1).mean() * 100:.1f}%)\n"
    f"    ASD Negative (0) : {int((df_a['Class'] == 0).sum()):,}  "
    f"({(df_a['Class'] == 0).mean() * 100:.1f}%)"
)


# ===========================================================================
# 4.  DIABETES RISK — Cleaning
# ===========================================================================
banner("STEP 3 — CLEANING: DIABETES RISK")

df_d: pd.DataFrame = df_diabetes.copy()
before_shape_d = df_d.shape
before_missing_d = int(df_d.isnull().sum().sum())

# ── 4.1  Strip whitespace ───────────────────────────────────────────────────
df_d = strip_string_columns(df_d)
sub_step("Stripped leading/trailing whitespace from all object columns.")

# ── 4.2  Encode 'Gender' → Male=1, Female=0 ────────────────────────────────
df_d["Gender"] = df_d["Gender"].replace({"Male": 1, "Female": 0})
assert_binary_column(cast(pd.Series, df_d["Gender"]), "Gender", "Diabetes")
sub_step("Encoded  'Gender'  \u2192  Male=1, Female=0.")

# ── 4.3  Encode all 14 symptom Yes/No columns → 1/0 ───────────────────────
SYMPTOM_COLS = [
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]

for _col in SYMPTOM_COLS:
    df_d[_col] = df_d[_col].replace({"Yes": 1, "No": 0})
    assert_binary_column(cast(pd.Series, df_d[_col]), _col, "Diabetes")

sub_step(f"Encoded {len(SYMPTOM_COLS)} symptom columns  \u2192  Yes=1, No=0.")
sub_step(f"  Columns: {SYMPTOM_COLS}")

# ── 4.4  Encode target 'class' → Positive=1, Negative=0 ────────────────────
df_d["class"] = df_d["class"].replace({"Positive": 1, "Negative": 0})
assert_binary_column(cast(pd.Series, df_d["class"]), "class", "Diabetes")
sub_step("Encoded  target 'class'  \u2192  Positive=1, Negative=0.")

# ── 4.5  Rename 'class' → 'Class' for consistency ───────────────────────────
# Fix: assignment form instead of inplace=True — resolves Pylance overload error
df_d = df_d.rename(columns={"class": "Class"})
sub_step("Renamed  'class'  \u2192  'Class'  (consistent naming across all datasets).")

# ── 4.6  Enforce compact dtypes ─────────────────────────────────────────────
binary_cols_d: list[str] = ["Gender"] + SYMPTOM_COLS + ["Class"]
df_d[binary_cols_d] = df_d[binary_cols_d].astype(np.int8)
df_d["Age"] = df_d["Age"].astype(np.int16)
sub_step("Enforced compact dtypes  (int8 for binary cols, int16 for Age).")

# ── 4.7  Final validation ────────────────────────────────────────────────────
assert_no_missing(cast(pd.DataFrame, df_d), "Diabetes")
print_change_summary(
    "Diabetes",
    before_shape_d,
    df_d.shape,
    before_missing_d,
    int(df_d.isnull().sum().sum()),
)

print(
    f"\n  Target distribution:\n"
    f"    Diabetic    (1) : {int((df_d['Class'] == 1).sum()):,}  "
    f"({(df_d['Class'] == 1).mean() * 100:.1f}%)\n"
    f"    No Diabetes (0) : {int((df_d['Class'] == 0).sum()):,}  "
    f"({(df_d['Class'] == 0).mean() * 100:.1f}%)"
)


# ===========================================================================
# 5.  STROKE PREDICTION — Cleaning
# ===========================================================================
banner("STEP 4 — CLEANING: STROKE PREDICTION")

df_s: pd.DataFrame = df_stroke.copy()
before_shape_s = df_s.shape
before_missing_s = int(df_s.isnull().sum().sum())

# ── 5.1  Strip whitespace ───────────────────────────────────────────────────
df_s = cast(pd.DataFrame, strip_string_columns(df_s))
sub_step("Stripped leading/trailing whitespace from all object columns.")

# ── 5.2  Drop the 'id' column (meaningless surrogate key) ──────────────────
df_s = cast(pd.DataFrame, df_s.drop(columns=["id"]))
sub_step("Dropped column  'id'  (surrogate key \u2014 no predictive value).")

# ── 5.3  Drop the single row where gender == 'Other' ───────────────────────
other_mask = df_s["gender"].str.lower() == "other"
other_count = int(other_mask.sum())
df_s = cast(pd.DataFrame, df_s.loc[~cast(pd.Series, other_mask)]).reset_index(drop=True)  # type: ignore[assignment]
sub_step(
    f"Removed {other_count} row(s) where gender == 'Other'  "
    f"(too sparse for encoding \u2014 avoids dummy trap)."
)

# ── 5.4  BMI IMPUTATION — median per age bracket (smart grouped strategy) ──
#
#   Why grouped median instead of global median?
#   BMI changes systematically with age:
#     0-18  →  median BMI ≈ 20.1  (children/adolescents — leaner)
#     19-40 →  median BMI ≈ 28.0
#     41-60 →  median BMI ≈ 30.3
#     61-80 →  median BMI ≈ 29.3
#     81+   →  median BMI ≈ 27.5
#   Using the global median (≈28.9) would over-impute children and
#   under-impute middle-aged adults, introducing systematic bias.

print()
sub_step("BMI Imputation \u2014 strategy: median per age bracket")

AGE_BINS: list[float] = [0.0, 18.0, 40.0, 60.0, 80.0, float("inf")]
AGE_LABELS: list[str] = ["0-18", "19-40", "41-60", "61-80", "81+"]

df_s["age_bracket"] = pd.cut(
    cast(pd.Series, df_s["age"]),
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=True,
)

missing_bmi_before = int(cast(pd.Series, df_s["bmi"]).isnull().sum())

# Per-bracket diagnostic table
bmi_series: pd.Series = cast(pd.Series, df_s["bmi"])
bracket_medians: pd.Series = cast(  # type: ignore[assignment]
    pd.Series,
    df_s.groupby("age_bracket", observed=True)["bmi"].median(),
)
missing_mask: pd.Series = cast(pd.Series, bmi_series.isnull())
missing_rows: pd.DataFrame = cast(pd.DataFrame, df_s.loc[missing_mask])  # type: ignore[assignment]
bracket_missing: pd.Series = cast(  # type: ignore[assignment]
    pd.Series,
    missing_rows.groupby("age_bracket", observed=True).size(),
)

print(f"\n      {'Bracket':<10} {'BMI Median':>12} {'Missing Count':>14}")
print(f"      {'-' * 10} {'-' * 12} {'-' * 14}")
for _bracket in AGE_LABELS:
    _med: float = (
        float(bracket_medians.loc[_bracket])  # type: ignore[arg-type]
        if _bracket in bracket_medians.index
        else float("nan")
    )
    _miss: int = (
        int(bracket_missing.loc[_bracket]) if _bracket in bracket_missing.index else 0
    )
    print(f"      {_bracket:<10} {_med:>12.1f} {_miss:>14,}")

# Perform the grouped median imputation
# Fix: wrap transform() result in pd.Series() — resolves NDArray type error
df_s["bmi"] = pd.Series(
    df_s.groupby("age_bracket", observed=True)["bmi"].transform(
        lambda grp: grp.fillna(grp.median())
    ),
    index=df_s.index,
)

missing_bmi_after = int(cast(pd.Series, df_s["bmi"]).isnull().sum())
sub_step(
    f"BMI imputed: {missing_bmi_before} missing  \u2192  {missing_bmi_after} missing."
)

# Fallback: global median for any edge bracket that is entirely NaN
if missing_bmi_after > 0:
    global_median = float(cast(pd.Series, df_s["bmi"]).median())
    df_s["bmi"] = cast(pd.Series, df_s["bmi"]).fillna(global_median)
    sub_step(
        f"  Fallback: {missing_bmi_after} remaining NaN(s) filled with "
        f"global BMI median ({global_median:.1f})."
    )

# Drop the temporary helper column
df_s = cast(pd.DataFrame, df_s.drop(columns=["age_bracket"]))
sub_step("Removed temporary 'age_bracket' helper column.")

# -- 5.5  Encode binary string columns -> 1/0 --------------------------------
#
#   gender         : Male/Female -> 1/0
#   ever_married   : Yes/No      -> 1/0
#   Residence_type : Urban/Rural -> 1/0
#
#   NOTE: work_type (5 categories) and smoking_status (4 categories)
#         are left as strings and will be one-hot encoded in Phase 3
#         (etl_pipeline/feature_engineering.py) to keep concerns separated.

df_s["gender"] = df_s["gender"].replace({"Male": 1, "Female": 0})
df_s["ever_married"] = df_s["ever_married"].replace({"Yes": 1, "No": 0})
df_s["Residence_type"] = df_s["Residence_type"].replace({"Urban": 1, "Rural": 0})

assert_binary_column(cast(pd.Series, df_s["gender"]), "gender", "Stroke")
assert_binary_column(cast(pd.Series, df_s["ever_married"]), "ever_married", "Stroke")
assert_binary_column(
    cast(pd.Series, df_s["Residence_type"]), "Residence_type", "Stroke"
)

sub_step("Encoded  'gender'         \u2192  Male=1, Female=0.")
sub_step("Encoded  'ever_married'   \u2192  Yes=1, No=0.")
sub_step("Encoded  'Residence_type' \u2192  Urban=1, Rural=0.")
sub_step(
    "Left     'work_type' & 'smoking_status' as strings "
    "(multi-category \u2192 Phase 3 one-hot encoding)."
)

# ── 5.6  Rename 'stroke' → 'Class' for consistency ─────────────────────────
# Fix: assignment form instead of inplace=True — resolves Pylance overload error
df_s = cast(pd.DataFrame, df_s.rename(columns={"stroke": "Class"}))
sub_step("Renamed  'stroke'  \u2192  'Class'  (consistent target column name).")

# ── 5.7  Enforce dtypes ─────────────────────────────────────────────────────
int8_cols_s = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "Residence_type",
    "Class",
]
for _col in int8_cols_s:
    df_s[_col] = cast(pd.Series, df_s[_col]).astype(np.int8)

df_s["age"] = cast(pd.Series, df_s["age"]).astype(np.float32)
df_s["avg_glucose_level"] = cast(pd.Series, df_s["avg_glucose_level"]).astype(
    np.float32
)
df_s["bmi"] = cast(pd.Series, df_s["bmi"]).astype(np.float32)
sub_step("Enforced dtypes  (int8 for binary, float32 for continuous).")

# ── 5.8  Final validation ────────────────────────────────────────────────────
assert_no_missing(cast(pd.DataFrame, df_s), "Stroke")
print_change_summary(
    "Stroke",
    before_shape_s,
    df_s.shape,
    before_missing_s,
    int(df_s.isnull().sum().sum()),
)

n_stroke = int((df_s["Class"] == 1).sum())
n_no_stroke = int((df_s["Class"] == 0).sum())
ratio = n_no_stroke / n_stroke if n_stroke > 0 else float("inf")
print(
    f"\n  Target distribution:\n"
    f"    Stroke     (1) : {n_stroke:,}  "
    f"({(df_s['Class'] == 1).mean() * 100:.2f}%)\n"
    f"    No Stroke  (0) : {n_no_stroke:,}  "
    f"({(df_s['Class'] == 0).mean() * 100:.2f}%)\n"
    f"    \u26a0  Imbalance ratio  \u2248  {ratio:.1f}:1  "
    f"(severe \u2014 SMOTE required in Phase 3)"
)


# ===========================================================================
# 6.  SAVE CLEANED DATASETS
# ===========================================================================
banner("STEP 5 — SAVING CLEANED DATASETS")

SAVE_MAP: dict[str, tuple[pd.DataFrame, str]] = {
    OUT_AUTISM: (df_a, "autism_cleaned.csv"),
    OUT_DIABETES: (df_d, "diabetes_cleaned.csv"),
    OUT_STROKE: (df_s, "stroke_cleaned.csv"),
}

for _filepath, (_df, _filename) in SAVE_MAP.items():
    _df.to_csv(_filepath, index=False)
    size_kb = os.path.getsize(_filepath) / 1024
    print(
        f"  [SAVED]  {_filename:<30}  "
        f"{_df.shape[0]:>6,} rows \u00d7 {_df.shape[1]:>2} cols  "
        f"({size_kb:.1f} KB)  \u2192  {_filepath}"
    )


# ===========================================================================
# 7.  FINAL CROSS-DATASET SUMMARY
# ===========================================================================
banner("CLEANING COMPLETE — FINAL SUMMARY", char="-")

summary_data: dict[str, list] = {
    "Dataset": ["Autism Screening", "Diabetes Risk", "Stroke Prediction"],
    "Rows": [df_a.shape[0], df_d.shape[0], df_s.shape[0]],
    "Cols": [df_a.shape[1], df_d.shape[1], df_s.shape[1]],
    "Missing Cells": [
        int(df_a.isnull().sum().sum()),
        int(df_d.isnull().sum().sum()),
        int(df_s.isnull().sum().sum()),
    ],
    "Target=1 (%)": [
        f"{(df_a['Class'] == 1).mean() * 100:.1f}%",
        f"{(df_d['Class'] == 1).mean() * 100:.1f}%",
        f"{(df_s['Class'] == 1).mean() * 100:.2f}%",
    ],
    "Output File": [
        "autism_cleaned.csv",
        "diabetes_cleaned.csv",
        "stroke_cleaned.csv",
    ],
}

df_summary = pd.DataFrame(summary_data)
print()
print(df_summary.to_string(index=False))
print()

banner(
    "Next Step: etl_pipeline/warehouse.py  \u2192  Load cleaned CSVs into SQLite Star Schema",
    char="-",
)
print()
