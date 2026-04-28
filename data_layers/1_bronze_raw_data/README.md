# 🥉 Bronze Data Layer — Raw Data (Immutable Source of Truth)

## Purpose
The Bronze Layer contains the **original, unmodified** CSV files exactly as
downloaded from Kaggle. No cleaning, no encoding, no imputation has been
applied. This layer serves as the **immutable audit trail** — we can always
trace any downstream result back to its raw origin.

## Script
- **`load_and_inspect.py`** — Loads all three CSVs and produces a data quality
  audit report (shapes, dtypes, missing values). **Does NOT modify any file.**

## Files

| File | Source | Rows | Columns | Missing Values |
|---|---|---|---|---|
| `Autism_Screening_Data_Combined.csv` | Kaggle (Thabtah, 2017) | 6,075 | 15 | 0 |
| `diabetes_data_upload.csv` | Kaggle (Islam et al., 2020) | 520 | 17 | 0 |
| `healthcare-dataset-stroke-data.csv` | Kaggle (fedesoriano, 2021) | 5,110 | 12 | **201** (BMI column) |

## Data Quality Issues Detected

### Autism Screening
- Column name typo: `Jauundice` (double 'u') instead of `Jaundice`
- `Sex` column uses inconsistent encoding: `m`/`f` instead of `Male`/`Female`
- Binary features stored as text strings (`yes`/`no`) instead of 0/1
- Target `Class` stored as text (`YES`/`NO`)

### Diabetes Risk
- Target column named `class` (lowercase) — inconsistent with other datasets
- All 14 symptom columns stored as text strings (`Yes`/`No`)
- No missing values detected

### Stroke Prediction
- Contains meaningless surrogate `id` column (no predictive value)
- 1 row with `gender = 'Other'` — too sparse for reliable encoding
- **201 missing BMI values (3.93%)** — requires imputation in Silver Layer
- `work_type` has 5 categories, `smoking_status` has 4 categories (multi-level)

## Key Principle
> **Bronze data is NEVER modified.** All transformations happen in the
> Silver Layer. This ensures full data lineage and reproducibility.
