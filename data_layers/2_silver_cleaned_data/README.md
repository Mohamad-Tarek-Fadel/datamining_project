# 🥈 Silver Data Layer — Cleaned, Validated & Standardised Data

## Purpose
The Silver Layer contains **cleaned and standardised** versions of the Bronze
data. All data quality issues identified during inspection have been resolved
here. The output CSVs are ready for loading into the Gold Layer (Warehouse)
and for downstream analytics.

## Script
- **`clean.py`** — Full ETL cleaning pipeline with validation assertions.
  Every transformation is logged to stdout for auditability.

## Output Files

| File | Rows | Columns | Missing | Target Positive % |
|---|---|---|---|---|
| `autism_cleaned.csv` | 6,075 | 16 (+1 engineered) | **0** | 29.7% |
| `diabetes_cleaned.csv` | 520 | 17 | **0** | 61.5% |
| `stroke_cleaned.csv` | 5,109 (-1 dropped) | 11 (-1 dropped) | **0** | 4.87% |

## Transformations Applied

### Autism Screening (Bronze → Silver)
| Step | Transformation | Justification |
|---|---|---|
| 1 | Strip whitespace from all string columns | Data hygiene |
| 2 | Rename `Jauundice` → `Jaundice` | Fix source typo |
| 3 | Normalise `Sex`: `m`→`Male`, `f`→`Female`, then encode Male=1, Female=0 | Standardise + numeric encoding |
| 4 | Encode `Jaundice`, `Family_ASD`: yes→1, no→0 | Binary encoding |
| 5 | Encode target `Class`: YES→1, NO→0 | Binary encoding |
| 6 | Verify AQ-10 items (A1–A10) are already 0/1 | Integrity check |
| 7 | **Engineer `AQ_Score` = sum(A1..A10)** | Clinical AQ-10 total score (range 0–10) |
| 8 | Enforce compact dtypes (int8/int16) | Memory optimisation |
| 9 | **Assert zero missing values** | Hard validation gate |

### Diabetes Risk (Bronze → Silver)
| Step | Transformation | Justification |
|---|---|---|
| 1 | Strip whitespace from all string columns | Data hygiene |
| 2 | Encode `Gender`: Male→1, Female→0 | Binary encoding |
| 3 | Encode 14 symptom columns: Yes→1, No→0 | Binary encoding |
| 4 | Encode target `class`: Positive→1, Negative→0 | Binary encoding |
| 5 | Rename `class` → `Class` | Cross-dataset naming consistency |
| 6 | Enforce compact dtypes (int8/int16) | Memory optimisation |
| 7 | **Assert zero missing values** | Hard validation gate |

### Stroke Prediction (Bronze → Silver)
| Step | Transformation | Justification |
|---|---|---|
| 1 | Strip whitespace from all string columns | Data hygiene |
| 2 | Drop `id` column | No predictive value (surrogate key) |
| 3 | Remove 1 row where gender='Other' | Too sparse for encoding (1 out of 5,110) |
| 4 | **BMI Imputation: grouped-median by age bracket** | See details below |
| 5 | Encode `gender`: Male→1, Female→0 | Binary encoding |
| 6 | Encode `ever_married`: Yes→1, No→0 | Binary encoding |
| 7 | Encode `Residence_type`: Urban→1, Rural→0 | Binary encoding |
| 8 | Keep `work_type` & `smoking_status` as strings | Multi-category → OHE deferred to Feature Engineering |
| 9 | Rename `stroke` → `Class` | Cross-dataset naming consistency |
| 10 | Enforce dtypes (int8/float32) | Memory optimisation |
| 11 | **Assert zero missing values** | Hard validation gate |

## BMI Imputation Strategy (Critical Decision)

**Problem:** 201 missing BMI values (3.93% of stroke dataset).

**Why NOT global median?**
- Global median BMI ≈ 28.9
- BMI varies **systematically** with age — using a single value would
  introduce age-dependent bias

**Solution: Grouped-Median Imputation by Age Bracket**

| Age Bracket | Median BMI | Missing Count | Effect of Global Median |
|---|---|---|---|
| 0–18 | **20.1** | 21 | Would over-impute by +8.8 kg/m² |
| 19–40 | 28.0 | 39 | Would over-impute by +0.9 |
| 41–60 | 30.3 | 54 | Would under-impute by −1.4 |
| 61–80 | 29.3 | 86 | Would under-impute by −0.4 |
| 81+ | 27.5 | 1 | Would over-impute by +1.4 |

> Using global median would **systematically over-impute children's BMI by
> nearly 9 kg/m²**, which corrupts the most age-sensitive feature. The
> grouped strategy preserves the natural BMI-age relationship.

## Validation Gates
Every dataset passes through these hard assertions before saving:
1. **`assert_no_missing()`** — Raises `ValueError` if ANY NaN remains
2. **`assert_binary_column()`** — Confirms encoded columns contain only {0, 1}
3. **`print_change_summary()`** — Logs before/after shape and missing count

## Class Imbalance Summary (Silver Layer)

| Dataset | Positive | Negative | Ratio | Severity | ML Strategy |
|---|---|---|---|---|---|
| Autism | 1,804 (29.7%) | 4,271 (70.3%) | 1:2.4 | Mild | `class_weight='balanced'` |
| Diabetes | 320 (61.5%) | 200 (38.5%) | 1:0.6 | Mild | `class_weight='balanced'` |
| Stroke | 249 (4.87%) | 4,860 (95.13%) | **1:19.5** | **Severe** | **SMOTE** |
