# 🥈 Silver Data Layer — NLP-Extracted, Cleaned & Standardised Data

## Purpose
The Silver Layer transforms **unstructured clinical text** from the Bronze
Layer into **clean, structured DataFrames** using NLP and text mining
techniques. This is the core data engineering step that demonstrates
extraction from real-world unstructured healthcare data.

## Script
- **`parse_unstructured.py`** — NLP/Regex extraction pipeline that reads
  free-text clinical notes and extracts structured tabular data.

## NLP Extraction Techniques Used

| Technique | Application |
|---|---|
| **Regex Pattern Matching** | Extract age, gender, clinical measurements |
| **Medical Terminology Detection** | Identify symptoms (polyuria, polydipsia, etc.) |
| **Clinical Entity Extraction** | Parse AQ-10 item responses, comorbidities |
| **Context-Aware Classification** | Distinguish "reports" vs "denies" symptoms |
| **Data Type Inference** | Convert extracted text to int8/float32 |

## Output Files

| File | Rows | Columns | Missing | Target Positive % |
|---|---|---|---|---|
| `autism_cleaned.csv` | 6,075 | 16 (+1 engineered) | **0** | 29.7% |
| `diabetes_cleaned.csv` | 520 | 17 | **0** | 61.5% |
| `stroke_cleaned.csv` | 5,109 (-1 dropped) | 11 (-1 dropped) | **0** | 4.87% |

## Extraction Pipeline (Bronze → Silver)

### Autism Screening
| Step | Operation | NLP Technique |
|---|---|---|
| 1 | Extract age from "X-year-old" pattern | Regex: `(\d+)-year-old` |
| 2 | Extract sex from demographic sentence | Regex: `\d+-year-old\s+(male\|female)` |
| 3 | Extract AQ-10 items from questionnaire | Regex: `Item A\d+:.*endorsed\|not endorsed` |
| 4 | Calculate AQ_Score (sum of A1..A10) | Derived feature |
| 5 | Extract jaundice history | Text search: "history of neonatal jaundice" |
| 6 | Extract family ASD history | Text search: "family history of ASD" |
| 7 | Extract classification | Regex: `Classification:\s*(YES\|NO)` |

### Diabetes Risk
| Step | Operation | NLP Technique |
|---|---|---|
| 1 | Extract age and gender | Regex pattern matching |
| 2 | Parse "reports" symptom section | Context-aware section detection |
| 3 | Parse "denies" symptom section | Context-aware section detection |
| 4 | Match 14 medical symptoms | Medical terminology regex |
| 5 | Extract risk classification | Regex: `Risk Classification:\s*(Positive\|Negative)` |

### Stroke Prediction
| Step | Operation | NLP Technique |
|---|---|---|
| 1 | Extract demographics (gender, age) | Regex matching |
| 2 | Extract comorbidities (hypertension, heart disease) | Clinical entity detection |
| 3 | Extract clinical measurements (glucose, BMI) | Numeric extraction regex |
| 4 | Handle missing BMI ("Not recorded") | Missing value detection |
| 5 | **BMI Imputation: grouped-median by age bracket** | Same strategy as original |
| 6 | Drop gender="Other" row (1 record) | Data quality filtering |
| 7 | Extract stroke outcome | Regex: `Outcome:\s*Stroke\|No Stroke` |

## Validation Gates
1. **`assert_no_missing()`** — Raises `ValueError` if ANY NaN remains
2. **Record count verification** — Must match source record count
3. **Data type enforcement** — int8 for binary, float32 for continuous
