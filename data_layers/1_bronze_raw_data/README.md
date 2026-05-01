# 🥉 Bronze Data Layer — Unstructured Clinical Notes (Raw Source of Truth)

## Purpose
The Bronze Layer contains **unstructured free-text clinical notes** that
simulate real-world Electronic Health Record (EHR) data. This layer
demonstrates how healthcare data typically arrives in practice — as
doctor's notes, discharge summaries, and intake forms — NOT as clean CSVs.

## Script
- **`generate_unstructured.py`** — Converts the 3 structured CSV datasets into
  realistic free-text clinical notes with natural language variation.

## Files

| File | Records | Size | Description |
|---|---|---|---|
| `autism_clinical_notes.txt` | 6,075 | 11.4 MB | AQ-10 screening assessment notes |
| `diabetes_clinical_notes.txt` | 520 | 702 KB | Endocrinology consultation notes |
| `stroke_clinical_notes.txt` | 5,110 | 4.0 MB | Cerebrovascular discharge summaries |

## Sample Unstructured Record (Autism)

```text
======================================================================
AUTISM SCREENING — CLINICAL ASSESSMENT NOTE
Record #00001
======================================================================

PATIENT DEMOGRAPHICS:
  The patient is a 15-year-old male presenting for routine
  developmental and behavioral screening.

CLINICAL HISTORY:
  No history of neonatal jaundice.
  No family history of Autism Spectrum Disorder (ASD).

AQ-10 BEHAVIORAL SCREENING QUESTIONNAIRE:
  Item A1: "I often notice small sounds when others do not" — endorsed (YES)
  Item A2: "I usually concentrate more on the whole picture..." — endorsed (YES)
  ...
  TOTAL AQ-10 SCORE: 5 out of 10

SCREENING OUTCOME:
  Classification: NO
  Assessment: ASD NEGATIVE
```

## Data Quality Characteristics (Unstructured)
- Free-text format with natural language variation
- 3 different note templates per dataset (random selection)
- Clinical terminology embedded in narrative text
- Missing BMI values represented as "Not recorded (missing value)"
- Gender "Other" preserved in stroke notes (handled during extraction)

## Key Principle
> **Bronze data preserves the raw, unstructured nature of clinical text.**
> All NLP extraction and structuring happens in the Silver Layer.
> This demonstrates the full data engineering lifecycle from raw text → analytics.
