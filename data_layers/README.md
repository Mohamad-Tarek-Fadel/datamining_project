# Data Layers — Medallion Architecture (Bronze → Silver → Gold)

This directory organises the project's data pipeline into the three layers
of the **Medallion Data Architecture**, a widely-adopted pattern in modern
data engineering (Databricks, Azure Synapse, AWS Lake Formation).

```
data_layers/
├── 1_bronze_raw_data/        ← Unstructured clinical text notes (raw source of truth)
├── 2_silver_cleaned_data/    ← NLP-extracted, cleaned, validated, standardised CSVs
└── 3_gold_warehouse/         ← Star Schema SQLite Data Warehouse (OLAP-ready)
```

## Layer Transition Rules

| Transition | Script | Key Operations |
|---|---|---|
| **Source → Bronze** | `etl_pipeline/generate_unstructured.py` | Convert structured CSVs → free-text clinical notes (.txt) |
| **Bronze → Silver** | `etl_pipeline/parse_unstructured.py` | NLP/Regex extraction → structured DataFrames; BMI imputation; encoding |
| **Silver → Gold** | `etl_pipeline/warehouse.py` | Star Schema DDL, parameterized INSERT, FK constraints, indexes, views |

## Data Lineage

```
Raw CSVs (Kaggle Sources)
    │
    ▼  generate_unstructured.py — convert to free-text clinical notes
🥉 BRONZE (datasets/unstructured/)
    │  3 unstructured .txt files (11.4 MB + 702 KB + 4.0 MB)
    │
    ▼  parse_unstructured.py — NLP/Regex extraction + cleaning
🥈 SILVER (datasets/cleaned/)
    │  3 structured CSVs, 0 missing values, all encoded
    │
    ▼  warehouse.py — Star Schema ETL
🥇 GOLD (warehouse/health_warehouse.db)
       4 tables, 8 indexes, 4 views, 11,704 patient records
```

## Unstructured Data Approach

The Bronze Layer demonstrates a real-world healthcare data scenario where
clinical information arrives as **unstructured free-text notes** (e.g.,
doctor's notes, discharge summaries, intake forms) rather than clean CSVs.

The Silver Layer applies **NLP and text mining techniques** to extract
structured data from the unstructured text:
- Regular expression pattern matching
- Medical terminology detection
- Clinical entity extraction
- Data type inference and conversion

> **Note:** The files in each layer folder are **copies** of the originals.
> The master copies remain in their original locations (`datasets/`, `datasets/cleaned/`, `warehouse/`)
> to keep the project pipeline functional.
