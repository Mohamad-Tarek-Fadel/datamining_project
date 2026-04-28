# Data Layers — Medallion Architecture (Bronze → Silver → Gold)

This directory organises the project's data pipeline into the three layers
of the **Medallion Data Architecture**, a widely-adopted pattern in modern
data engineering (Databricks, Azure Synapse, AWS Lake Formation).

```
data_layers/
├── 1_bronze_raw_data/        ← Raw CSV files — immutable source of truth
├── 2_silver_cleaned_data/    ← Cleaned, validated, standardised CSVs
└── 3_gold_warehouse/         ← Star Schema SQLite Data Warehouse (OLAP-ready)
```

## Layer Transition Rules

| Transition | Script | Key Operations |
|---|---|---|
| **Source → Bronze** | Manual download from Kaggle | No transformation |
| **Bronze → Silver** | `etl_pipeline/clean.py` | Whitespace stripping, typo fixes, type encoding, grouped-median BMI imputation, feature engineering (AQ_Score) |
| **Silver → Gold** | `etl_pipeline/warehouse.py` | Star Schema DDL, parameterized INSERT, FK constraints, indexes, analytical views |

## Data Lineage

```
Kaggle Sources (3 CSV files)
    │
    ▼
🥉 BRONZE (datasets/)
    │  load_and_inspect.py — audit only, no modification
    ▼
🥈 SILVER (datasets/cleaned/)
    │  clean.py — 201 BMI values imputed, 1 row dropped, 30+ columns encoded
    ▼
🥇 GOLD (warehouse/health_warehouse.db)
       warehouse.py — 4 tables, 8 indexes, 4 views, 11,704 patient records
```

> **Note:** The files in each layer folder are **copies** of the originals.
> The master copies remain in their original locations (`datasets/`, `datasets/cleaned/`, `warehouse/`)
> to keep the project pipeline functional.
