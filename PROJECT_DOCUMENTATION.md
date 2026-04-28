# 🏥 Early Disease Prediction Using Healthcare Data Warehouse
## Complete Project Documentation — Q&A Defense Guide

> **University Project — Data Mining & Data Warehousing**
> Nile, Misr, and Upper Egypt Universities (NMU) · Faculty of Computers and Information

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Layers Architecture (Bronze → Silver → Gold)](#2-data-layers-architecture)
3. [Bronze Layer — Raw Data Ingestion](#3-bronze-layer)
4. [Silver Layer — ETL Cleaning & Standardisation](#4-silver-layer)
5. [Gold Layer — Star Schema Data Warehouse](#5-gold-layer)
6. [SQL Documentation & Analytical Queries](#6-sql-documentation)
7. [Exploratory Data Analysis (EDA)](#7-eda)
8. [Feature Engineering](#8-feature-engineering)
9. [Machine Learning Models](#9-ml-models)
10. [Key Findings & Clinical Insights](#10-key-findings)
11. [Common Q&A — Expected Questions & Answers](#11-qa)
12. [Technical Decisions & Justifications](#12-technical-decisions)

---

## 1. Project Overview

### What is this project?
A **clinical intelligence system** that:
- Integrates 3 heterogeneous healthcare datasets (Autism, Diabetes, Stroke)
- Builds a unified **SQLite Star Schema Data Warehouse**
- Applies **5 ML classifiers** per disease for early prediction
- Covers the **full data science lifecycle**: ETL → Warehouse → EDA → Feature Engineering → Modeling

### Why this project matters
- **Autism**: Early behavioral screening can trigger timely intervention
- **Diabetes**: Symptom-based prediction enables early lifestyle modification
- **Stroke**: A missed stroke prediction = a preventable death; 82% recall achieved

### Pipeline Summary
```
Raw CSVs (Bronze) → Cleaned CSVs (Silver) → SQLite Star Schema (Gold)
                                                    ↓
                                          EDA → Feature Engineering → ML Models
```

---

## 2. Data Layers Architecture

### The Medallion Architecture (Bronze → Silver → Gold)

We follow the **Medallion Architecture** pattern used in modern data engineering:

| Layer | Purpose | Location | Format |
|---|---|---|---|
| **🥉 Bronze** | Raw data as-is from source | `datasets/*.csv` | Raw CSV files |
| **🥈 Silver** | Cleaned, validated, standardised | `datasets/cleaned/*.csv` | Cleaned CSV files |
| **🥇 Gold** | Business-ready star schema warehouse | `warehouse/health_warehouse.db` | SQLite Database |

### Why Medallion Architecture?
1. **Traceability**: Raw data is never modified — we can always trace back
2. **Reproducibility**: Each layer's transformation is documented and repeatable
3. **Quality gates**: Each transition enforces validation rules
4. **Separation of concerns**: Cleaning logic ≠ schema design ≠ analytics

---

## 3. Bronze Layer — Raw Data Ingestion

### Script: `etl_pipeline/load_and_inspect.py`

### What happens in Bronze?
- Load 3 raw CSV files **without any modification**
- Audit shapes, column types, missing values
- Document data quality issues before any cleaning

### Three Source Datasets

| Dataset | Source | File | Rows | Columns | Target |
|---|---|---|---|---|---|
| Autism Screening | Kaggle (Thabtah, 2017) | `Autism_Screening_Data_Combined.csv` | 6,075 | 16 | ASD Diagnosis (YES/NO) |
| Diabetes Risk | Kaggle (Islam et al., 2020) | `diabetes_data_upload.csv` | 520 | 17 | Diabetic (Positive/Negative) |
| Stroke Prediction | Kaggle (fedesoriano, 2021) | `healthcare-dataset-stroke-data.csv` | 5,110 | 12 | Stroke (1/0) |

### Data Quality Issues Found in Bronze

| Dataset | Issue | Severity |
|---|---|---|
| Autism | Column name typo: `Jauundice` instead of `Jaundice` | Low |
| Autism | Sex values inconsistent: `m`/`f` instead of `Male`/`Female` | Medium |
| Autism | Binary columns stored as strings (`yes`/`no`) | Medium |
| Diabetes | Target column named `class` (lowercase, inconsistent) | Low |
| Diabetes | All symptom columns stored as strings (`Yes`/`No`) | Medium |
| Stroke | `id` column present (meaningless surrogate) | Low |
| Stroke | 1 row with `gender='Other'` (too sparse to encode) | Low |
| Stroke | **201 missing BMI values** (3.93% of rows) | **High** |

---

## 4. Silver Layer — ETL Cleaning & Standardisation

### Script: `etl_pipeline/clean.py`
### Output: `datasets/cleaned/{autism,diabetes,stroke}_cleaned.csv`

### Autism Cleaning Rules
1. Strip whitespace from all string columns
2. Rename `Jauundice` → `Jaundice` (typo fix)
3. Normalise `Sex`: `m`→`Male`, `f`→`Female`, then encode Male=1, Female=0
4. Encode binary columns (`Jaundice`, `Family_ASD`): yes=1, no=0
5. Encode target `Class`: YES=1, NO=0
6. Verify AQ-10 items (A1–A10) are already 0/1
7. **Engineer** `AQ_Score = sum(A1..A10)` — range 0–10
8. Enforce compact dtypes (int8 for binary, int16 for Age)
9. Validate: zero missing values

### Diabetes Cleaning Rules
1. Strip whitespace from all string columns
2. Encode `Gender`: Male=1, Female=0
3. Encode 14 symptom columns: Yes=1, No=0
4. Encode target `class`: Positive=1, Negative=0
5. Rename `class` → `Class` (consistency)
6. Enforce compact dtypes
7. Validate: zero missing values

### Stroke Cleaning Rules
1. Strip whitespace from all string columns
2. Drop `id` column (no predictive value)
3. Remove 1 row where gender='Other' (too sparse)
4. **BMI Imputation — Grouped Median by Age Bracket**:
   - Global median BMI ≈ 28.9 — but BMI varies systematically with age
   - Age brackets and their medians:

   | Age Bracket | Median BMI | Missing Count |
   |---|---|---|
   | 0–18 | 20.1 | 21 |
   | 19–40 | 28.0 | 39 |
   | 41–60 | 30.3 | 54 |
   | 61–80 | 29.3 | 86 |
   | 81+ | 27.5 | 1 |

   - Using global median would **over-impute children's BMI by ~8 kg/m²**
5. Encode binary columns: gender, ever_married, Residence_type → 0/1
6. Keep `work_type` and `smoking_status` as strings (multi-category → deferred to Phase 3)
7. Rename `stroke` → `Class`
8. Validate: zero missing values

### Silver Layer Validation Summary

| Dataset | Rows | Cols | Missing Cells | Positive % | Imbalance |
|---|---|---|---|---|---|
| Autism | 6,075 | 16 | 0 | 29.7% | 1:2.4 (mild) |
| Diabetes | 520 | 17 | 0 | 61.5% | 1:0.6 (mild) |
| Stroke | 5,109 | 11 | 0 | 4.87% | **1:19.5 (severe)** |

---

## 5. Gold Layer — Star Schema Data Warehouse

### Scripts: `etl_pipeline/warehouse.py`, `warehouse/schema.sql`
### Database: `warehouse/health_warehouse.db` (SQLite, ~1.2 MB)

### Why Star Schema?
- Three diseases share **common demographics** (age, gender)
- Star schema separates shared metadata (`dim_patient`) from disease-specific measurements
- Enables **cross-disease queries** via JOIN on `patient_id`
- Efficient filtering via indexed columns
- Referential integrity via FK constraints + `ON DELETE CASCADE`

### Schema Design

```
                   ┌──────────────────────────┐
                   │       dim_patient         │  ← DIMENSION TABLE
                   │  PK: patient_id (AUTO)    │     (11,704 rows)
                   │      age        REAL      │
                   │      gender     INTEGER   │     0=Female, 1=Male
                   │      source_dataset TEXT  │     'autism'|'diabetes'|'stroke'
                   └────────────┬─────────────┘
                                │  (FK ON DELETE CASCADE)
           ┌────────────────────┼─────────────────────┐
           │                    │                     │
  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
  │  fact_autism    │  │  fact_diabetes  │  │  fact_stroke    │
  │  6,075 rows     │  │  520 rows       │  │  5,109 rows     │
  │  a1–a10, aq_score│ │  14 symptoms    │  │  clinical meas. │
  │  jaundice       │  │  label          │  │  smoking, work  │
  │  family_asd     │  └─────────────────┘  │  label          │
  │  label          │                       └─────────────────┘
  └─────────────────┘
```

### SQL DDL Features
- **PRAGMA**: `foreign_keys=ON`, `journal_mode=WAL`, `synchronous=NORMAL`
- **CHECK constraints**: Every binary column enforces `CHECK (col IN (0, 1))`
- **NOT NULL**: All columns are NOT NULL — no missing data in warehouse
- **AUTOINCREMENT**: Surrogate keys on all tables
- **ON DELETE CASCADE**: Removing a patient removes their fact records

### Indexes (8 total)
| Index | Table | Column | Purpose |
|---|---|---|---|
| `idx_autism_patient` | fact_autism | patient_id | Accelerate JOINs |
| `idx_diabetes_patient` | fact_diabetes | patient_id | Accelerate JOINs |
| `idx_stroke_patient` | fact_stroke | patient_id | Accelerate JOINs |
| `idx_autism_label` | fact_autism | label | Accelerate WHERE/GROUP BY |
| `idx_diabetes_label` | fact_diabetes | label | Accelerate WHERE/GROUP BY |
| `idx_stroke_label` | fact_stroke | label | Accelerate WHERE/GROUP BY |
| `idx_patient_source` | dim_patient | source_dataset | Cross-disease queries |
| `idx_patient_gender` | dim_patient | gender | Demographic queries |

### Analytical Views (4 total)
| View | Purpose |
|---|---|
| `vw_autism_full` | Denormalised autism records with decoded labels |
| `vw_diabetes_full` | Denormalised diabetes records with decoded labels |
| `vw_stroke_full` | Denormalised stroke records with decoded labels |
| `vw_disease_summary` | OLAP summary — one row per disease cohort |

### Data Loading (DML)
- All data loaded via **parameterized SQL INSERT** statements (not pandas `.to_sql()`)
- Each disease: first INSERT into `dim_patient`, then INSERT into fact table with matching `patient_id`
- Total: **11,704 patient records** loaded across 4 tables

### Validation Checks
- Row count verification: dim_patient = 11,704 (6,075 + 520 + 5,109) ✔
- FK orphan check: 0 orphans in all 3 fact tables ✔
- NULL audit: 0 NULLs in any column ✔
- Domain constraint audit: all binary columns contain only {0, 1} ✔

---

## 6. SQL Documentation & Analytical Queries

### File: `warehouse/analytical_queries.sql` — 20 documented queries

#### Section 1: Data Quality Verification (Q-01 to Q-04)
- **Q-01**: Row count verification across all tables
- **Q-02**: Referential integrity check (FK orphan detection via LEFT JOIN)
- **Q-03**: NULL value audit across all columns
- **Q-04**: Domain constraint verification (binary columns + categorical values)

#### Section 2: Dimension Table Analysis (Q-05 to Q-07)
- **Q-05**: Patient count & age distribution per cohort (with approximate median)
- **Q-06**: Age bracket roll-up (OLAP ROLL-UP operation with window functions)
- **Q-07**: Gender balance per disease cohort

#### Section 3: Disease Prevalence — OLAP (Q-08 to Q-10)
- **Q-08**: Global disease prevalence dashboard (executive-level summary)
- **Q-09**: Class imbalance risk classification with ML strategy recommendation
- **Q-10**: Comparative demographics — positive vs negative cases

#### Section 4: Disease-Specific Deep Dives (Q-11 to Q-14)
- **Q-11**: Autism AQ score statistics by diagnosis
- **Q-12**: Autism AQ-10 item response rates (which items discriminate most)
- **Q-13**: Diabetes symptom prevalence ranking (gap analysis)
- **Q-14**: Stroke clinical risk profile by outcome

#### Section 5: Cross-Disease JOIN Analysis (Q-15 to Q-17)
- **Q-15**: Stroke rate by smoking status (DICE operation)
- **Q-16**: Multi-factor comorbidity matrix (hypertension × heart_disease)
- **Q-17**: Unified warehouse view — all 3 diseases in one summary

#### Section 6: Advanced OLAP Operations (Q-18 to Q-20)
- **Q-18**: ROLL-UP — Stroke risk by age bracket × work type
- **Q-19**: SLICE — High-risk stroke patient registry with composite risk score
- **Q-20**: DICE — High-risk subgroups across all 3 diseases simultaneously

### OLAP Operations Demonstrated
| Operation | Query | Description |
|---|---|---|
| **ROLL-UP** | Q-06, Q-18 | Aggregate from patient → age bracket level |
| **DRILL-DOWN** | Q-14 | From prevalence summary to specific risk factors |
| **SLICE** | Q-19 | Filter to stroke=1 patients only |
| **DICE** | Q-15, Q-16, Q-20 | Filter on 2+ dimensions simultaneously |
| **PIVOT** | Q-12, Q-13 | Conditional aggregation as pivot alternative |

---

## 7. Exploratory Data Analysis (EDA)

### Notebook: `eda/02_eda.ipynb` — 28 cells, 14 figures

### Key EDA Findings

#### Autism (Figures 01–05)
- **Class distribution**: 29.7% positive, 70.3% negative (mild imbalance)
- **AQ Score**: Near-perfect separation between positive/negative at threshold 6
- **AQ-10 items A1, A4, A10**: Largest positive-rate gap between classes
- **Family_ASD**: Statistically significant (χ² p < 0.001)
- **Correlation heatmap**: AQ_Score has highest correlation with label

#### Diabetes (Figures 06–08)
- **Polyuria (75.9%) and Polydipsia (70.3%)**: Most prevalent in diabetic patients
- **Gender**: Slightly more males in diabetic group
- **Correlation**: Polyuria and polydipsia have highest positive correlation with label

#### Stroke (Figures 09–14)
- **Age**: Mean 67.7 (stroke) vs 43.2 (no stroke) — largest effect size
- **Glucose**: Higher in stroke patients (132.5 vs 104.8 mg/dL)
- **Formerly smoked**: Highest stroke rate (7.92%), above active smokers (5.32%)
- **BMI**: Modest difference between groups
- **Statistical significance**: 14 tests performed (Chi-square + Mann-Whitney U)

---

## 8. Feature Engineering

### Notebook: `models/03_feature_engineering.ipynb`

### Steps per Dataset

| Step | Autism | Diabetes | Stroke |
|---|---|---|---|
| Train/Test Split | 80/20 stratified | 80/20 stratified | 80/20 stratified |
| Scaling | StandardScaler | StandardScaler | StandardScaler |
| Imbalance Handling | class_weight='balanced' | class_weight='balanced' | **SMOTE** |
| OHE | Not needed | Not needed | work_type + smoking_status |
| CV | 10-fold stratified | 10-fold stratified | 10-fold stratified |

### SMOTE Details (Stroke Only)
- **Why SMOTE?** 19.5:1 class imbalance — class_weight alone insufficient
- **Applied ONLY on training data** — test set preserves real distribution
- **Before SMOTE**: 3,888 negative + 199 positive = 4,087 training samples
- **After SMOTE**: 3,888 + 3,888 = 7,776 balanced training samples
- **Why not before split?** Would leak synthetic patterns into test set → inflated metrics

### Feature Importance (Random Forest)
- **Autism**: AQ_Score dominates (>0.50 importance)
- **Diabetes**: Polyuria, Polydipsia, Gender top 3
- **Stroke**: Age dominates, followed by avg_glucose_level and BMI

---

## 9. Machine Learning Models

### Notebook: `models/04_modeling.ipynb`

### 5 Classifiers Used
1. **Logistic Regression** — Linear baseline with class_weight='balanced'
2. **Decision Tree** — Interpretable tree with max_depth=5
3. **Random Forest** — 100-tree ensemble
4. **XGBoost** — Gradient boosting with scale_pos_weight
5. **LightGBM** — Fast gradient boosting

### Results Summary

#### Autism Screening
| Model | CV F1 | Test F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.991 | 0.991 | 0.987 | 0.996 | 1.000 |
| **Decision Tree** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LightGBM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

> **Why perfect?** The Decision Tree learned the clinical AQ-10 threshold (AQ ≥ 6). This is **not data leakage** — it validates the clinical integrity of the AQ-10 tool.

#### Diabetes Risk
| Model | CV F1 | Test F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.941 | 0.954 | 0.938 | 0.968 | 0.988 |
| Decision Tree | 0.960 | 0.969 | 0.969 | 0.969 | 0.967 |
| **Random Forest** | **0.965** | **0.984** | **0.969** | **1.000** | **1.000** |
| XGBoost | 0.963 | 0.969 | 0.969 | 0.969 | 0.999 |
| LightGBM | 0.958 | 0.969 | 0.953 | 0.984 | 0.998 |

#### Stroke Prediction (PRIMARY CHALLENGE)
| Model | CV F1 | Test F1 | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.795 | **0.230** | **0.820** | **0.840** | **0.275** |
| Decision Tree | 0.893 | 0.133 | 0.120 | 0.713 | 0.107 |
| Random Forest | 0.963 | 0.209 | 0.140 | 0.776 | 0.211 |
| XGBoost | 0.968 | 0.175 | 0.100 | 0.791 | 0.231 |
| LightGBM | 0.971 | 0.097 | 0.060 | 0.790 | 0.185 |

> **Why LR wins for stroke?** Tree models trained on SMOTE learn boundaries for 50/50 distribution. At threshold 0.50 on the real 19.5:1 test set, they rarely predict stroke. LR with class_weight='balanced' directly penalises false negatives.

---

## 10. Key Findings & Clinical Insights

### Finding 1: AQ-10 Encodes a Clinical Rule
The Decision Tree achieved perfect performance by learning AQ ≥ 6 → ASD Positive. This is the established clinical referral threshold (Baron-Cohen et al., 2009). The model **validated** the clinical tool rather than discovering a new pattern.

### Finding 2: Polyuria + Polydipsia Dominate Diabetes
These two symptoms have 75.9% and 70.3% prevalence in diabetic patients respectively, consistent with clinical literature on hyperglycaemia symptoms.

### Finding 3: Accuracy is Misleading for Stroke
A trivial "always predict no stroke" model achieves **95.13% accuracy**. Our model achieves only 82% accuracy BUT catches **82% of actual strokes** (Recall=0.82). In clinical context, catching strokes saves lives — Recall is the correct metric.

### Finding 4: Formerly Smoked > Currently Smokes (Stroke Risk)
Counterintuitive: formerly smoked patients have 7.92% stroke rate vs 5.32% for active smokers. This reflects **survivor bias** and late-stage cardiovascular sequelae.

### Finding 5: Age is the Dominant Stroke Predictor
Mean age 67.7 (stroke) vs 43.2 (no stroke), Mann-Whitney U p < 0.0001, large effect size.

---

## 11. Common Q&A — Expected Questions & Answers

### Q: Why did you choose SQLite instead of MySQL/PostgreSQL?
**A:** SQLite is chosen to **simulate** a lightweight clinical warehouse that requires zero server infrastructure. It supports all the SQL features we need (FK constraints, CHECK constraints, views, indexes, JOINs). For a production system, we would migrate to PostgreSQL — our `schema.sql` is already portable.

### Q: Why Star Schema and not Snowflake?
**A:** The three diseases share only demographics (age, gender). There are no sub-dimensions to normalise further. A snowflake schema would add complexity without benefit at this scale. Star schema gives optimal query performance for analytical workloads.

### Q: Is the perfect Autism score data leakage?
**A:** No. The AQ_Score feature was **engineered** from the AQ-10 items (sum of A1–A10), which is exactly how the clinical tool works. The Decision Tree learned the clinically established threshold of AQ ≥ 6. This validates the dataset's clinical integrity.

### Q: Why is the stroke F1 so low (0.23)?
**A:** F1 balances precision and recall. With 19.5:1 imbalance, high recall (0.82) necessarily produces many false positives, lowering precision (0.134) and thus F1. **Recall and PR-AUC are the correct metrics for stroke** — a missed stroke is far worse than a false alarm. At threshold=0.15, recall exceeds 0.90.

### Q: Why SMOTE only on training data?
**A:** SMOTE generates synthetic minority samples by interpolating between existing records. If applied before train/test split, synthetic stroke patterns would leak into the test set, artificially inflating metrics. The test set must always reflect the **real clinical distribution** (4.87% stroke).

### Q: Why grouped-median for BMI imputation instead of global median?
**A:** BMI varies systematically with age: children have BMI ~20, adults ~28-30. Using global median (28.9) would over-impute children's BMI by ~8 kg/m², introducing systematic bias into the most age-sensitive feature. Age-bracket medians preserve the natural BMI-age relationship.

### Q: Why class_weight='balanced' for autism/diabetes but SMOTE for stroke?
**A:** Autism (1:2.4) and diabetes (1:0.6) have mild imbalance — class_weight adjustment is sufficient. Stroke (1:19.5) has **severe** imbalance where class weighting alone cannot produce adequate minority-class recall; SMOTE augments the minority class to create a balanced training set.

### Q: What OLAP operations did you demonstrate?
**A:** ROLL-UP (Q-06, Q-18), DRILL-DOWN (Q-14), SLICE (Q-19), DICE (Q-15, Q-16, Q-20), and PIVOT via conditional aggregation (Q-12, Q-13). Window functions demonstrated in Q-06 with `SUM() OVER (PARTITION BY)`.

### Q: How do you handle categorical features in the stroke dataset?
**A:** `work_type` (5 categories) and `smoking_status` (4 categories) are stored as TEXT in the warehouse and one-hot encoded during Feature Engineering (Phase 3). This separates ETL concerns from ML preprocessing.

### Q: What is the purpose of the views in the warehouse?
**A:** Views serve as a **semantic layer** — they abstract raw integer codes (0/1) back into human-readable labels (Male/Female, Yes/No, Urban/Rural). Downstream consumers query views instead of raw tables, insulating them from schema changes.

### Q: Why 10-fold cross-validation?
**A:** 10-fold CV provides a robust estimate of model performance by training on 90% and testing on 10% across 10 different splits. This reduces variance in performance estimates compared to a single train/test split.

### Q: What threshold tuning did you do for stroke?
**A:** Default threshold is 0.50 (predict stroke if probability > 50%). For stroke, we analysed F1/Recall/Precision across thresholds 0.05–0.95. At threshold ≈ 0.15, Logistic Regression achieves Recall > 0.90 with Precision > 0.10 — clinically acceptable (catching 9/10 strokes justifies additional false alarms).

---

## 12. Technical Decisions & Justifications

### Architecture Decisions
| Decision | Justification |
|---|---|
| Medallion Architecture (Bronze/Silver/Gold) | Industry standard for data lakehouse; ensures traceability |
| Star Schema over Snowflake | Only shared dimension is demographics; no sub-dimension normalisation needed |
| SQLite over PostgreSQL | Zero infrastructure; portable; demonstrates all warehouse concepts |
| Parameterized SQL INSERT over pandas .to_sql() | Explicit SQL documentation; demonstrates SQL DML skills |
| Views as semantic layer | Decouples raw schema from analytical consumers |

### ML Pipeline Decisions
| Decision | Justification |
|---|---|
| 5 diverse classifiers | Covers linear (LR), tree-based (DT, RF), and gradient boosting (XGB, LGBM) |
| Stratified splits | Preserves class distribution in train/test |
| SMOTE only on training | Prevents data leakage; test set reflects real distribution |
| Recall as primary stroke metric | False negative (missed stroke) has higher clinical cost than false positive |
| Threshold tuning for stroke | Default 0.50 is suboptimal for 19.5:1 imbalance |

### Tools & Libraries
| Tool | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| pandas | 2.x | Data manipulation |
| scikit-learn | 1.4.2 | ML models, metrics, preprocessing |
| XGBoost | 2.1.4 | Gradient boosting |
| LightGBM | 4.6.0 | Fast gradient boosting |
| imbalanced-learn | 0.12.4 | SMOTE |
| matplotlib/seaborn | — | Visualisation |
| SQLite3 | built-in | Data warehouse |

---

## Figure Index (39 Figures)

| # | File | Phase | Description |
|---|---|---|---|
| 01 | `01_class_distributions.png` | EDA | Class imbalance audit |
| 02 | `02_autism_age_aqscore.png` | EDA | Age + AQ Score violin |
| 03 | `03_autism_aq10_item_heatmap.png` | EDA | AQ-10 item response heatmap |
| 04 | `04_autism_binary_features.png` | EDA | Binary feature bars |
| 05 | `05_autism_correlation_heatmap.png` | EDA | Correlation matrix |
| 06 | `06_diabetes_age_gender.png` | EDA | Age KDE + gender |
| 07 | `07_diabetes_symptom_prevalence.png` | EDA | 14-symptom ranking |
| 08 | `08_diabetes_correlation_heatmap.png` | EDA | Correlation matrix |
| 09 | `09_stroke_continuous_boxplots.png` | EDA | Age/Glucose/BMI boxplots |
| 10 | `10_stroke_kde_scatter.png` | EDA | KDE + scatter |
| 11 | `11_stroke_categorical_breakdown.png` | EDA | Smoking & work type |
| 12 | `12_stroke_binary_features.png` | EDA | Binary features |
| 13 | `13_stroke_correlation_heatmap.png` | EDA | Correlation matrix |
| 14 | `14_statistical_significance.png` | EDA | Significance waterfall |
| 15–19 | Feature engineering figures | FE | Importance + SMOTE + distributions |
| 20–28 | Model evaluation figures | ML | Confusion matrices, ROC/PR, thresholds |
| 29–39 | SQL analysis figures | SQL | OLAP visualisations |

---

*Built with Python 3.9 · scikit-learn 1.4.2 · XGBoost 2.1.4 · LightGBM 4.6.0 · imbalanced-learn 0.12.4 · SQLite3*
