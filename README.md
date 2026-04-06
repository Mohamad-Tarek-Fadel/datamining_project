# Early Disease Prediction Using Healthcare Data Warehouse

> **University Project — Data Mining & Data Warehousing**
> Nile, Misr, and Upper Egypt Universities (NMU) · Faculty of Computers and Information

---

## Abstract

This project simulates a **clinical intelligence system** that integrates three heterogeneous healthcare datasets into a unified SQLite data warehouse and applies ensemble machine learning to predict early-stage disease. The pipeline covers the full data science lifecycle: ETL, star-schema warehousing, exploratory analysis, feature engineering (including SMOTE for severe class imbalance), and evaluation of five classifiers — Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM. Key findings show that the AQ-10 behavioral questionnaire encodes a near-perfect clinical threshold for autism screening, polyuria and polydipsia are dominant discriminators for diabetes, and stroke prediction under a 19.5:1 class imbalance requires recall-prioritised evaluation with threshold tuning rather than accuracy maximisation.

---

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [Datasets](#2-datasets)
3. [Repository Structure](#3-repository-structure)
4. [How to Run](#4-how-to-run)
5. [Results Summary](#5-results-summary)
6. [Key Findings](#6-key-findings)
7. [Technical Decisions](#7-technical-decisions)
8. [Limitations & Future Work](#8-limitations--future-work)
9. [References](#9-references)

---

## 1. Project Architecture

```
Raw CSVs (datasets/)
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│              LAYER 1 — ETL PIPELINE                     │
│  load_and_inspect.py → clean.py → warehouse.py          │
│  • Fix typos, encode binaries, impute BMI by age bracket│
│  • One-hot encode multi-category stroke features        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           LAYER 2 — DATA WAREHOUSE (SQLite)             │
│  Star Schema: dim_patient + 3 fact tables               │
│  4 analytical views · 8 indexes · FK enforcement        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LAYER 3 — EDA (Phase 2)                    │
│  Class distributions · Correlation heatmaps             │
│  Chi-square tests · Mann-Whitney U · Feature importance │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         LAYER 4 — FEATURE ENGINEERING (Phase 3)         │
│  Stratified splits · StandardScaler · SMOTE             │
│  Class weights · CV setup · Artifact persistence        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           LAYER 5 — ML ENGINE (Phase 4)                 │
│  5 models × 3 diseases · ROC/PR/CM evaluation           │
│  Threshold optimisation · Model persistence             │
└─────────────────────────────────────────────────────────┘
```

### Star Schema (Data Warehouse)

```
                   ┌──────────────────────────┐
                   │       dim_patient         │
                   │  patient_id  PK AUTO      │
                   │  age         REAL         │
                   │  gender      INTEGER 0/1  │
                   │  source_dataset TEXT      │
                   └────────────┬─────────────┘
                                │  (FK ON DELETE CASCADE)
           ┌────────────────────┼─────────────────────┐
           │                    │                     │
  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
  │  fact_autism    │  │  fact_diabetes  │  │  fact_stroke    │
  │  a1–a10 items   │  │  14 symptoms    │  │  clinical meas. │
  │  aq_score       │  │  label          │  │  smoking, work  │
  │  jaundice       │  └─────────────────┘  │  label          │
  │  family_asd     │                       └─────────────────┘
  │  label          │
  └─────────────────┘

  Views: vw_autism_full · vw_diabetes_full · vw_stroke_full
         vw_disease_summary
  Indexes: 8 (FK + label + source_dataset + gender)
```

---

## 2. Datasets

| Dataset | Source | Raw Rows | Features | Target | Class Imbalance |
|---|---|---|---|---|---|
| **Autism Screening** | Kaggle / Fadi Fayez Thabtah | 6,075 | 16 | ASD Diagnosis (YES/NO) | ~1 : 2.4 (mild) |
| **Diabetes Risk** | Kaggle / UCI | 520 | 17 | Diabetic (Positive/Negative) | ~1 : 1.6 (mild) |
| **Stroke Prediction** | Kaggle / fedesoriano | 5,110 | 12 | Stroke event (1/0) | ~1 : 19.5 (**severe**) |

### Cleaning Rules Applied

| Dataset | Actions |
|---|---|
| Autism | Fixed `Jauundice` typo → `Jaundice`; normalised `Sex` (m/f → Male/Female); encoded all binary columns to 0/1; engineered `AQ_Score = sum(A1..A10)` |
| Diabetes | Encoded `Gender` (Male=1, Female=0); mapped 14 Yes/No symptom columns → 1/0; renamed `class` → `Class` |
| Stroke | Dropped `id` column; removed 1 `gender='Other'` row; imputed 201 missing `bmi` values using **median grouped by age bracket** (0–18, 19–40, 41–60, 61–80, 81+) to avoid age-group bias |

---

## 3. Repository Structure

```
early-disease-prediction/
│
├── datasets/                             Raw CSVs — never modified
│   ├── Autism_Screening_Data_Combined.csv
│   ├── diabetes_data_upload.csv
│   ├── healthcare-dataset-stroke-data.csv
│   └── cleaned/                          Output of clean.py
│       ├── autism_cleaned.csv
│       ├── diabetes_cleaned.csv
│       └── stroke_cleaned.csv
│
├── etl_pipeline/
│   ├── load_and_inspect.py               Phase 0 — shape & missing-value audit
│   ├── clean.py                          Phase 1A — all cleaning rules
│   └── warehouse.py                      Phase 1B — SQLite star schema builder
│
├── warehouse/
│   └── health_warehouse.db               Auto-generated SQLite DB (1.2 MB)
│
├── eda/
│   ├── create_eda_notebook.py            Notebook generator script
│   └── 02_eda.ipynb                      Phase 2 — 28 cells, 14 figures
│
├── models/
│   ├── create_feature_engineering_notebook.py
│   ├── 03_feature_engineering.ipynb      Phase 3 — split, scale, SMOTE
│   ├── create_modeling_notebook.py
│   ├── 04_modeling.ipynb                 Phase 4 — 5 models × 3 diseases
│   └── saved/                            Persisted artifacts
│       ├── autism_artifacts.pkl          Train/test arrays + scaler + CV
│       ├── diabetes_artifacts.pkl
│       ├── stroke_artifacts.pkl          Includes SMOTE arrays
│       ├── autism_best_model.pkl
│       ├── diabetes_best_model.pkl
│       ├── stroke_best_model.pkl
│       ├── autism_results.csv
│       ├── diabetes_results.csv
│       └── stroke_results.csv
│
├── reports/
│   └── figures/                          28 publication-quality PNG figures
│       ├── 01_class_distributions.png
│       ├── 02–14_*                       EDA figures
│       ├── 15–19_*                       Feature engineering figures
│       └── 20–28_*                       Modeling figures
│
├── requirements.txt
└── README.md
```

---

## 4. How to Run

### Prerequisites

```bash
# Python 3.9 or higher required
python --version

# Install all dependencies
pip install -r requirements.txt
```

### Step-by-Step Execution

```bash
# ── PHASE 0: Inspect raw datasets ───────────────────────────────────────────
python etl_pipeline/load_and_inspect.py

# ── PHASE 1A: Clean all three datasets ──────────────────────────────────────
python etl_pipeline/clean.py
# Output: datasets/cleaned/{autism,diabetes,stroke}_cleaned.csv

# ── PHASE 1B: Build the SQLite Data Warehouse ───────────────────────────────
python etl_pipeline/warehouse.py
# Output: warehouse/health_warehouse.db  (11,704 patients, 4 tables, 4 views)

# ── PHASE 2: Exploratory Data Analysis ──────────────────────────────────────
python eda/create_eda_notebook.py
python -m jupyter nbconvert --to notebook --execute eda/02_eda.ipynb \
       --inplace --ExecutePreprocessor.timeout=180
# Output: 14 figures in reports/figures/

# ── PHASE 3: Feature Engineering ────────────────────────────────────────────
python models/create_feature_engineering_notebook.py
python -m jupyter nbconvert --to notebook --execute \
       models/03_feature_engineering.ipynb \
       --inplace --ExecutePreprocessor.timeout=300
# Output: models/saved/{autism,diabetes,stroke}_artifacts.pkl + 5 figures

# ── PHASE 4: Model Training & Evaluation ────────────────────────────────────
python models/create_modeling_notebook.py
python -m jupyter nbconvert --to notebook --execute \
       models/04_modeling.ipynb \
       --inplace --ExecutePreprocessor.timeout=600
# Output: 3 best models + results CSVs + 9 figures

# ── OPEN NOTEBOOKS INTERACTIVELY ────────────────────────────────────────────
jupyter notebook
```

### Quick Verification

```bash
# Confirm the warehouse built correctly
python -c "
import sqlite3
conn = sqlite3.connect('warehouse/health_warehouse.db')
for table in ['dim_patient','fact_autism','fact_diabetes','fact_stroke']:
    n = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f'{table}: {n:,} rows')
conn.close()
"

# Confirm all 28 figures exist
python -c "
import glob; figs = glob.glob('reports/figures/*.png')
print(f'{len(figs)} figures saved')
"
```

---

## 5. Results Summary

### Best Model per Disease (Test Set Evaluation)

| Disease | Best Model | F1 | Recall | Precision | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|---|
| **Autism Screening** | Decision Tree | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Diabetes Risk** | Random Forest | **0.984** | 0.969 | 1.000 | 1.000 | 1.000 | 0.961 |
| **Stroke Prediction** | Logistic Regression | 0.230 | **0.820** | 0.134 | **0.840** | **0.275** | 0.258 |

> **Primary metric for stroke is Recall and PR-AUC**, not F1, because a missed
> stroke (False Negative) carries far greater clinical cost than a false alarm.

### Full Model Comparison — Stroke (Primary Challenge)

| Model | CV F1 | Test F1 | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.795 | **0.230** | **0.820** | **0.840** | **0.275** |
| Decision Tree | 0.893 | 0.133 | 0.120 | 0.713 | 0.107 |
| Random Forest | 0.963 | 0.209 | 0.140 | 0.776 | 0.211 |
| XGBoost | 0.968 | 0.175 | 0.100 | 0.791 | 0.231 |
| LightGBM | 0.971 | 0.097 | 0.060 | 0.790 | 0.185 |

> The large gap between CV F1 (0.96–0.97) and Test F1 (~0.10–0.23) for tree models
> is explained by threshold miscalibration: models trained on SMOTE-balanced data
> use a 0.50 threshold, which is far too conservative for the original 19.5:1 test
> distribution. Threshold tuning (see Figure 26) recovers Recall to 0.90+ at threshold ≈ 0.15.

---

## 6. Key Findings

### 6.1 Autism Screening

- **AQ Score encodes the clinical rule.** The Decision Tree learned the established AQ-10 clinical referral threshold (AQ ≥ 6 → ASD Positive), achieving perfect test-set performance. This validates the dataset's clinical integrity rather than indicating data leakage.
- **Family history is statistically significant** (χ² p < 0.001). Models should never drop `Family_ASD` during feature selection.
- **AQ items A1, A4, and A10** showed the largest positive rate gap between ASD-positive and ASD-negative groups in the AQ-10 heatmap.

### 6.2 Diabetes Risk

- **Polyuria (75.9%) and Polydipsia (70.3%)** are by far the most prevalent symptoms in diabetic patients, consistent with clinical literature on early-stage Type 2 diabetes.
- **Random Forest and LightGBM** achieved perfect precision (0.0 false positives on the 104-sample test set). Given n=520, this should be validated on a larger independent cohort.
- **Overfitting risk is non-trivial**: only 416 training samples. The 10-fold CV F1 ≈ 0.96 closely matching test F1 ≈ 0.98 suggests the current models generalise, but deeper models (e.g., XGBoost with high depth) should be approached cautiously.

### 6.3 Stroke Prediction

- **Accuracy is a misleading metric.** A trivial baseline that predicts "no stroke" for every patient achieves **95.13% accuracy** — higher than most trained models. PR-AUC and Recall are the appropriate primary metrics.
- **Formerly smoked has the highest stroke rate (7.92%)**, exceeding active smokers (5.32%) — a clinically counter-intuitive finding that reflects survivor bias and late-stage sequelae.
- **Threshold tuning is essential.** At threshold = 0.15 (clinical threshold), Logistic Regression achieves Recall > 0.90 while maintaining Precision > 0.10 — a clinically acceptable trade-off where catching 9 in 10 strokes justifies the additional false alarms.
- **Age is the single strongest predictor** (mean 67.7 for stroke vs 43.2 for no stroke, Mann-Whitney U p < 0.0001, large effect size).

---

## 7. Technical Decisions

### Why Grouped-Median BMI Imputation?

Global median BMI ≈ 28.9 kg/m². Age-stratified medians reveal:

| Age Bracket | Median BMI | Missing Count |
|---|---|---|
| 0–18 | 20.1 | 21 |
| 19–40 | 28.0 | 39 |
| 41–60 | 30.3 | 54 |
| 61–80 | 29.3 | 86 |
| 81+ | 27.5 | 1 |

Using a global median would over-impute children's BMI by ~8 kg/m², introducing systematic bias into the most age-sensitive feature.

### Why SMOTE Only on Training Data?

SMOTE generates synthetic minority-class samples by interpolating between existing minority records. Applying it before the train/test split would leak synthetic stroke patterns into the test set, inflating performance metrics. In this project, SMOTE is applied exclusively to the training partition; the test set always reflects the true clinical distribution (19.5:1 imbalance).

### Why Star Schema for the Warehouse?

The three diseases share a common demographic dimension (age, gender). The star schema separates shared patient metadata (`dim_patient`) from disease-specific measurements (three fact tables), enabling:
- Cross-disease queries via `JOIN` on `patient_id`
- Efficient filtering via indexed label and source_dataset columns
- Referential integrity enforced through FK constraints and `ON DELETE CASCADE`

### Why LR Outperforms Ensembles on Stroke?

Tree-based models trained on SMOTE-balanced data learn a decision boundary calibrated for equal class distribution. At the default threshold of 0.50, they rarely predict stroke on the original-distribution test set (high precision, catastrophically low recall). Logistic Regression with `class_weight='balanced'` directly penalises false negatives proportional to class frequency, achieving **Recall = 0.820** vs. LightGBM's 0.060 at the same threshold.

---

## 8. Limitations & Future Work

### Current Limitations

| Limitation | Impact |
|---|---|
| Diabetes dataset is very small (n=520) | Results may not generalise to larger populations |
| All datasets sourced from Kaggle — not from real EHR systems | Selection bias; unknown provenance of some labels |
| Autism dataset contains only AQ-10 items — no biomarkers | Model essentially learns a scoring rule, not a clinical pattern |
| No temporal data (no patient follow-up records) | Cannot model disease progression or recurrence |
| SMOTE before CV for stroke (computational constraint) | Slight optimism in CV F1 scores for tree models |
| Single-node SQLite warehouse | Cannot simulate concurrent multi-department write loads |

### Future Work

1. **SHAP explainability** — Deploy SHAP (SHapley Additive exPlanations) beeswarm plots for the stroke model to identify which features push individual predictions toward or away from stroke risk. This is the current gold standard for clinical AI transparency.

2. **Hyperparameter tuning at scale** — Apply `RandomizedSearchCV` with `n_iter=50` and `cv=5` for XGBoost and Random Forest specifically on the stroke dataset, using `scoring='average_precision'` (PR-AUC) rather than F1.

3. **Calibration** — Apply `CalibratedClassifierCV` (isotonic regression) to tree models trained on SMOTE data so their probability scores are properly calibrated for the original imbalanced test distribution.

4. **Extended warehouse** — Migrate from SQLite to PostgreSQL and add a `fact_lab_results` table to incorporate blood glucose, HbA1c, and lipid panel measurements from electronic health records.

5. **Neural network baselines** — Add a simple MLP (`sklearn.neural_network.MLPClassifier`) as a sixth model, particularly for the stroke dataset where non-linear interactions between age, glucose, and BMI may yield better PR-AUC.

6. **Prospective validation** — Validate the stroke model on the MIMIC-IV clinical database or PhysioNet datasets to measure real-world generalisation.

---

## 9. References

### Datasets

1. **Autism Screening Adult Dataset** — Thabtah, F. (2017). Autism Spectrum Disorder Screening: Machine Learning Adaptation and DSM-5 Fulfillment. *ACM CHIL 2020 / Kaggle*. Retrieved from https://www.kaggle.com/datasets/faizunnabi/autism-screening

2. **Early Stage Diabetes Risk Prediction Dataset** — Islam, M.M.F., Ferdousi, R., Rahman, S., Bushra, H.Y. (2020). Likelihood Prediction of Diabetes at Early Stage Using Data Mining Techniques. *Springer Nature, Computer Vision and Machine Intelligence in Medical Image Analysis*. Retrieved from https://www.kaggle.com/datasets/ishandutta/early-stage-diabetes-risk-prediction-dataset

3. **Stroke Prediction Dataset** — fedesoriano (2021). *Kaggle*. Retrieved from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

### Methods & Libraries

4. Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

5. Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD 2016*.

6. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... Liu, T.Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.

7. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

8. Lemaître, G., Nogueira, F., Aridas, C.K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *JMLR*, 18(17), 1–5.

9. Kimball, R., Ross, M. (2013). *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modelling* (3rd ed.). Wiley.

---

## Figure Index (28 Figures in `reports/figures/`)

| # | File | Phase | Description |
|---|---|---|---|
| 01 | `01_class_distributions.png` | EDA | Class imbalance audit — all 3 diseases |
| 02 | `02_autism_age_aqscore.png` | EDA | Age histogram + AQ Score violin by diagnosis |
| 03 | `03_autism_aq10_item_heatmap.png` | EDA | AQ-10 item response rates per class |
| 04 | `04_autism_binary_features.png` | EDA | Sex / Jaundice / Family_ASD stacked bars |
| 05 | `05_autism_correlation_heatmap.png` | EDA | Pearson correlation matrix — autism |
| 06 | `06_diabetes_age_gender.png` | EDA | Age KDE + gender distribution by diagnosis |
| 07 | `07_diabetes_symptom_prevalence.png` | EDA | 14-symptom prevalence: diabetic vs non-diabetic |
| 08 | `08_diabetes_correlation_heatmap.png` | EDA | Pearson correlation matrix — diabetes |
| 09 | `09_stroke_continuous_boxplots.png` | EDA | Age / Glucose / BMI boxplots by outcome |
| 10 | `10_stroke_kde_scatter.png` | EDA | Age KDE + Glucose vs BMI scatter |
| 11 | `11_stroke_categorical_breakdown.png` | EDA | Stroke rate by smoking & work type |
| 12 | `12_stroke_binary_features.png` | EDA | Binary feature breakdown by stroke outcome |
| 13 | `13_stroke_correlation_heatmap.png` | EDA | Pearson correlation matrix — stroke |
| 14 | `14_statistical_significance.png` | EDA | −log₁₀(p) significance waterfall (14 tests) |
| 15 | `15_autism_feature_importance.png` | Feat. Eng. | RF feature importances — autism |
| 16 | `16_diabetes_feature_importance.png` | Feat. Eng. | RF feature importances — diabetes |
| 17 | `17_stroke_smote_before_after.png` | Feat. Eng. | SMOTE class balance before vs after |
| 18 | `18_stroke_feature_importance.png` | Feat. Eng. | RF feature importances — stroke |
| 19 | `19_train_test_distribution_check.png` | Feat. Eng. | KS test: train/test distribution alignment |
| 20 | `20_autism_confusion_matrices.png` | Modeling | 5 confusion matrices — autism |
| 21 | `21_autism_roc_pr.png` | Modeling | ROC + PR curves — autism |
| 22 | `22_diabetes_confusion_matrices.png` | Modeling | 5 confusion matrices — diabetes |
| 23 | `23_diabetes_roc_pr.png` | Modeling | ROC + PR curves — diabetes |
| 24 | `24_stroke_confusion_matrices.png` | Modeling | 5 confusion matrices — stroke |
| 25 | `25_stroke_roc_pr.png` | Modeling | ROC + PR curves — stroke |
| 26 | `26_stroke_threshold_optimisation.png` | Modeling | F1/Recall/Precision vs threshold + sensitivity/specificity |
| 27 | `27_model_comparison_heatmap.png` | Modeling | 3-disease × 5-model metric heatmap |
| 28 | `28_best_model_per_disease.png` | Modeling | Best-model comparison bar chart |

---

*Built with Python 3.9 · scikit-learn 1.4.2 · XGBoost 2.1.4 · LightGBM 4.6.0 · imbalanced-learn 0.12.4 · SQLite3*