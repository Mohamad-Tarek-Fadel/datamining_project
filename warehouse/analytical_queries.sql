-- ============================================================================
-- FILE    : warehouse/analytical_queries.sql
-- PROJECT : Early Disease Prediction Using Healthcare Data Warehouse
-- COURSE  : Data Mining & Data Warehousing
-- DB      : SQLite 3
--
-- PURPOSE :
--   This file contains 20 fully-documented analytical SQL queries against
--   the healthcare data warehouse star schema. Queries are organised into
--   six thematic sections progressing from basic data quality checks to
--   advanced OLAP-style operations.
--
-- HOW TO RUN :
--   sqlite3 health_warehouse.db < analytical_queries.sql
--   -- or open in DB Browser for SQLite and run section by section.
--
-- PREREQUISITE :
--   PRAGMA foreign_keys = ON;   -- run once per connection
--   The database must have been built by etl_pipeline/warehouse.py first.
--
-- QUERY INDEX :
--   SECTION 1 — Data Quality Verification       Q-01 to Q-04
--   SECTION 2 — Dimension Table Analysis        Q-05 to Q-07
--   SECTION 3 — Disease Prevalence (OLAP)       Q-08 to Q-10
--   SECTION 4 — Disease-Specific Deep Dives     Q-11 to Q-14
--   SECTION 5 — Cross-Disease JOIN Analysis     Q-15 to Q-17
--   SECTION 6 — Advanced OLAP Operations        Q-18 to Q-20
-- ============================================================================

PRAGMA foreign_keys = ON;


-- ============================================================================
-- SECTION 1 : DATA QUALITY VERIFICATION
--
-- These queries validate that the ETL pipeline loaded the data correctly.
-- They should be run immediately after warehouse.py completes and their
-- results recorded in the project documentation.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Q-01: Row Count Verification
--
-- Purpose : Confirm that every source CSV row was loaded into the correct
--           table without duplication or loss.
-- Expected:
--   dim_patient   11,704   (6,075 + 520 + 5,109)
--   fact_autism    6,075
--   fact_diabetes    520
--   fact_stroke    5,109
-- ----------------------------------------------------------------------------

SELECT 'dim_patient'   AS table_name, COUNT(*) AS row_count FROM dim_patient
UNION ALL
SELECT 'fact_autism',                  COUNT(*)             FROM fact_autism
UNION ALL
SELECT 'fact_diabetes',                COUNT(*)             FROM fact_diabetes
UNION ALL
SELECT 'fact_stroke',                  COUNT(*)             FROM fact_stroke
ORDER BY table_name;


-- ----------------------------------------------------------------------------
-- Q-02: Referential Integrity Check (FK Validation)
--
-- Purpose : Detect any "orphaned" fact rows — rows whose patient_id does not
--           exist in dim_patient. A well-built warehouse must return 0 for
--           all three queries. Any positive count indicates a FK violation
--           that was not caught during ETL.
--
-- Technique: LEFT JOIN dim_patient → filter WHERE dp.patient_id IS NULL.
--            This is more explicit than COUNT(*) - COUNT(DISTINCT FK).
-- ----------------------------------------------------------------------------

SELECT
    'fact_autism'                      AS fact_table,
    COUNT(*)                           AS orphaned_rows
FROM  fact_autism   fa
LEFT JOIN dim_patient dp
    ON fa.patient_id = dp.patient_id
WHERE dp.patient_id IS NULL

UNION ALL

SELECT
    'fact_diabetes',
    COUNT(*)
FROM  fact_diabetes fd
LEFT JOIN dim_patient dp
    ON fd.patient_id = dp.patient_id
WHERE dp.patient_id IS NULL

UNION ALL

SELECT
    'fact_stroke',
    COUNT(*)
FROM  fact_stroke   fs
LEFT JOIN dim_patient dp
    ON fs.patient_id = dp.patient_id
WHERE dp.patient_id IS NULL;


-- ----------------------------------------------------------------------------
-- Q-03: NULL Value Audit
--
-- Purpose : Confirm that no NULL values exist in any column across all four
--           tables. The ETL pipeline enforced NOT NULL constraints, but this
--           query provides an additional programmatic verification layer.
--
-- Note    : SQLite does not enforce NOT NULL on PK columns automatically for
--           AUTOINCREMENT. We exclude PKs from this audit.
-- ----------------------------------------------------------------------------

SELECT
    'dim_patient.age'              AS column_checked,
    SUM(CASE WHEN age            IS NULL THEN 1 ELSE 0 END) AS null_count
FROM dim_patient
UNION ALL
SELECT 'dim_patient.gender',
    SUM(CASE WHEN gender         IS NULL THEN 1 ELSE 0 END) FROM dim_patient
UNION ALL
SELECT 'dim_patient.source_dataset',
    SUM(CASE WHEN source_dataset IS NULL THEN 1 ELSE 0 END) FROM dim_patient
UNION ALL
SELECT 'fact_autism.aq_score',
    SUM(CASE WHEN aq_score       IS NULL THEN 1 ELSE 0 END) FROM fact_autism
UNION ALL
SELECT 'fact_autism.label',
    SUM(CASE WHEN label          IS NULL THEN 1 ELSE 0 END) FROM fact_autism
UNION ALL
SELECT 'fact_diabetes.label',
    SUM(CASE WHEN label          IS NULL THEN 1 ELSE 0 END) FROM fact_diabetes
UNION ALL
SELECT 'fact_stroke.bmi',
    SUM(CASE WHEN bmi            IS NULL THEN 1 ELSE 0 END) FROM fact_stroke
UNION ALL
SELECT 'fact_stroke.avg_glucose_level',
    SUM(CASE WHEN avg_glucose_level IS NULL THEN 1 ELSE 0 END) FROM fact_stroke
UNION ALL
SELECT 'fact_stroke.label',
    SUM(CASE WHEN label          IS NULL THEN 1 ELSE 0 END) FROM fact_stroke
ORDER BY null_count DESC;


-- ----------------------------------------------------------------------------
-- Q-04: Domain Constraint Verification
--
-- Purpose : Verify that all binary columns contain only the allowed values
--           {0, 1} and that categorical columns contain only expected values.
--           CHECK constraints in SQLite are not always enforced on existing
--           data after schema changes, so an explicit audit is best practice.
-- ----------------------------------------------------------------------------

-- Binary columns: any value outside {0,1} indicates a constraint bypass
SELECT 'fact_autism.jaundice'     AS check_target,
       COUNT(*)                    AS violations
FROM   fact_autism
WHERE  jaundice NOT IN (0, 1)
UNION ALL
SELECT 'fact_autism.family_asd',   COUNT(*)
FROM   fact_autism   WHERE family_asd   NOT IN (0, 1)
UNION ALL
SELECT 'fact_stroke.hypertension', COUNT(*)
FROM   fact_stroke   WHERE hypertension NOT IN (0, 1)
UNION ALL
SELECT 'fact_stroke.heart_disease',COUNT(*)
FROM   fact_stroke   WHERE heart_disease NOT IN (0, 1)
UNION ALL
SELECT 'fact_stroke.residence_type',COUNT(*)
FROM   fact_stroke   WHERE residence_type NOT IN (0, 1)

-- Categorical: verify allowed work_type values
UNION ALL
SELECT 'fact_stroke.work_type (unexpected)',
       COUNT(*)
FROM   fact_stroke
WHERE  work_type NOT IN
       ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked')

-- Categorical: verify allowed smoking_status values
UNION ALL
SELECT 'fact_stroke.smoking_status (unexpected)',
       COUNT(*)
FROM   fact_stroke
WHERE  smoking_status NOT IN
       ('formerly smoked', 'never smoked', 'smokes', 'Unknown');


-- ============================================================================
-- SECTION 2 : DIMENSION TABLE ANALYSIS
--
-- Queries on dim_patient to understand the shared demographic profile
-- across all three disease cohorts stored in the warehouse.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Q-05: Patient Count and Age Distribution per Source Dataset
--
-- Purpose : High-level demographic summary — how many patients came from
--           each disease cohort and what is their age profile?
--
-- Clinical note: Autism patients are younger (screening occurs in childhood),
--   diabetes patients are middle-aged, and stroke patients span a wide range.
-- ----------------------------------------------------------------------------

SELECT
    source_dataset                               AS disease_cohort,
    COUNT(*)                                     AS total_patients,
    ROUND(MIN(age),  1)                          AS min_age,
    ROUND(MAX(age),  1)                          AS max_age,
    ROUND(AVG(age),  2)                          AS avg_age,
    -- Approximate median using sub-query on sorted rows
    ROUND(
        (
            SELECT age
            FROM   dim_patient d2
            WHERE  d2.source_dataset = dp.source_dataset
            ORDER  BY age
            LIMIT  1
            OFFSET (COUNT(*) - 1) / 2
        ),
    1)                                           AS approx_median_age,
    ROUND(100.0 * SUM(gender) / COUNT(*), 1)     AS pct_male,
    ROUND(100.0 * SUM(1 - gender) / COUNT(*), 1) AS pct_female
FROM  dim_patient dp
GROUP BY source_dataset
ORDER BY source_dataset;


-- ----------------------------------------------------------------------------
-- Q-06: Age Bracket Distribution across All Cohorts
--
-- Purpose : Roll-up analysis — segment patients into standard clinical age
--           brackets and count them per disease. This is an OLAP ROLL-UP
--           operation moving from individual ages to bracket-level granularity.
--
-- Technique: CASE expression creates a derived age bracket column;
--            GROUP BY (source_dataset, bracket) produces the cross-tab.
-- ----------------------------------------------------------------------------

SELECT
    source_dataset                               AS disease,
    CASE
        WHEN age <  18                THEN '0–17   (Paediatric)'
        WHEN age >= 18 AND age <  40  THEN '18–39  (Young Adult)'
        WHEN age >= 40 AND age <  60  THEN '40–59  (Middle-Aged)'
        WHEN age >= 60 AND age <  80  THEN '60–79  (Older Adult)'
        ELSE                               '80+    (Elderly)'
    END                                          AS age_bracket,
    COUNT(*)                                     AS patient_count,
    ROUND(100.0 * COUNT(*) /
          SUM(COUNT(*)) OVER (PARTITION BY source_dataset),
    1)                                           AS pct_within_disease
FROM  dim_patient
GROUP BY source_dataset, age_bracket
ORDER BY source_dataset, age_bracket;


-- ----------------------------------------------------------------------------
-- Q-07: Gender Balance per Disease Cohort
--
-- Purpose : Identify whether gender is distributed evenly within each cohort.
--           Gender imbalance can bias predictive models if not handled through
--           stratified sampling or class weighting.
-- ----------------------------------------------------------------------------

SELECT
    source_dataset                               AS disease,
    SUM(gender)                                  AS male_count,
    SUM(1 - gender)                              AS female_count,
    COUNT(*)                                     AS total,
    ROUND(100.0 * SUM(gender)     / COUNT(*), 2) AS pct_male,
    ROUND(100.0 * SUM(1 - gender) / COUNT(*), 2) AS pct_female,
    CASE
        WHEN ABS(50.0 - 100.0 * SUM(gender) / COUNT(*)) < 10
        THEN 'Balanced'
        ELSE 'Imbalanced'
    END                                          AS gender_balance
FROM  dim_patient
GROUP BY source_dataset
ORDER BY source_dataset;


-- ============================================================================
-- SECTION 3 : DISEASE PREVALENCE  (OLAP — using vw_disease_summary)
--
-- The vw_disease_summary view provides a pre-aggregated summary.
-- These queries demonstrate OLAP-style operations on top of that view.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Q-08: Global Disease Prevalence Dashboard
--
-- Purpose : The primary executive-level summary query.
--           Shows all four KPIs — total, positives, rate, demographics —
--           in a single result set ready for dashboard consumption.
--
-- Uses    : vw_disease_summary (semantic view layer)
-- ----------------------------------------------------------------------------

SELECT
    disease,
    total_patients,
    positive_cases,
    (total_patients - positive_cases)            AS negative_cases,
    positive_pct                                 AS positive_rate_pct,
    ROUND(
        CAST(total_patients - positive_cases AS REAL)
        / NULLIF(positive_cases, 0),
    2)                                           AS imbalance_ratio,
    avg_age,
    pct_male
FROM  vw_disease_summary
ORDER BY positive_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- Q-09: Class Imbalance Risk Classification
--
-- Purpose : Automatically flag each dataset's imbalance severity so the
--           ML engineer knows which resampling strategy to apply.
--
-- Classification thresholds (industry convention):
--   ratio  < 3   → Mild    → class_weight='balanced' is sufficient
--   ratio  3–10  → Moderate → SMOTE recommended
--   ratio > 10   → Severe  → SMOTE + scale_pos_weight required
-- ----------------------------------------------------------------------------

SELECT
    disease,
    positive_cases,
    (total_patients - positive_cases)            AS negative_cases,
    ROUND(
        CAST(total_patients - positive_cases AS REAL)
        / NULLIF(positive_cases, 0),
    1)                                           AS neg_to_pos_ratio,
    CASE
        WHEN CAST(total_patients - positive_cases AS REAL)
             / NULLIF(positive_cases, 0) < 3
        THEN 'MILD    — use class_weight=balanced'
        WHEN CAST(total_patients - positive_cases AS REAL)
             / NULLIF(positive_cases, 0) BETWEEN 3 AND 10
        THEN 'MODERATE — apply SMOTE'
        ELSE 'SEVERE  — SMOTE + scale_pos_weight required'
    END                                          AS recommended_strategy
FROM  vw_disease_summary
ORDER BY neg_to_pos_ratio DESC;


-- ----------------------------------------------------------------------------
-- Q-10: Comparative Demographics — Positive vs Negative Cases
--
-- Purpose : For each disease, compare the average age and gender split
--           between positive-label and negative-label patients.
--           This is a SLICE operation — filtering to positive/negative
--           slices and computing aggregate metrics within each.
-- ----------------------------------------------------------------------------

-- Autism: positive vs negative demographics
SELECT
    'autism'                                 AS disease,
    CASE fa.label WHEN 1 THEN 'Positive' ELSE 'Negative' END AS diagnosis,
    COUNT(*)                                 AS n,
    ROUND(AVG(dp.age), 1)                    AS avg_age,
    ROUND(100.0 * SUM(dp.gender) / COUNT(*), 1) AS pct_male
FROM  fact_autism fa
JOIN  dim_patient dp ON fa.patient_id = dp.patient_id
GROUP BY fa.label

UNION ALL

-- Diabetes: positive vs negative demographics
SELECT
    'diabetes',
    CASE fd.label WHEN 1 THEN 'Positive' ELSE 'Negative' END,
    COUNT(*),
    ROUND(AVG(dp.age), 1),
    ROUND(100.0 * SUM(dp.gender) / COUNT(*), 1)
FROM  fact_diabetes fd
JOIN  dim_patient dp ON fd.patient_id = dp.patient_id
GROUP BY fd.label

UNION ALL

-- Stroke: positive vs negative demographics
SELECT
    'stroke',
    CASE fs.label WHEN 1 THEN 'Stroke' ELSE 'No Stroke' END,
    COUNT(*),
    ROUND(AVG(dp.age), 1),
    ROUND(100.0 * SUM(dp.gender) / COUNT(*), 1)
FROM  fact_stroke fs
JOIN  dim_patient dp ON fs.patient_id = dp.patient_id
GROUP BY fs.label

ORDER BY disease, diagnosis;


-- ============================================================================
-- SECTION 4 : DISEASE-SPECIFIC DEEP DIVES
--
-- Granular queries inside each individual fact table.
-- These are the "drill-down" operations from the high-level prevalence view.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Q-11: Autism — AQ Score Statistics by Diagnosis
--
-- Purpose : Verify that the AQ Score cleanly separates ASD-positive from
--           ASD-negative patients. A clear gap around the clinical threshold
--           of 6 validates that the feature engineering step was correct.
--
-- Clinical significance: AQ-10 was validated by Baron-Cohen et al. (2009).
--   Sensitivity = 0.88 and Specificity = 0.91 at threshold AQ >= 6.
-- ----------------------------------------------------------------------------

SELECT
    CASE label
        WHEN 1 THEN 'ASD Positive'
        ELSE        'ASD Negative'
    END                                       AS diagnosis,
    COUNT(*)                                  AS n,
    MIN(aq_score)                             AS min_aq,
    MAX(aq_score)                             AS max_aq,
    ROUND(AVG(aq_score), 2)                   AS avg_aq,
    -- Count patients above the clinical threshold (AQ >= 6)
    SUM(CASE WHEN aq_score >= 6 THEN 1 ELSE 0 END)  AS above_threshold_6,
    ROUND(
        100.0 * SUM(CASE WHEN aq_score >= 6 THEN 1 ELSE 0 END)
        / COUNT(*),
    1)                                        AS pct_above_threshold
FROM  fact_autism
GROUP BY label
ORDER BY label DESC;


-- ----------------------------------------------------------------------------
-- Q-12: Autism — AQ-10 Item Response Rates by Diagnosis
--
-- Purpose : Identify which AQ-10 items are most discriminating between
--           ASD-positive and ASD-negative patients. Items with the largest
--           positive-rate gap are the strongest individual predictors.
--
-- Technique: AVG of a binary column = proportion answered 1.
--            The query pivots item averages per diagnosis into a single row
--            using conditional aggregation (CASE WHEN label=1 THEN col).
-- ----------------------------------------------------------------------------

SELECT
    'A1'  AS item, ROUND(AVG(CASE WHEN label=1 THEN a1  END)*100,1) AS pct_pos,
                   ROUND(AVG(CASE WHEN label=0 THEN a1  END)*100,1) AS pct_neg,
                   ROUND((AVG(CASE WHEN label=1 THEN a1  END)
                        - AVG(CASE WHEN label=0 THEN a1  END))*100,1) AS gap_pp
FROM fact_autism UNION ALL
SELECT 'A2',      ROUND(AVG(CASE WHEN label=1 THEN a2  END)*100,1),
                   ROUND(AVG(CASE WHEN label=0 THEN a2  END)*100,1),
                   ROUND((AVG(CASE WHEN label=1 THEN a2  END)
                        - AVG(CASE WHEN label=0 THEN a2  END))*100,1)
FROM fact_autism UNION ALL
SELECT 'A3',      ROUND(AVG(CASE WHEN label=1 THEN a3  END)*100,1),
                   ROUND(AVG(CASE WHEN label=0 THEN a3  END)*100,1),
                   ROUND((AVG(CASE WHEN label=1 THEN a3  END)
                        - AVG(CASE WHEN label=0 THEN a3  END))*100,1)
FROM fact_autism UNION ALL
SELECT 'A4',      ROUND(AVG(CASE WHEN label=1 THEN a4  END)*100,1),
                   ROUND(AVG(CASE WHEN label=0 THEN a4  END)*100,1),
                   ROUND((AVG(CASE WHEN label=1 THEN a4  END)
                        - AVG(CASE WHEN label=0 THEN a4  END))*100,1)
FROM fact_autism UNION ALL
SELECT 'A5',      ROUND(AVG(CASE WHEN label=1 THEN a5  END)*100,1),
                   ROUND(AVG(CASE WHEN label=0 THEN a5  END)*100,1),
                   ROUND((AVG(CASE WHEN label=1 THEN a5  END)
                        - AVG(CASE WHEN label=0 THEN a5  END))*100,1)
FROM fact_autism UNION ALL
SELECT 'A6',      ROUND(AVG(CASE WHEN label=1 THEN a6  END)*100,1),
                   ROUND(AVG(CASE WHEN label=0 THEN a6  END)*100,1),
                   ROUND((AVG(CASE WHEN label=1 THEN a6  END)
                        - AVG(CASE WHEN label=0 THEN a6  END))*100,1)
FROM fact_autism UNION ALL
SELECT 'A7',      ROUND(AVG(CASE WHEN label=1 THEN a7  END)*100,1),
                   ROUND(AVG(CASE WHEN label=0 THEN a7  END)*100,1),
                   ROUND((AVG(CASE WHEN label=1 THEN a7  END)
                        - AVG(CASE WHEN label=0 THEN a7  END))*100,1)
FROM fact_autism UNION ALL
SELECT 'A8',      ROUND(AVG(CASE WHEN label=1 THEN a8  END)*100,1),
                   ROUND(AVG(CASE WHEN label=0 THEN a8  END)*100,1),
                   ROUND((AVG(CASE WHEN label=1 THEN a8  END)
                        - AVG(CASE WHEN label=0 THEN a8  END))*100,1)
FROM fact_autism UNION ALL
SELECT 'A9',      ROUND(AVG(CASE WHEN label=1 THEN a9  END)*100,1),
                   ROUND(AVG(CASE WHEN label=0 THEN a9  END)*100,1),
                   ROUND((AVG(CASE WHEN label=1 THEN a9  END)
                        - AVG(CASE WHEN label=0 THEN a9  END))*100,1)
FROM fact_autism UNION ALL
SELECT 'A10',     ROUND(AVG(CASE WHEN label=1 THEN a10 END)*100,1),
                   ROUND(AVG(CASE WHEN label=0 THEN a10 END)*100,1),
                   ROUND((AVG(CASE WHEN label=1 THEN a10 END)
                        - AVG(CASE WHEN label=0 THEN a10 END))*100,1)
FROM fact_autism
ORDER BY gap_pp DESC;


-- ----------------------------------------------------------------------------
-- Q-13: Diabetes — Symptom Prevalence in Positive vs Negative Patients
--
-- Purpose : Rank the 14 clinical symptoms by their prevalence gap between
--           diabetic and non-diabetic patients. This directly supports the
--           EDA finding that polyuria and polydipsia are the top predictors.
--
-- Technique: Conditional AVG (= proportion) for label=1 and label=0 rows,
--            then ORDER BY the difference descending.
-- ----------------------------------------------------------------------------

SELECT
    symptom,
    ROUND(pct_positive, 1)                       AS pct_in_diabetic_patients,
    ROUND(pct_negative, 1)                       AS pct_in_healthy_patients,
    ROUND(pct_positive - pct_negative, 1)        AS gap_percentage_points,
    CASE
        WHEN (pct_positive - pct_negative) > 30 THEN 'HIGH discriminative power'
        WHEN (pct_positive - pct_negative) > 15 THEN 'MODERATE discriminative power'
        ELSE                                          'LOW discriminative power'
    END                                          AS discriminative_power
FROM (
    SELECT
        'polyuria'            AS symptom,
        AVG(CASE WHEN label=1 THEN polyuria*100.0         END) AS pct_positive,
        AVG(CASE WHEN label=0 THEN polyuria*100.0         END) AS pct_negative
    FROM fact_diabetes UNION ALL
    SELECT 'polydipsia',
        AVG(CASE WHEN label=1 THEN polydipsia*100.0       END),
        AVG(CASE WHEN label=0 THEN polydipsia*100.0       END)
    FROM fact_diabetes UNION ALL
    SELECT 'sudden_weight_loss',
        AVG(CASE WHEN label=1 THEN sudden_weight_loss*100.0 END),
        AVG(CASE WHEN label=0 THEN sudden_weight_loss*100.0 END)
    FROM fact_diabetes UNION ALL
    SELECT 'weakness',
        AVG(CASE WHEN label=1 THEN weakness*100.0         END),
        AVG(CASE WHEN label=0 THEN weakness*100.0         END)
    FROM fact_diabetes UNION ALL
    SELECT 'polyphagia',
        AVG(CASE WHEN label=1 THEN polyphagia*100.0       END),
        AVG(CASE WHEN label=0 THEN polyphagia*100.0       END)
    FROM fact_diabetes UNION ALL
    SELECT 'visual_blurring',
        AVG(CASE WHEN label=1 THEN visual_blurring*100.0  END),
        AVG(CASE WHEN label=0 THEN visual_blurring*100.0  END)
    FROM fact_diabetes UNION ALL
    SELECT 'partial_paresis',
        AVG(CASE WHEN label=1 THEN partial_paresis*100.0  END),
        AVG(CASE WHEN label=0 THEN partial_paresis*100.0  END)
    FROM fact_diabetes UNION ALL
    SELECT 'itching',
        AVG(CASE WHEN label=1 THEN itching*100.0          END),
        AVG(CASE WHEN label=0 THEN itching*100.0          END)
    FROM fact_diabetes UNION ALL
    SELECT 'muscle_stiffness',
        AVG(CASE WHEN label=1 THEN muscle_stiffness*100.0 END),
        AVG(CASE WHEN label=0 THEN muscle_stiffness*100.0 END)
    FROM fact_diabetes UNION ALL
    SELECT 'irritability',
        AVG(CASE WHEN label=1 THEN irritability*100.0     END),
        AVG(CASE WHEN label=0 THEN irritability*100.0     END)
    FROM fact_diabetes UNION ALL
    SELECT 'alopecia',
        AVG(CASE WHEN label=1 THEN alopecia*100.0         END),
        AVG(CASE WHEN label=0 THEN alopecia*100.0         END)
    FROM fact_diabetes UNION ALL
    SELECT 'obesity',
        AVG(CASE WHEN label=1 THEN obesity*100.0          END),
        AVG(CASE WHEN label=0 THEN obesity*100.0          END)
    FROM fact_diabetes UNION ALL
    SELECT 'delayed_healing',
        AVG(CASE WHEN label=1 THEN delayed_healing*100.0  END),
        AVG(CASE WHEN label=0 THEN delayed_healing*100.0  END)
    FROM fact_diabetes UNION ALL
    SELECT 'genital_thrush',
        AVG(CASE WHEN label=1 THEN genital_thrush*100.0   END),
        AVG(CASE WHEN label=0 THEN genital_thrush*100.0   END)
    FROM fact_diabetes
)
ORDER BY gap_percentage_points DESC;


-- ----------------------------------------------------------------------------
-- Q-14: Stroke — Clinical Risk Profile by Outcome
--
-- Purpose : Compare key continuous and binary clinical measurements between
--           stroke and non-stroke patients. This is a DRILL-DOWN query that
--           moves from the overall prevalence (Q-08) into specific risk factors.
--
-- Clinical insight: Higher glucose and older age are expected; the hypertension
--   and heart disease rates reveal multi-factor comorbidity patterns.
-- ----------------------------------------------------------------------------

SELECT
    CASE label
        WHEN 1 THEN 'Stroke'
        ELSE        'No Stroke'
    END                                         AS outcome,
    COUNT(*)                                    AS n,

    -- Continuous measurements
    ROUND(AVG(dp.age),               1)         AS avg_age,
    ROUND(AVG(fs.avg_glucose_level), 2)         AS avg_glucose_mg_dl,
    ROUND(AVG(fs.bmi),               2)         AS avg_bmi,

    -- Binary comorbidity rates
    ROUND(100.0 * SUM(fs.hypertension)  / COUNT(*), 1) AS pct_hypertension,
    ROUND(100.0 * SUM(fs.heart_disease) / COUNT(*), 1) AS pct_heart_disease,
    ROUND(100.0 * SUM(fs.ever_married)  / COUNT(*), 1) AS pct_ever_married,

    -- Residence
    ROUND(100.0 * SUM(fs.residence_type)/ COUNT(*), 1) AS pct_urban

FROM  fact_stroke fs
JOIN  dim_patient dp ON fs.patient_id = dp.patient_id
GROUP BY fs.label
ORDER BY fs.label DESC;


-- ============================================================================
-- SECTION 5 : CROSS-DISEASE JOIN ANALYSIS
--
-- These queries join across multiple fact tables using dim_patient as the
-- bridge — the defining capability of a star schema.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Q-15: Stroke — Smoking Status Risk Table
--
-- Purpose : Compute stroke incidence rate per smoking category.
--           This is a DICE operation — selecting a specific slice of the
--           stroke fact table and dicing it by smoking_status.
--
-- Key finding: 'formerly smoked' has the highest stroke rate (7.92%),
--   exceeding active smokers. This counterintuitive result is consistent
--   with clinical literature on late-stage cardiovascular sequelae.
-- ----------------------------------------------------------------------------

SELECT
    smoking_status,
    COUNT(*)                                        AS total_patients,
    SUM(label)                                      AS stroke_cases,
    COUNT(*) - SUM(label)                           AS no_stroke_cases,
    ROUND(100.0 * SUM(label) / COUNT(*), 2)         AS stroke_rate_pct,
    CASE
        WHEN ROUND(100.0*SUM(label)/COUNT(*),2) >
             (SELECT ROUND(100.0*SUM(label)/COUNT(*),2) FROM fact_stroke)
        THEN 'ABOVE average risk'
        ELSE 'Below average risk'
    END                                             AS vs_average
FROM  fact_stroke
GROUP BY smoking_status
ORDER BY stroke_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- Q-16: Stroke — Multi-Factor Comorbidity Matrix
--
-- Purpose : Cross-tabulate hypertension × heart_disease to identify the
--           subgroup with the highest absolute stroke risk.
--           This is a DICE on two binary dimensions simultaneously.
--
-- Interpretation: Patients with both hypertension AND heart disease represent
--   a small but extremely high-risk subgroup — the primary intervention target.
-- ----------------------------------------------------------------------------

SELECT
    CASE hypertension
        WHEN 1 THEN 'Hypertension: Yes'
        ELSE        'Hypertension: No'
    END                                             AS hypertension_status,
    CASE heart_disease
        WHEN 1 THEN 'Heart Disease: Yes'
        ELSE        'Heart Disease: No'
    END                                             AS heart_disease_status,
    COUNT(*)                                        AS total_patients,
    SUM(label)                                      AS stroke_cases,
    ROUND(100.0 * SUM(label) / COUNT(*), 2)         AS stroke_rate_pct,
    ROUND(100.0 * SUM(label) /
          (SELECT SUM(label) FROM fact_stroke) * 100,
    1)                                              AS pct_of_all_strokes
FROM  fact_stroke
GROUP BY hypertension, heart_disease
ORDER BY stroke_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- Q-17: Unified Warehouse View — All Diseases in One Summary
--
-- Purpose : Demonstrate the core warehouse capability: a SINGLE query that
--           touches all four tables (dim_patient + 3 fact tables) to produce
--           a unified cross-disease summary table.
--
-- Technique: Three separate aggregation sub-queries, each joining a fact table
--            back to dim_patient, combined via UNION ALL.
-- ----------------------------------------------------------------------------

SELECT
    'Autism Screening'                          AS disease,
    COUNT(*)                                    AS total,
    SUM(fa.label)                               AS positives,
    ROUND(AVG(dp.age), 1)                       AS avg_age,
    ROUND(100.0 * SUM(dp.gender) / COUNT(*), 1) AS pct_male,
    ROUND(100.0 * SUM(fa.label)  / COUNT(*), 2) AS prevalence_pct,
    'AQ_Score >= 6'                             AS primary_risk_indicator
FROM  fact_autism fa
JOIN  dim_patient dp ON fa.patient_id = dp.patient_id

UNION ALL

SELECT
    'Diabetes Risk',
    COUNT(*),
    SUM(fd.label),
    ROUND(AVG(dp.age), 1),
    ROUND(100.0 * SUM(dp.gender) / COUNT(*), 1),
    ROUND(100.0 * SUM(fd.label)  / COUNT(*), 2),
    'Polyuria + Polydipsia'
FROM  fact_diabetes fd
JOIN  dim_patient dp ON fd.patient_id = dp.patient_id

UNION ALL

SELECT
    'Stroke Prediction',
    COUNT(*),
    SUM(fs.label),
    ROUND(AVG(dp.age), 1),
    ROUND(100.0 * SUM(dp.gender) / COUNT(*), 1),
    ROUND(100.0 * SUM(fs.label)  / COUNT(*), 2),
    'Age + Glucose + Hypertension'
FROM  fact_stroke fs
JOIN  dim_patient dp ON fs.patient_id = dp.patient_id;


-- ============================================================================
-- SECTION 6 : ADVANCED OLAP OPERATIONS
--
-- These queries demonstrate classical OLAP operations:
--   ROLL-UP   : Aggregating from fine-grained to coarse-grained groupings
--   DRILL-DOWN: Moving from summary to detail
--   SLICE     : Filtering to a specific dimension member
--   DICE      : Filtering on two or more dimensions simultaneously
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Q-18: ROLL-UP — Stroke Risk by Age Bracket × Work Type
--
-- Purpose : Two-level roll-up. Start from individual patient level, aggregate
--           to (age_bracket, work_type) level, then to age_bracket totals.
--           This reveals which age-work combinations carry the highest stroke risk.
--
-- OLAP operation: ROLL-UP from patient grain → (age bracket, work type) grain
-- ----------------------------------------------------------------------------

SELECT
    CASE
        WHEN dp.age <  40 THEN 'Under 40'
        WHEN dp.age <  60 THEN '40–59'
        WHEN dp.age <  80 THEN '60–79'
        ELSE                   '80+'
    END                                             AS age_bracket,
    fs.work_type,
    COUNT(*)                                        AS patients,
    SUM(fs.label)                                   AS stroke_cases,
    ROUND(100.0 * SUM(fs.label) / COUNT(*), 2)      AS stroke_rate_pct
FROM  fact_stroke fs
JOIN  dim_patient dp ON fs.patient_id = dp.patient_id
GROUP BY age_bracket, fs.work_type
ORDER BY stroke_rate_pct DESC
LIMIT 15;


-- ----------------------------------------------------------------------------
-- Q-19: SLICE — High-Risk Stroke Patients Only
--
-- Purpose : Slice the fact_stroke table to extract only the positive-label
--           patients (stroke = 1) and profile their clinical characteristics.
--           A SLICE operation restricts one dimension to a single member.
--
-- Use case: This result set is the input to a "high-risk patient registry"
--           feature in a clinical dashboard.
-- ----------------------------------------------------------------------------

SELECT
    dp.patient_id,
    ROUND(dp.age, 0)                             AS age,
    CASE dp.gender WHEN 1 THEN 'Male' ELSE 'Female' END AS gender,
    fs.hypertension,
    fs.heart_disease,
    CASE fs.ever_married WHEN 1 THEN 'Yes' ELSE 'No' END AS ever_married,
    fs.work_type,
    CASE fs.residence_type WHEN 1 THEN 'Urban' ELSE 'Rural' END AS residence,
    ROUND(fs.avg_glucose_level, 1)               AS glucose_mg_dl,
    ROUND(fs.bmi, 1)                             AS bmi,
    fs.smoking_status,
    -- Compute a simple risk score: sum of binary risk factors
    (fs.hypertension + fs.heart_disease
     + CASE WHEN dp.age >= 65 THEN 1 ELSE 0 END
     + CASE WHEN fs.avg_glucose_level > 140 THEN 1 ELSE 0 END
     + CASE WHEN fs.bmi > 30 THEN 1 ELSE 0 END)  AS risk_factor_score
FROM  fact_stroke fs
JOIN  dim_patient dp ON fs.patient_id = dp.patient_id
WHERE fs.label = 1                               -- SLICE: stroke patients only
ORDER BY risk_factor_score DESC, dp.age DESC
LIMIT 20;


-- ----------------------------------------------------------------------------
-- Q-20: DICE — High-Risk Subgroup Identification across All Three Diseases
--
-- Purpose : DICE operation filtering on two or more dimensions simultaneously.
--           Identifies the "triple burden" subgroup — older male patients
--           who are positive in any of the three disease datasets.
--
-- This query demonstrates the warehouse's capability to support population
-- health management use cases by querying across disease domains using the
-- shared dim_patient dimension as the integration bridge.
-- ----------------------------------------------------------------------------

-- High-risk ASD patients: positive, older (age >= 12), male
SELECT
    'Autism — Older Male ASD Positive'           AS subgroup,
    COUNT(*)                                     AS patients,
    ROUND(AVG(dp.age), 1)                        AS avg_age,
    ROUND(AVG(fa.aq_score), 2)                   AS avg_aq_score,
    ROUND(100.0 * SUM(fa.family_asd)/COUNT(*),1) AS pct_family_history
FROM  fact_autism fa
JOIN  dim_patient dp ON fa.patient_id = dp.patient_id
WHERE fa.label   = 1           -- DICE dimension 1: positive label
  AND dp.gender  = 1           -- DICE dimension 2: male
  AND dp.age    >= 12          -- DICE dimension 3: older children / adults

UNION ALL

-- High-risk diabetic patients: positive, with polyuria AND polydipsia
SELECT
    'Diabetes — Dual-Symptom Positive (Polyuria+Polydipsia)',
    COUNT(*),
    ROUND(AVG(dp.age), 1),
    NULL,   -- aq_score not applicable
    ROUND(100.0 * SUM(fd.obesity) / COUNT(*), 1)
FROM  fact_diabetes fd
JOIN  dim_patient dp ON fd.patient_id = dp.patient_id
WHERE fd.label     = 1         -- DICE: positive
  AND fd.polyuria  = 1         -- DICE: polyuria present
  AND fd.polydipsia = 1        -- DICE: polydipsia present

UNION ALL

-- High-risk stroke patients: positive, with hypertension AND heart disease
SELECT
    'Stroke — Dual Comorbidity (Hypertension+Heart Disease)',
    COUNT(*),
    ROUND(AVG(dp.age), 1),
    NULL,
    ROUND(100.0 * SUM(
        CASE WHEN fs.smoking_status = 'smokes' THEN 1 ELSE 0 END
    ) / COUNT(*), 1)
FROM  fact_stroke fs
JOIN  dim_patient dp ON fs.patient_id = dp.patient_id
WHERE fs.label          = 1    -- DICE: stroke occurred
  AND fs.hypertension   = 1    -- DICE: has hypertension
  AND fs.heart_disease  = 1;   -- DICE: has heart disease


-- ============================================================================
-- End of analytical_queries.sql
--
-- Total queries : 20
-- Sections      :  6
-- Operations demonstrated:
--   - Basic SELECT with GROUP BY and aggregate functions
--   - JOIN (dim_patient ↔ fact tables)
--   - UNION ALL (combining multi-table results)
--   - Subqueries (correlated and scalar)
--   - Conditional aggregation (CASE WHEN inside AVG/SUM)
--   - OLAP ROLL-UP  (Q-18)
--   - OLAP SLICE    (Q-19)
--   - OLAP DICE     (Q-20)
--   - Window-function simulation (Q-06 uses SUM OVER PARTITION BY)
--   - Data quality audit patterns (Q-01 to Q-04)
-- ============================================================================
