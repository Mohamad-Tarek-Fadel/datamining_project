-- ============================================================================
-- FILE    : warehouse/schema.sql
-- PROJECT : Early Disease Prediction Using Healthcare Data Warehouse
-- COURSE  : Data Mining & Data Warehousing
-- DB      : SQLite 3
--
-- PURPOSE :
--   This file contains the complete, fully-documented DDL (Data Definition
--   Language) for the Healthcare Data Warehouse Star Schema.
--   It is designed to be idempotent — safe to run multiple times.
--
-- STAR SCHEMA OVERVIEW :
--
--                  +---------------------------+
--                  |       dim_patient          |
--                  |  PK: patient_id (AUTO)     |
--                  |      age        REAL        |
--                  |      gender     INTEGER     |
--                  |      source_dataset TEXT    |
--                  +-------------+-------------+
--                                |
--             +------------------+-----------------+
--             |                  |                 |
--   +---------+-------+  +-------+--------+  +----+----------+
--   |   fact_autism   |  | fact_diabetes  |  | fact_stroke   |
--   |  AQ-10 items    |  | 14 symptoms    |  | clinical meas.|
--   |  aq_score       |  | label          |  | smoking/work  |
--   |  jaundice       |  +----------------+  | label         |
--   |  family_asd     |                      +---------------+
--   |  label          |
--   +-----------------+
--
-- DESIGN DECISIONS :
--   1. Star schema chosen over snowflake: the three disease domains share
--      only demographics (age, gender); no additional normalisation of
--      dim_patient sub-dimensions is warranted at this dataset scale.
--   2. SQLite chosen to simulate a lightweight clinical warehouse that
--      requires zero server infrastructure — appropriate for a prototype
--      that demonstrates warehousing concepts without production overhead.
--   3. Every binary column enforces CHECK (col IN (0,1)) to replicate the
--      NOT NULL + domain constraint typical of a production OLTP source.
--   4. ON DELETE CASCADE on all FK references ensures referential integrity
--      is maintained if a patient record is retracted.
--   5. The target column is uniformly named "label" across all fact tables
--      to simplify cross-table analytical queries.
-- ============================================================================


-- ============================================================================
-- SECTION 0 : PRAGMA CONFIGURATION
-- ============================================================================

-- Enable foreign key enforcement (SQLite disables FKs by default).
-- Must be run once per connection — not stored in the schema.
PRAGMA foreign_keys = ON;

-- Write-Ahead Logging: improves read/write concurrency for dashboards
-- that query while ETL is still loading data.
PRAGMA journal_mode  = WAL;

-- NORMAL synchronous mode: safe for analytics workloads; avoids the
-- performance overhead of FULL (fsync after every transaction).
PRAGMA synchronous   = NORMAL;


-- ============================================================================
-- SECTION 1 : DROP EXISTING OBJECTS  (Idempotent clean-slate re-run)
--
-- Views must be dropped before the tables they reference.
-- Fact tables must be dropped before dim_patient due to FK dependencies.
-- ============================================================================

DROP VIEW  IF EXISTS vw_disease_summary;
DROP VIEW  IF EXISTS vw_stroke_full;
DROP VIEW  IF EXISTS vw_diabetes_full;
DROP VIEW  IF EXISTS vw_autism_full;

DROP TABLE IF EXISTS fact_stroke;
DROP TABLE IF EXISTS fact_diabetes;
DROP TABLE IF EXISTS fact_autism;
DROP TABLE IF EXISTS dim_patient;


-- ============================================================================
-- SECTION 2 : DIMENSION TABLE
-- ============================================================================

-- ----------------------------------------------------------------------------
-- TABLE: dim_patient
--
-- The single shared dimension in this star schema.
--
-- GRAIN  : One row per patient record ingested from any source dataset.
--          A patient appearing in both (hypothetically) the diabetes and
--          stroke cohorts would have two separate rows — the datasets do
--          not share a real-world patient identifier, so we treat each
--          source record as an independent entity.
--
-- COLUMNS:
--   patient_id     Surrogate primary key (AUTOINCREMENT). Replaces all
--                  source natural keys (autism has no ID; diabetes has no ID;
--                  stroke's "id" column was dropped during ETL because it
--                  was a meaningless surrogate in the source file).
--
--   age            Patient age. Stored as REAL to accommodate:
--                    - Autism  : age in months (integer in source)
--                    - Diabetes: age in years  (integer in source)
--                    - Stroke  : age in years  (float in source, e.g. 0.08)
--
--   gender         Binary integer flag.
--                    1 = Male    0 = Female
--                  The CHECK constraint enforces domain integrity.
--                  The "Other" gender category (1 row in stroke source) was
--                  removed during ETL because it was too sparse to encode.
--
--   source_dataset Identifies which disease cohort the patient belongs to.
--                  Acts as a de-facto type discriminator in queries that
--                  join dim_patient back to multiple fact tables.
--                  Allowed values: 'autism', 'diabetes', 'stroke'
--
-- TOTAL ROWS LOADED: 11,704
--   autism   : 6,075
--   diabetes :   520
--   stroke   : 5,109
-- ----------------------------------------------------------------------------

CREATE TABLE dim_patient (
    patient_id     INTEGER  PRIMARY KEY AUTOINCREMENT,

    age            REAL     NOT NULL,

    gender         INTEGER  NOT NULL
                            CHECK (gender IN (0, 1)),
                            -- 0 = Female | 1 = Male

    source_dataset TEXT     NOT NULL
                            CHECK (source_dataset IN ('autism', 'diabetes', 'stroke'))
);


-- ============================================================================
-- SECTION 3 : FACT TABLES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- TABLE: fact_autism
--
-- SUBJECT AREA : Autism Spectrum Disorder (ASD) Screening
-- SOURCE FILE  : Autism_Screening_Data_Combined.csv
-- GRAIN        : One row per AQ-10 screening session per patient.
--
-- CLINICAL CONTEXT:
--   The AQ-10 (Autism-Spectrum Quotient, 10-item version) is a validated
--   behavioural screening questionnaire. Each item (a1–a10) is scored 0 or 1.
--   Patients scoring AQ >= 6 (aq_score >= 6) are typically referred for full
--   diagnostic assessment (Baron-Cohen et al., 2009).
--
-- COLUMNS:
--   fact_id    Surrogate PK for the fact record.
--   patient_id FK → dim_patient. ON DELETE CASCADE removes orphaned facts.
--   a1–a10     AQ-10 item responses (0 = non-autistic-like, 1 = autistic-like).
--              CHECK (col IN (0,1)) enforces the binary domain for all items.
--   aq_score   Derived total = SUM(a1..a10). Range 0–10. Engineered during ETL
--              to capture the standard clinical screening threshold in a single
--              column. CHECK ensures the value is always valid.
--   jaundice   History of jaundice at birth (0=No, 1=Yes).
--   family_asd Family history of ASD (0=No, 1=Yes).
--   label      Target variable: 1 = ASD Positive, 0 = ASD Negative.
--              Derived from the "Class" column (YES/NO) in the source file.
--
-- CLASS DISTRIBUTION (after ETL):
--   Positive (1): 1,804  (29.7%)
--   Negative (0): 4,271  (70.3%)
--   Imbalance ratio: ~1 : 2.4  (mild)
-- ----------------------------------------------------------------------------

CREATE TABLE fact_autism (
    fact_id    INTEGER  PRIMARY KEY AUTOINCREMENT,

    patient_id INTEGER  NOT NULL
                        REFERENCES dim_patient(patient_id)
                        ON DELETE CASCADE,

    -- AQ-10 Behavioural Screening Items (binary: 0 = no, 1 = yes)
    a1         INTEGER  NOT NULL  CHECK (a1  IN (0, 1)),
    a2         INTEGER  NOT NULL  CHECK (a2  IN (0, 1)),
    a3         INTEGER  NOT NULL  CHECK (a3  IN (0, 1)),
    a4         INTEGER  NOT NULL  CHECK (a4  IN (0, 1)),
    a5         INTEGER  NOT NULL  CHECK (a5  IN (0, 1)),
    a6         INTEGER  NOT NULL  CHECK (a6  IN (0, 1)),
    a7         INTEGER  NOT NULL  CHECK (a7  IN (0, 1)),
    a8         INTEGER  NOT NULL  CHECK (a8  IN (0, 1)),
    a9         INTEGER  NOT NULL  CHECK (a9  IN (0, 1)),
    a10        INTEGER  NOT NULL  CHECK (a10 IN (0, 1)),

    -- Derived clinical score (engineered feature: sum of a1..a10)
    -- Clinical referral threshold: aq_score >= 6
    aq_score   INTEGER  NOT NULL  CHECK (aq_score BETWEEN 0 AND 10),

    -- Risk factors
    jaundice   INTEGER  NOT NULL  CHECK (jaundice   IN (0, 1)),
    family_asd INTEGER  NOT NULL  CHECK (family_asd IN (0, 1)),

    -- Target variable
    label      INTEGER  NOT NULL  CHECK (label IN (0, 1))
    -- 1 = ASD Positive  |  0 = ASD Negative
);


-- ----------------------------------------------------------------------------
-- TABLE: fact_diabetes
--
-- SUBJECT AREA : Early-Stage Diabetes Risk Prediction
-- SOURCE FILE  : diabetes_data_upload.csv
-- GRAIN        : One clinical assessment record per patient.
--
-- CLINICAL CONTEXT:
--   Features are patient-reported binary symptoms characteristic of early-
--   stage Type 2 diabetes. The two most clinically discriminating symptoms
--   (polyuria and polydipsia) are classic hallmarks of hyperglycaemia.
--   Source: Islam et al. (2020), Sylhet Diabetes Hospital, Bangladesh.
--
-- COLUMNS:
--   fact_id … patient_id : as in fact_autism.
--
--   Symptom columns (all binary 0=No, 1=Yes):
--     polyuria            Excessive urination
--     polydipsia          Excessive thirst
--     sudden_weight_loss  Unexplained weight loss
--     weakness            Generalised weakness
--     polyphagia          Excessive hunger
--     genital_thrush      Genital yeast infection
--     visual_blurring     Blurred vision episodes
--     itching             Generalised pruritus
--     irritability        Mood changes / irritability
--     delayed_healing     Slow wound healing
--     partial_paresis     Partial muscle weakness
--     muscle_stiffness    Muscle stiffness
--     alopecia            Hair loss
--     obesity             Clinical obesity
--
--   label   Target: 1 = Diabetic (Positive), 0 = Non-diabetic (Negative)
--           Derived from the "class" column (Positive/Negative) in source.
--
-- NOTE: Column names in the source CSV contained spaces (e.g. "sudden weight
--       loss"). These were renamed to snake_case during ETL to comply with
--       SQL identifier conventions.
--
-- CLASS DISTRIBUTION:
--   Positive (1): 320  (61.5%)
--   Negative (0): 200  (38.5%)
--   Imbalance ratio: ~1 : 0.6  (positive is the majority class)
-- ----------------------------------------------------------------------------

CREATE TABLE fact_diabetes (
    fact_id            INTEGER  PRIMARY KEY AUTOINCREMENT,

    patient_id         INTEGER  NOT NULL
                                REFERENCES dim_patient(patient_id)
                                ON DELETE CASCADE,

    -- Clinical symptom flags (0 = absent, 1 = present)
    polyuria           INTEGER  NOT NULL  CHECK (polyuria           IN (0, 1)),
    polydipsia         INTEGER  NOT NULL  CHECK (polydipsia         IN (0, 1)),
    sudden_weight_loss INTEGER  NOT NULL  CHECK (sudden_weight_loss IN (0, 1)),
    weakness           INTEGER  NOT NULL  CHECK (weakness           IN (0, 1)),
    polyphagia         INTEGER  NOT NULL  CHECK (polyphagia         IN (0, 1)),
    genital_thrush     INTEGER  NOT NULL  CHECK (genital_thrush     IN (0, 1)),
    visual_blurring    INTEGER  NOT NULL  CHECK (visual_blurring    IN (0, 1)),
    itching            INTEGER  NOT NULL  CHECK (itching            IN (0, 1)),
    irritability       INTEGER  NOT NULL  CHECK (irritability       IN (0, 1)),
    delayed_healing    INTEGER  NOT NULL  CHECK (delayed_healing    IN (0, 1)),
    partial_paresis    INTEGER  NOT NULL  CHECK (partial_paresis    IN (0, 1)),
    muscle_stiffness   INTEGER  NOT NULL  CHECK (muscle_stiffness   IN (0, 1)),
    alopecia           INTEGER  NOT NULL  CHECK (alopecia           IN (0, 1)),
    obesity            INTEGER  NOT NULL  CHECK (obesity            IN (0, 1)),

    -- Target variable
    label              INTEGER  NOT NULL  CHECK (label IN (0, 1))
    -- 1 = Diabetic  |  0 = Non-diabetic
);


-- ----------------------------------------------------------------------------
-- TABLE: fact_stroke
--
-- SUBJECT AREA : Stroke Event Prediction
-- SOURCE FILE  : healthcare-dataset-stroke-data.csv
-- GRAIN        : One clinical record per patient (point-in-time snapshot).
--
-- CLINICAL CONTEXT:
--   Stroke is the second leading cause of death globally (WHO, 2021).
--   Early prediction using patient demographics and comorbidities enables
--   preventive intervention. This dataset presents a severe class imbalance
--   (4.87% stroke prevalence), reflecting the real-world clinical incidence
--   rate in hospital populations.
--
-- ETL TRANSFORMATIONS APPLIED:
--   - Dropped "id" column (meaningless surrogate from source).
--   - Removed 1 row where gender = 'Other' (too sparse to encode reliably).
--   - Imputed 201 missing "bmi" values using MEDIAN grouped by age bracket
--     (0–18, 19–40, 41–60, 61–80, 81+) to avoid systematic age-group bias
--     that a global median would introduce.
--   - Binary columns (gender, ever_married, residence_type) encoded to 0/1.
--   - work_type and smoking_status kept as TEXT (multi-category; one-hot
--     encoding deferred to the ML feature engineering phase).
--
-- COLUMNS:
--   hypertension      Diagnosed hypertension (0=No, 1=Yes)
--   heart_disease     History of heart disease (0=No, 1=Yes)
--   ever_married      Marital status (0=No, 1=Yes)
--   work_type         Employment category: 'Private' | 'Self-employed' |
--                       'Govt_job' | 'children' | 'Never_worked'
--   residence_type    Residential area (0=Rural, 1=Urban)
--   avg_glucose_level Fasting average blood glucose level (mg/dL)
--   bmi               Body Mass Index (kg/m²) — imputed where missing
--   smoking_status    'formerly smoked' | 'never smoked' |
--                       'smokes' | 'Unknown'
--   label             Target: 1 = Stroke occurred, 0 = No stroke
--
-- CLASS DISTRIBUTION:
--   Stroke     (1):   249  (4.87%)
--   No Stroke  (0): 4,860  (95.13%)
--   Imbalance ratio: ~1 : 19.5  (SEVERE — requires SMOTE in ML pipeline)
-- ----------------------------------------------------------------------------

CREATE TABLE fact_stroke (
    fact_id           INTEGER  PRIMARY KEY AUTOINCREMENT,

    patient_id        INTEGER  NOT NULL
                               REFERENCES dim_patient(patient_id)
                               ON DELETE CASCADE,

    -- Comorbidities (binary)
    hypertension      INTEGER  NOT NULL  CHECK (hypertension  IN (0, 1)),
    heart_disease     INTEGER  NOT NULL  CHECK (heart_disease IN (0, 1)),

    -- Demographics (binary-encoded)
    ever_married      INTEGER  NOT NULL  CHECK (ever_married  IN (0, 1)),
    -- 0 = Never married  |  1 = Ever married

    -- Categorical features (stored as TEXT; one-hot encoded in ML phase)
    work_type         TEXT     NOT NULL,
    -- Values: 'Private' | 'Self-employed' | 'Govt_job' | 'children' | 'Never_worked'

    residence_type    INTEGER  NOT NULL  CHECK (residence_type IN (0, 1)),
    -- 0 = Rural  |  1 = Urban

    -- Continuous clinical measurements
    avg_glucose_level REAL     NOT NULL,
    -- Blood glucose in mg/dL; range in dataset: 55.12 – 271.74

    bmi               REAL     NOT NULL,
    -- Body Mass Index in kg/m²; 201 values imputed from age-bracket medians

    smoking_status    TEXT     NOT NULL,
    -- Values: 'formerly smoked' | 'never smoked' | 'smokes' | 'Unknown'
    -- NOTE: 'Unknown' is a valid clinical category (not a missing value);
    --       it represents patients for whom smoking history is unavailable.

    -- Target variable
    label             INTEGER  NOT NULL  CHECK (label IN (0, 1))
    -- 1 = Stroke occurred  |  0 = No stroke
);


-- ============================================================================
-- SECTION 4 : INDEXES
--
-- Purpose: Accelerate the most common query patterns in the analytical layer.
--
-- Index strategy:
--   - FK columns (patient_id in each fact table): required for JOIN performance.
--   - label columns: frequently filtered in GROUP BY and WHERE clauses.
--   - dim_patient.source_dataset: used in every cross-disease query.
--   - dim_patient.gender: used in demographic segmentation queries.
--
-- SQLite creates a B-tree index by default. These indexes are not unique
-- (UNIQUE keyword omitted intentionally — FK columns are not PK in fact tables).
-- ============================================================================

-- Fact table FK indexes (accelerate JOIN on patient_id)
CREATE INDEX idx_autism_patient    ON fact_autism   (patient_id);
CREATE INDEX idx_diabetes_patient  ON fact_diabetes (patient_id);
CREATE INDEX idx_stroke_patient    ON fact_stroke   (patient_id);

-- Label indexes (accelerate WHERE label = 1 / GROUP BY label)
CREATE INDEX idx_autism_label      ON fact_autism   (label);
CREATE INDEX idx_diabetes_label    ON fact_diabetes (label);
CREATE INDEX idx_stroke_label      ON fact_stroke   (label);

-- Dimension indexes (accelerate cross-disease and demographic queries)
CREATE INDEX idx_patient_source    ON dim_patient   (source_dataset);
CREATE INDEX idx_patient_gender    ON dim_patient   (gender);


-- ============================================================================
-- SECTION 5 : ANALYTICAL VIEWS
--
-- Views serve as the semantic layer of the warehouse, abstracting raw integer
-- codes back into human-readable labels (Male/Female, Yes/No, Urban/Rural).
-- Downstream consumers (Jupyter notebooks, dashboards) query views rather
-- than raw tables, insulating them from schema changes.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- VIEW: vw_autism_full
--
-- Purpose : Full denormalised row for each autism screening record.
--           Joins dim_patient demographics with all fact_autism columns.
--           Binary codes for gender, jaundice, and family_asd are decoded
--           to readable strings via CASE expressions.
-- Use     : SELECT * FROM vw_autism_full WHERE asd_diagnosis = 1;
-- ----------------------------------------------------------------------------

CREATE VIEW vw_autism_full AS
SELECT
    dp.patient_id,
    dp.age,
    CASE dp.gender
        WHEN 1 THEN 'Male'
        ELSE        'Female'
    END                         AS gender,
    dp.source_dataset,
    fa.a1,
    fa.a2,
    fa.a3,
    fa.a4,
    fa.a5,
    fa.a6,
    fa.a7,
    fa.a8,
    fa.a9,
    fa.a10,
    fa.aq_score,
    CASE fa.jaundice
        WHEN 1 THEN 'Yes'
        ELSE        'No'
    END                         AS jaundice,
    CASE fa.family_asd
        WHEN 1 THEN 'Yes'
        ELSE        'No'
    END                         AS family_asd,
    fa.label                    AS asd_diagnosis
FROM  fact_autism fa
JOIN  dim_patient dp
    ON fa.patient_id = dp.patient_id;


-- ----------------------------------------------------------------------------
-- VIEW: vw_diabetes_full
--
-- Purpose : Full denormalised row for each diabetes risk assessment.
--           Gender is decoded; all 14 symptom columns are left as integers
--           (0/1) for direct use in pivot/aggregation queries.
-- Use     : SELECT gender, AVG(polyuria) FROM vw_diabetes_full GROUP BY gender;
-- ----------------------------------------------------------------------------

CREATE VIEW vw_diabetes_full AS
SELECT
    dp.patient_id,
    dp.age,
    CASE dp.gender
        WHEN 1 THEN 'Male'
        ELSE        'Female'
    END                         AS gender,
    dp.source_dataset,
    fd.polyuria,
    fd.polydipsia,
    fd.sudden_weight_loss,
    fd.weakness,
    fd.polyphagia,
    fd.genital_thrush,
    fd.visual_blurring,
    fd.itching,
    fd.irritability,
    fd.delayed_healing,
    fd.partial_paresis,
    fd.muscle_stiffness,
    fd.alopecia,
    fd.obesity,
    fd.label                    AS diabetes_diagnosis
FROM  fact_diabetes fd
JOIN  dim_patient dp
    ON fd.patient_id = dp.patient_id;


-- ----------------------------------------------------------------------------
-- VIEW: vw_stroke_full
--
-- Purpose : Full denormalised row for each stroke prediction record.
--           Decodes gender, ever_married, and residence_type to text.
--           work_type and smoking_status are already stored as TEXT.
-- Use     : SELECT * FROM vw_stroke_full WHERE stroke_diagnosis = 1;
-- ----------------------------------------------------------------------------

CREATE VIEW vw_stroke_full AS
SELECT
    dp.patient_id,
    dp.age,
    CASE dp.gender
        WHEN 1 THEN 'Male'
        ELSE        'Female'
    END                         AS gender,
    dp.source_dataset,
    fs.hypertension,
    fs.heart_disease,
    CASE fs.ever_married
        WHEN 1 THEN 'Yes'
        ELSE        'No'
    END                         AS ever_married,
    fs.work_type,
    CASE fs.residence_type
        WHEN 1 THEN 'Urban'
        ELSE        'Rural'
    END                         AS residence_type,
    fs.avg_glucose_level,
    fs.bmi,
    fs.smoking_status,
    fs.label                    AS stroke_diagnosis
FROM  fact_stroke fs
JOIN  dim_patient dp
    ON fs.patient_id = dp.patient_id;


-- ----------------------------------------------------------------------------
-- VIEW: vw_disease_summary
--
-- Purpose : High-level OLAP summary view — one row per disease cohort.
--           Aggregates total patients, positive case count, positive
--           percentage, average age, and percentage male.
--
--           Uses a LEFT JOIN triangle pattern to reach all three fact tables
--           from the shared dim_patient dimension, with CASE expressions to
--           select the correct label column per source_dataset.
--
-- Use     : SELECT * FROM vw_disease_summary ORDER BY disease;
-- ----------------------------------------------------------------------------

CREATE VIEW vw_disease_summary AS
SELECT
    dp.source_dataset                                        AS disease,
    COUNT(*)                                                 AS total_patients,

    SUM(
        CASE dp.source_dataset
            WHEN 'autism'   THEN fa.label
            WHEN 'diabetes' THEN fd.label
            WHEN 'stroke'   THEN fs.label
        END
    )                                                        AS positive_cases,

    ROUND(
        100.0 * SUM(
            CASE dp.source_dataset
                WHEN 'autism'   THEN fa.label
                WHEN 'diabetes' THEN fd.label
                WHEN 'stroke'   THEN fs.label
            END
        ) / COUNT(*),
    2)                                                       AS positive_pct,

    ROUND(AVG(dp.age), 1)                                    AS avg_age,

    ROUND(100.0 * SUM(dp.gender) / COUNT(*), 1)              AS pct_male

FROM  dim_patient dp
LEFT JOIN fact_autism   fa
    ON  dp.patient_id     = fa.patient_id
    AND dp.source_dataset = 'autism'
LEFT JOIN fact_diabetes fd
    ON  dp.patient_id     = fd.patient_id
    AND dp.source_dataset = 'diabetes'
LEFT JOIN fact_stroke   fs
    ON  dp.patient_id     = fs.patient_id
    AND dp.source_dataset = 'stroke'
GROUP BY dp.source_dataset;


-- ============================================================================
-- SECTION 6 : DATA DICTIONARY SUMMARY (as SQL comments)
--
-- dim_patient columns
--   patient_id      INTEGER   Surrogate PK. Auto-incremented. Not meaningful.
--   age             REAL      Age of patient (months for autism, years for rest).
--   gender          INTEGER   0 = Female, 1 = Male.
--   source_dataset  TEXT      'autism' | 'diabetes' | 'stroke'.
--
-- fact_autism additional columns
--   a1..a10         INTEGER   AQ-10 behavioural items. 0=non-ASD-like, 1=ASD-like.
--   aq_score        INTEGER   Clinical sum score (0–10). Threshold >= 6 = referral.
--   jaundice        INTEGER   Neonatal jaundice history. 0=No, 1=Yes.
--   family_asd      INTEGER   Family ASD history. 0=No, 1=Yes.
--   label           INTEGER   ASD diagnosis. 1=Positive, 0=Negative.
--
-- fact_diabetes additional columns
--   polyuria..obesity  INTEGER  14 binary symptom flags. 0=Absent, 1=Present.
--   label              INTEGER  Diabetes diagnosis. 1=Positive, 0=Negative.
--
-- fact_stroke additional columns
--   hypertension      INTEGER  Diagnosed hypertension. 0=No, 1=Yes.
--   heart_disease     INTEGER  History of heart disease. 0=No, 1=Yes.
--   ever_married      INTEGER  Marital history. 0=No, 1=Yes.
--   work_type         TEXT     Employment category (5 values).
--   residence_type    INTEGER  0=Rural, 1=Urban.
--   avg_glucose_level REAL     Blood glucose mg/dL.
--   bmi               REAL     Body Mass Index kg/m². Partially imputed.
--   smoking_status    TEXT     Smoking history (4 categories incl. Unknown).
--   label             INTEGER  Stroke event. 1=Stroke, 0=No stroke.
-- ============================================================================

-- End of schema.sql
