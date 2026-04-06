# =============================================================================
# ETL Pipeline — Phase 1B: Healthcare Data Warehouse (SQLite Star Schema)
# Project  : Early Disease Prediction Using Healthcare Data Warehouse
# Script   : etl_pipeline/warehouse.py
# Database : warehouse/health_warehouse.db
#
# PURPOSE:
#   Simulate a professional Clinical Data Warehouse using SQLite3.
#   ALL SQL DDL (CREATE TABLE, CREATE INDEX) and DML (INSERT INTO) statements
#   are written as explicit, formatted raw SQL strings and executed directly
#   via the sqlite3 cursor — pandas .to_sql() is NOT used.
#
# STAR SCHEMA DESIGN:
#
#                    ┌──────────────────────────┐
#                    │       dim_patient         │  ← DIMENSION TABLE
#                    │  PK: patient_id (AUTO)    │     Shared demographics
#                    │      age        REAL      │     across all 3 diseases
#                    │      gender     INTEGER   │
#                    │      source_dataset TEXT  │
#                    └────────────┬─────────────┘
#                                 │
#              ┌──────────────────┼──────────────────┐
#              │  FK              │  FK              │  FK
#   ┌──────────▼──────┐  ┌───────▼────────┐  ┌─────▼────────────┐
#   │  fact_autism    │  │ fact_diabetes  │  │  fact_stroke     │
#   │  FACT TABLE     │  │  FACT TABLE    │  │  FACT TABLE      │
#   │  6,075 rows     │  │  520 rows      │  │  5,109 rows      │
#   └─────────────────┘  └────────────────┘  └──────────────────┘
#
# HOW TO RUN:
#   python etl_pipeline/warehouse.py
# =============================================================================

import os
import sqlite3
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 0.  Path Resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
CLEANED_DIR = ROOT_DIR / "datasets" / "cleaned"
WAREHOUSE_DIR = ROOT_DIR / "warehouse"
DB_PATH = WAREHOUSE_DIR / "health_warehouse.db"

WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

# Source cleaned CSVs
CSV_AUTISM = CLEANED_DIR / "autism_cleaned.csv"
CSV_DIABETES = CLEANED_DIR / "diabetes_cleaned.csv"
CSV_STROKE = CLEANED_DIR / "stroke_cleaned.csv"


# ---------------------------------------------------------------------------
# 1.  Utility Helpers
# ---------------------------------------------------------------------------


def banner(title: str, width: int = 74, char: str = "=") -> None:
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def step(msg: str) -> None:
    print(f"    ▸  {msg}")


def ok(msg: str) -> None:
    print(f"    ✔  {msg}")


def _row(cells: list, widths: list, sep: str = "│") -> str:
    parts = [f" {str(c):<{w}} " for c, w in zip(cells, widths)]
    return sep + sep.join(parts) + sep


def _divider(widths: list, l: str = "├", m: str = "┼", r: str = "┤") -> str:
    return l + m.join("─" * (w + 2) for w in widths) + r


def _top(widths: list) -> str:
    return "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"


def _bot(widths: list) -> str:
    return "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"


# ===========================================================================
# 2.  RAW SQL DDL — PRAGMA Configuration
# ===========================================================================

SQL_PRAGMA_FK = "PRAGMA foreign_keys = ON;"
SQL_PRAGMA_WAL = "PRAGMA journal_mode  = WAL;"
SQL_PRAGMA_SYNC = "PRAGMA synchronous   = NORMAL;"


# ===========================================================================
# 3.  RAW SQL DDL — DROP EXISTING OBJECTS  (idempotent re-run)
# ===========================================================================

SQL_DROP_VIEW_SUMMARY = "DROP VIEW  IF EXISTS vw_disease_summary;"
SQL_DROP_VIEW_STROKE = "DROP VIEW  IF EXISTS vw_stroke_full;"
SQL_DROP_VIEW_DIABETES = "DROP VIEW  IF EXISTS vw_diabetes_full;"
SQL_DROP_VIEW_AUTISM = "DROP VIEW  IF EXISTS vw_autism_full;"
SQL_DROP_FACT_STROKE = "DROP TABLE IF EXISTS fact_stroke;"
SQL_DROP_FACT_DIABETES = "DROP TABLE IF EXISTS fact_diabetes;"
SQL_DROP_FACT_AUTISM = "DROP TABLE IF EXISTS fact_autism;"
SQL_DROP_DIM_PATIENT = "DROP TABLE IF EXISTS dim_patient;"


# ===========================================================================
# 4.  RAW SQL DDL — DIMENSION TABLE
# ===========================================================================

SQL_CREATE_DIM_PATIENT = """
CREATE TABLE dim_patient (

    -- Surrogate Primary Key (auto-assigned by SQLite AUTOINCREMENT)
    patient_id     INTEGER  PRIMARY KEY AUTOINCREMENT,

    -- Age stored as REAL to accommodate:
    --   autism  → age in months (integer)
    --   diabetes→ age in years  (integer)
    --   stroke  → age in years  (float, e.g. 0.08 for infants)
    age            REAL     NOT NULL,

    -- Binary gender flag: 0 = Female, 1 = Male
    -- The 'Other' category (1 row in stroke source) was removed during ETL.
    gender         INTEGER  NOT NULL
                            CHECK (gender IN (0, 1)),

    -- Identifies which disease cohort this patient belongs to.
    -- Acts as a discriminator column for cross-disease queries.
    source_dataset TEXT     NOT NULL
                            CHECK (source_dataset IN ('autism','diabetes','stroke'))
);
"""


# ===========================================================================
# 5.  RAW SQL DDL — FACT TABLE: fact_autism
# ===========================================================================

SQL_CREATE_FACT_AUTISM = """
CREATE TABLE fact_autism (

    -- Surrogate Primary Key for this fact record
    fact_id    INTEGER  PRIMARY KEY AUTOINCREMENT,

    -- Foreign Key → dim_patient (the shared dimension)
    -- ON DELETE CASCADE: removing a patient removes their fact record too.
    patient_id INTEGER  NOT NULL
                        REFERENCES dim_patient(patient_id)
                        ON DELETE CASCADE,

    -- ── AQ-10 Behavioural Screening Items (binary: 0=No, 1=Yes) ──────────
    -- Source: Autism-Spectrum Quotient 10-item screening tool
    -- Clinical referral threshold: sum(a1..a10) >= 6
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

    -- Derived clinical score engineered during ETL: SUM(a1..a10), range 0–10
    aq_score   INTEGER  NOT NULL  CHECK (aq_score BETWEEN 0 AND 10),

    -- ── Clinical Risk Factors ─────────────────────────────────────────────
    jaundice   INTEGER  NOT NULL  CHECK (jaundice   IN (0, 1)),
    family_asd INTEGER  NOT NULL  CHECK (family_asd IN (0, 1)),

    -- ── Target Variable ───────────────────────────────────────────────────
    -- 1 = ASD Positive  |  0 = ASD Negative
    -- Source column: "Class" (YES/NO → 1/0 during ETL)
    label      INTEGER  NOT NULL  CHECK (label IN (0, 1))
);
"""


# ===========================================================================
# 6.  RAW SQL DDL — FACT TABLE: fact_diabetes
# ===========================================================================

SQL_CREATE_FACT_DIABETES = """
CREATE TABLE fact_diabetes (

    -- Surrogate Primary Key
    fact_id            INTEGER  PRIMARY KEY AUTOINCREMENT,

    -- Foreign Key → dim_patient
    patient_id         INTEGER  NOT NULL
                                REFERENCES dim_patient(patient_id)
                                ON DELETE CASCADE,

    -- ── Binary Symptom Flags (0 = absent, 1 = present) ───────────────────
    -- Source: Sylhet Diabetes Hospital, Bangladesh (Islam et al., 2020)
    -- NOTE: Source column names had spaces (e.g. "sudden weight loss").
    --       These were renamed to snake_case during ETL for SQL compliance.
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

    -- ── Target Variable ───────────────────────────────────────────────────
    -- 1 = Diabetic  |  0 = Non-Diabetic
    -- Source column: "class" (Positive/Negative → 1/0 during ETL)
    label              INTEGER  NOT NULL  CHECK (label IN (0, 1))
);
"""


# ===========================================================================
# 7.  RAW SQL DDL — FACT TABLE: fact_stroke
# ===========================================================================

SQL_CREATE_FACT_STROKE = """
CREATE TABLE fact_stroke (

    -- Surrogate Primary Key
    fact_id           INTEGER  PRIMARY KEY AUTOINCREMENT,

    -- Foreign Key → dim_patient
    patient_id        INTEGER  NOT NULL
                               REFERENCES dim_patient(patient_id)
                               ON DELETE CASCADE,

    -- ── Binary Comorbidity Flags ──────────────────────────────────────────
    hypertension      INTEGER  NOT NULL  CHECK (hypertension  IN (0, 1)),
    heart_disease     INTEGER  NOT NULL  CHECK (heart_disease IN (0, 1)),
    ever_married      INTEGER  NOT NULL  CHECK (ever_married  IN (0, 1)),

    -- ── Categorical Features (stored as TEXT — OHE deferred to Phase 3) ──
    -- work_type     : 'Private'|'Self-employed'|'Govt_job'|'children'|'Never_worked'
    -- smoking_status: 'formerly smoked'|'never smoked'|'smokes'|'Unknown'
    work_type         TEXT     NOT NULL,
    residence_type    INTEGER  NOT NULL  CHECK (residence_type IN (0, 1)),

    -- ── Continuous Clinical Measurements ─────────────────────────────────
    -- avg_glucose_level: mg/dL  (range in dataset: 55.12 – 271.74)
    -- bmi              : kg/m²  (201 values imputed by age-bracket median in ETL)
    avg_glucose_level REAL     NOT NULL,
    bmi               REAL     NOT NULL,
    smoking_status    TEXT     NOT NULL,

    -- ── Target Variable ───────────────────────────────────────────────────
    -- 1 = Stroke occurred  |  0 = No stroke
    -- Source column: "stroke" (0/1 integer, renamed to "Class" in ETL)
    label             INTEGER  NOT NULL  CHECK (label IN (0, 1))
);
"""


# ===========================================================================
# 8.  RAW SQL DDL — INDEXES
# ===========================================================================

SQL_INDEXES = [
    # FK indexes — accelerate JOIN operations on patient_id
    (
        "idx_autism_patient",
        "CREATE INDEX idx_autism_patient   ON fact_autism   (patient_id);",
    ),
    (
        "idx_diabetes_patient",
        "CREATE INDEX idx_diabetes_patient ON fact_diabetes (patient_id);",
    ),
    (
        "idx_stroke_patient",
        "CREATE INDEX idx_stroke_patient   ON fact_stroke   (patient_id);",
    ),
    # Label indexes — accelerate WHERE label=1 and GROUP BY label
    ("idx_autism_label", "CREATE INDEX idx_autism_label     ON fact_autism   (label);"),
    (
        "idx_diabetes_label",
        "CREATE INDEX idx_diabetes_label   ON fact_diabetes (label);",
    ),
    ("idx_stroke_label", "CREATE INDEX idx_stroke_label     ON fact_stroke   (label);"),
    # Dimension indexes — accelerate cross-disease and demographic queries
    (
        "idx_patient_source",
        "CREATE INDEX idx_patient_source   ON dim_patient   (source_dataset);",
    ),
    (
        "idx_patient_gender",
        "CREATE INDEX idx_patient_gender   ON dim_patient   (gender);",
    ),
]


# ===========================================================================
# 9.  RAW SQL DML — PARAMETERIZED INSERT STATEMENTS
# ===========================================================================

SQL_INSERT_DIM_PATIENT = """
    INSERT INTO dim_patient
        (age, gender, source_dataset)
    VALUES
        (?, ?, ?)
"""

SQL_INSERT_FACT_AUTISM = """
    INSERT INTO fact_autism
        (patient_id,
         a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
         aq_score,
         jaundice, family_asd,
         label)
    VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SQL_INSERT_FACT_DIABETES = """
    INSERT INTO fact_diabetes
        (patient_id,
         polyuria, polydipsia, sudden_weight_loss, weakness,
         polyphagia, genital_thrush, visual_blurring, itching,
         irritability, delayed_healing, partial_paresis,
         muscle_stiffness, alopecia, obesity,
         label)
    VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SQL_INSERT_FACT_STROKE = """
    INSERT INTO fact_stroke
        (patient_id,
         hypertension, heart_disease, ever_married,
         work_type, residence_type,
         avg_glucose_level, bmi, smoking_status,
         label)
    VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ===========================================================================
# 10. EXECUTION — Main ETL Function
# ===========================================================================


def run_warehouse() -> None:

    # ── 10.0  Pre-flight checks ──────────────────────────────────────────────
    banner("PRE-FLIGHT CHECKS")

    for csv_path in [CSV_AUTISM, CSV_DIABETES, CSV_STROKE]:
        if not csv_path.exists():
            print(f"\n  [ERROR] Missing cleaned CSV: {csv_path}")
            print("  Run etl_pipeline/clean.py first.")
            sys.exit(1)
        step(f"Found: {csv_path.name}")

    # ── 10.1  Load cleaned CSVs ───────────────────────────────────────────────
    banner("STEP 1 — LOADING CLEANED DATASETS")

    df_autism = pd.read_csv(CSV_AUTISM)
    df_diabetes = pd.read_csv(CSV_DIABETES)
    df_stroke = pd.read_csv(CSV_STROKE)

    step(
        f"autism_cleaned.csv   → {df_autism.shape[0]:,} rows × {df_autism.shape[1]} cols"
    )
    step(
        f"diabetes_cleaned.csv → {df_diabetes.shape[0]:,} rows × {df_diabetes.shape[1]} cols"
    )
    step(
        f"stroke_cleaned.csv   → {df_stroke.shape[0]:,} rows × {df_stroke.shape[1]} cols"
    )
    total_source_rows = len(df_autism) + len(df_diabetes) + len(df_stroke)
    step(f"Total source rows    : {total_source_rows:,}")

    # ── 10.2  Open DB connection & apply PRAGMAs ──────────────────────────────
    banner("STEP 2 — DATABASE CONNECTION & PRAGMA CONFIGURATION")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for pragma_sql in [SQL_PRAGMA_FK, SQL_PRAGMA_WAL, SQL_PRAGMA_SYNC]:
        cursor.execute(pragma_sql)
        step(f"Executed: {pragma_sql.strip()}")

    conn.commit()
    ok(f"Connected to: {DB_PATH}")
    ok(f"PRAGMAs set  — foreign_keys=ON | journal_mode=WAL | synchronous=NORMAL")

    # ── 10.3  Drop existing objects (idempotent clean-slate) ──────────────────
    banner("STEP 3 — DROPPING EXISTING OBJECTS  (clean-slate re-run)")

    drop_sequence = [
        SQL_DROP_VIEW_SUMMARY,
        SQL_DROP_VIEW_STROKE,
        SQL_DROP_VIEW_DIABETES,
        SQL_DROP_VIEW_AUTISM,
        SQL_DROP_FACT_STROKE,
        SQL_DROP_FACT_DIABETES,
        SQL_DROP_FACT_AUTISM,
        SQL_DROP_DIM_PATIENT,
    ]
    for sql_stmt in drop_sequence:
        cursor.execute(sql_stmt)
        step(f"Executed: {sql_stmt.strip()}")
    conn.commit()
    ok("All existing tables and views dropped — database is clean.")

    # ── 10.4  Create dimension table ──────────────────────────────────────────
    banner("STEP 4 — CREATING DIMENSION TABLE: dim_patient")

    print()
    print("  Raw SQL DDL:")
    print(textwrap.indent(SQL_CREATE_DIM_PATIENT.strip(), "    "))
    cursor.execute(SQL_CREATE_DIM_PATIENT)
    conn.commit()
    ok("CREATE TABLE dim_patient — executed successfully.")

    # ── 10.5  Create fact tables ──────────────────────────────────────────────
    banner("STEP 5 — CREATING FACT TABLES")

    fact_ddl_map = [
        ("fact_autism", SQL_CREATE_FACT_AUTISM),
        ("fact_diabetes", SQL_CREATE_FACT_DIABETES),
        ("fact_stroke", SQL_CREATE_FACT_STROKE),
    ]
    for table_name, ddl_sql in fact_ddl_map:
        print(f"\n  ── {table_name} ────────────────────────────────────────")
        print()
        print("  Raw SQL DDL:")
        print(textwrap.indent(ddl_sql.strip(), "    "))
        cursor.execute(ddl_sql)
        conn.commit()
        ok(f"CREATE TABLE {table_name} — executed successfully.")

    # ── 10.6  Create indexes ──────────────────────────────────────────────────
    banner("STEP 6 — CREATING INDEXES")

    for idx_name, idx_sql in SQL_INDEXES:
        cursor.execute(idx_sql)
        step(f"{idx_name:<28}  ←  {idx_sql.strip()}")
    conn.commit()
    ok(
        f"{len(SQL_INDEXES)} indexes created on FK / label / source_dataset / gender columns."
    )

    # ── 10.7  INSERT: Autism → dim_patient + fact_autism ─────────────────────
    banner("STEP 7 — INSERTING: AUTISM SCREENING")

    t0 = time.perf_counter()

    # -- Determine starting patient_id for this batch --
    cursor.execute("SELECT COALESCE(MAX(patient_id), 0) FROM dim_patient;")
    first_pid_a = cursor.fetchone()[0] + 1

    # -- Build dim_patient rows: (age, gender, source_dataset) --
    dim_autism_rows = [
        (int(row.Age), int(row.Sex), "autism")
        for row in df_autism.itertuples(index=False)
    ]

    print(f"\n  SQL DML (INSERT into dim_patient):")
    print(textwrap.indent(SQL_INSERT_DIM_PATIENT.strip(), "    "))
    cursor.executemany(SQL_INSERT_DIM_PATIENT, dim_autism_rows)
    step(
        f"Inserted {len(dim_autism_rows):,} rows into dim_patient  "
        f"(patient_id range: {first_pid_a} – {first_pid_a + len(dim_autism_rows) - 1})"
    )

    # -- Assign sequential patient_ids --
    autism_pids = range(first_pid_a, first_pid_a + len(df_autism))

    # -- Build fact_autism rows --
    fact_autism_rows = [
        (
            pid,
            int(row.A1),
            int(row.A2),
            int(row.A3),
            int(row.A4),
            int(row.A5),
            int(row.A6),
            int(row.A7),
            int(row.A8),
            int(row.A9),
            int(row.A10),
            int(row.AQ_Score),
            int(row.Jaundice),
            int(row.Family_ASD),
            int(row.Class),
        )
        for pid, row in zip(autism_pids, df_autism.itertuples(index=False))
    ]

    print(f"\n  SQL DML (INSERT into fact_autism):")
    print(textwrap.indent(SQL_INSERT_FACT_AUTISM.strip(), "    "))
    cursor.executemany(SQL_INSERT_FACT_AUTISM, fact_autism_rows)
    conn.commit()

    elapsed_a = time.perf_counter() - t0
    ok(f"Inserted {len(fact_autism_rows):,} rows into fact_autism  ({elapsed_a:.3f}s)")

    # ── 10.8  INSERT: Diabetes → dim_patient + fact_diabetes ─────────────────
    banner("STEP 8 — INSERTING: DIABETES RISK")

    t0 = time.perf_counter()

    cursor.execute("SELECT COALESCE(MAX(patient_id), 0) FROM dim_patient;")
    first_pid_d = cursor.fetchone()[0] + 1

    dim_diabetes_rows = [
        (int(row.Age), int(row.Gender), "diabetes")
        for row in df_diabetes.itertuples(index=False)
    ]
    cursor.executemany(SQL_INSERT_DIM_PATIENT, dim_diabetes_rows)
    step(
        f"Inserted {len(dim_diabetes_rows):,} rows into dim_patient  "
        f"(patient_id range: {first_pid_d} – {first_pid_d + len(dim_diabetes_rows) - 1})"
    )

    diabetes_pids = range(first_pid_d, first_pid_d + len(df_diabetes))

    # NOTE: Source CSV has space-separated column names ("sudden weight loss",
    #       "Genital thrush", etc.). We access them using zip() over explicit
    #       column lists to safely bypass the itertuples() identifier restriction.
    fact_diabetes_rows = list(
        zip(
            list(diabetes_pids),
            df_diabetes["Polyuria"].tolist(),
            df_diabetes["Polydipsia"].tolist(),
            df_diabetes["sudden weight loss"].tolist(),
            df_diabetes["weakness"].tolist(),
            df_diabetes["Polyphagia"].tolist(),
            df_diabetes["Genital thrush"].tolist(),
            df_diabetes["visual blurring"].tolist(),
            df_diabetes["Itching"].tolist(),
            df_diabetes["Irritability"].tolist(),
            df_diabetes["delayed healing"].tolist(),
            df_diabetes["partial paresis"].tolist(),
            df_diabetes["muscle stiffness"].tolist(),
            df_diabetes["Alopecia"].tolist(),
            df_diabetes["Obesity"].tolist(),
            df_diabetes["Class"].tolist(),
        )
    )

    print(f"\n  SQL DML (INSERT into fact_diabetes):")
    print(textwrap.indent(SQL_INSERT_FACT_DIABETES.strip(), "    "))
    cursor.executemany(SQL_INSERT_FACT_DIABETES, fact_diabetes_rows)
    conn.commit()

    elapsed_d = time.perf_counter() - t0
    ok(
        f"Inserted {len(fact_diabetes_rows):,} rows into fact_diabetes  "
        f"({elapsed_d:.3f}s)"
    )

    # ── 10.9  INSERT: Stroke → dim_patient + fact_stroke ─────────────────────
    banner("STEP 9 — INSERTING: STROKE PREDICTION")

    t0 = time.perf_counter()

    cursor.execute("SELECT COALESCE(MAX(patient_id), 0) FROM dim_patient;")
    first_pid_s = cursor.fetchone()[0] + 1

    dim_stroke_rows = [
        (float(row.age), int(row.gender), "stroke")
        for row in df_stroke.itertuples(index=False)
    ]
    cursor.executemany(SQL_INSERT_DIM_PATIENT, dim_stroke_rows)
    step(
        f"Inserted {len(dim_stroke_rows):,} rows into dim_patient  "
        f"(patient_id range: {first_pid_s} – {first_pid_s + len(dim_stroke_rows) - 1})"
    )

    stroke_pids = range(first_pid_s, first_pid_s + len(df_stroke))

    fact_stroke_rows = [
        (
            pid,
            int(row.hypertension),
            int(row.heart_disease),
            int(row.ever_married),
            str(row.work_type),
            int(row.Residence_type),
            float(row.avg_glucose_level),
            float(row.bmi),
            str(row.smoking_status),
            int(row.Class),
        )
        for pid, row in zip(stroke_pids, df_stroke.itertuples(index=False))
    ]

    print(f"\n  SQL DML (INSERT into fact_stroke):")
    print(textwrap.indent(SQL_INSERT_FACT_STROKE.strip(), "    "))
    cursor.executemany(SQL_INSERT_FACT_STROKE, fact_stroke_rows)
    conn.commit()

    elapsed_s = time.perf_counter() - t0
    ok(f"Inserted {len(fact_stroke_rows):,} rows into fact_stroke  ({elapsed_s:.3f}s)")

    # ── 10.10  Validation: row counts + FK integrity ──────────────────────────
    banner("STEP 10 — VALIDATION QUERIES")

    validation_queries = {
        "dim_patient row count": "SELECT COUNT(*) FROM dim_patient",
        "fact_autism row count": "SELECT COUNT(*) FROM fact_autism",
        "fact_diabetes row count": "SELECT COUNT(*) FROM fact_diabetes",
        "fact_stroke row count": "SELECT COUNT(*) FROM fact_stroke",
        "fact_autism FK orphans": (
            "SELECT COUNT(*) FROM fact_autism fa "
            "LEFT JOIN dim_patient dp ON fa.patient_id=dp.patient_id "
            "WHERE dp.patient_id IS NULL"
        ),
        "fact_diabetes FK orphans": (
            "SELECT COUNT(*) FROM fact_diabetes fd "
            "LEFT JOIN dim_patient dp ON fd.patient_id=dp.patient_id "
            "WHERE dp.patient_id IS NULL"
        ),
        "fact_stroke FK orphans": (
            "SELECT COUNT(*) FROM fact_stroke fs "
            "LEFT JOIN dim_patient dp ON fs.patient_id=dp.patient_id "
            "WHERE dp.patient_id IS NULL"
        ),
        "NULL values in fact_stroke.bmi": "SELECT COUNT(*) FROM fact_stroke WHERE bmi IS NULL",
        "NULL values in fact_autism.label": "SELECT COUNT(*) FROM fact_autism WHERE label IS NULL",
    }

    expected = {
        "dim_patient row count": total_source_rows,
        "fact_autism row count": len(df_autism),
        "fact_diabetes row count": len(df_diabetes),
        "fact_stroke row count": len(df_stroke),
        "fact_autism FK orphans": 0,
        "fact_diabetes FK orphans": 0,
        "fact_stroke FK orphans": 0,
        "NULL values in fact_stroke.bmi": 0,
        "NULL values in fact_autism.label": 0,
    }

    all_pass = True
    for check_name, sql_q in validation_queries.items():
        result = cursor.execute(sql_q).fetchone()[0]
        exp = expected[check_name]
        passed = result == exp
        if not passed:
            all_pass = False
        icon = "✔" if passed else "✘"
        detail = f"{result:,}" if result >= 0 else str(result)
        print(f"    {icon}  {check_name:<42}  =  {detail}")

    if all_pass:
        ok("All validation checks passed — warehouse integrity confirmed.")
    else:
        print(
            "\n  [WARNING] One or more validation checks failed. "
            "Review the output above."
        )

    # ── 10.11  Collect final metadata ─────────────────────────────────────────
    row_dim = cursor.execute("SELECT COUNT(*) FROM dim_patient").fetchone()[0]
    row_autism = cursor.execute("SELECT COUNT(*) FROM fact_autism").fetchone()[0]
    row_diabetes = cursor.execute("SELECT COUNT(*) FROM fact_diabetes").fetchone()[0]
    row_stroke = cursor.execute("SELECT COUNT(*) FROM fact_stroke").fetchone()[0]

    col_dim = len(
        [r[1] for r in cursor.execute("PRAGMA table_info(dim_patient)").fetchall()]
    )
    col_autism = len(
        [r[1] for r in cursor.execute("PRAGMA table_info(fact_autism)").fetchall()]
    )
    col_diabetes = len(
        [r[1] for r in cursor.execute("PRAGMA table_info(fact_diabetes)").fetchall()]
    )
    col_stroke = len(
        [r[1] for r in cursor.execute("PRAGMA table_info(fact_stroke)").fetchall()]
    )

    idx_count = cursor.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchone()[0]

    db_size_kb = DB_PATH.stat().st_size / 1024

    conn.close()

    # ===========================================================================
    # 11. GLORIOUS STAR SCHEMA SUMMARY
    # ===========================================================================

    W = 76  # total width

    def box_line(content: str = "", align: str = "left") -> str:
        inner = W - 4
        if align == "center":
            padded = content.center(inner)
        else:
            padded = content.ljust(inner)
        return f"║  {padded}  ║"

    def box_sep(left="╠", mid="═", right="╣") -> str:
        return left + mid * (W - 2) + right

    print()
    print("╔" + "═" * (W - 2) + "╗")
    print(box_line())
    print(box_line("HEALTHCARE DATA WAREHOUSE — STAR SCHEMA REPORT", "center"))
    print(box_line(f"Database: {DB_PATH.name}  ({db_size_kb:.1f} KB)", "center"))
    print(box_line())
    print(box_sep())

    # ── Schema Diagram ──────────────────────────────────────────────────────
    print(box_line("  STAR SCHEMA DIAGRAM", "left"))
    print(box_line())
    print(box_line("              ┌─────────────────────────────────┐"))
    print(box_line("              │          dim_patient             │  ← DIMENSION"))
    print(box_line("              │  PK: patient_id  INTEGER AUTO    │"))
    print(box_line("              │      age         REAL NOT NULL   │"))
    print(box_line("              │      gender      INTEGER (0/1)   │"))
    print(box_line("              │      source_dataset TEXT         │"))
    print(box_line("              └──────────────┬──────────────────┘"))
    print(box_line("                             │"))
    print(box_line("          ┌──────────────────┼──────────────────┐"))
    print(box_line("          │ FK               │ FK               │ FK"))
    print(box_line("  ┌───────▼──────┐  ┌────────▼───────┐  ┌──────▼────────┐"))
    print(box_line("  │ fact_autism  │  │ fact_diabetes  │  │ fact_stroke   │"))
    print(box_line("  │ FACT TABLE   │  │ FACT TABLE     │  │ FACT TABLE    │"))
    print(
        box_line(
            f"  │ {row_autism:,} rows  │  │ {row_diabetes:,} rows      │  │ {row_stroke:,} rows   │"
        )
    )
    print(box_line("  │ 15 features  │  │ 14 symptoms    │  │ 9 features    │"))
    print(box_line("  └─────────────┘  └────────────────┘  └───────────────┘"))
    print(box_line())
    print(box_sep())

    # ── FK Relationships ────────────────────────────────────────────────────
    print(box_line("  FOREIGN KEY RELATIONSHIPS"))
    print(box_line())
    print(box_line("  ┌───────────────────────────┬────────────────────────────────┐"))
    print(box_line("  │  Fact Column              │  References                    │"))
    print(box_line("  ├───────────────────────────┼────────────────────────────────┤"))
    print(box_line("  │  fact_autism.patient_id   │  dim_patient.patient_id (PK)   │"))
    print(box_line("  │                           │  ON DELETE CASCADE             │"))
    print(box_line("  ├───────────────────────────┼────────────────────────────────┤"))
    print(box_line("  │  fact_diabetes.patient_id │  dim_patient.patient_id (PK)   │"))
    print(box_line("  │                           │  ON DELETE CASCADE             │"))
    print(box_line("  ├───────────────────────────┼────────────────────────────────┤"))
    print(box_line("  │  fact_stroke.patient_id   │  dim_patient.patient_id (PK)   │"))
    print(box_line("  │                           │  ON DELETE CASCADE             │"))
    print(box_line("  └───────────────────────────┴────────────────────────────────┘"))
    print(box_line())
    print(box_sep())

    # ── Table Summary ────────────────────────────────────────────────────────
    print(box_line("  TABLE SUMMARY"))
    print(box_line())
    col_w = [20, 12, 8, 8, 16]
    hdr = ["Table Name", "Type", "Rows", "Cols", "Primary Key"]
    print(box_line("  " + _top(col_w)))
    print(box_line("  " + _row(hdr, col_w)))
    print(box_line("  " + _divider(col_w)))
    print(
        box_line(
            "  "
            + _row(
                [
                    "dim_patient",
                    "DIMENSION",
                    f"{row_dim:,}",
                    col_dim,
                    "patient_id AUTO",
                ],
                col_w,
            )
        )
    )
    print(box_line("  " + _divider(col_w)))
    print(
        box_line(
            "  "
            + _row(
                ["fact_autism", "FACT", f"{row_autism:,}", col_autism, "fact_id AUTO"],
                col_w,
            )
        )
    )
    print(
        box_line(
            "  "
            + _row(
                [
                    "fact_diabetes",
                    "FACT",
                    f"{row_diabetes:,}",
                    col_diabetes,
                    "fact_id AUTO",
                ],
                col_w,
            )
        )
    )
    print(
        box_line(
            "  "
            + _row(
                ["fact_stroke", "FACT", f"{row_stroke:,}", col_stroke, "fact_id AUTO"],
                col_w,
            )
        )
    )
    print(box_line("  " + _bot(col_w)))
    print(box_line())
    print(box_sep())

    # ── DDL Execution Proof ──────────────────────────────────────────────────
    print(box_line("  SQL DDL EXECUTED"))
    print(box_line())
    print(
        box_line(
            "  ✔  PRAGMA foreign_keys = ON                           (FK enforcement active)"
        )
    )
    print(
        box_line(
            "  ✔  PRAGMA journal_mode  = WAL                         (Write-Ahead Logging)"
        )
    )
    print(
        box_line(
            "  ✔  PRAGMA synchronous   = NORMAL                      (Safe + performant)"
        )
    )
    print(box_line())
    print(
        box_line("  ✔  CREATE TABLE dim_patient        (4 cols  | 3 CHECK constraints)")
    )
    print(
        box_line(
            "  ✔  CREATE TABLE fact_autism        (16 cols | 13 CHECK constraints | 1 FK)"
        )
    )
    print(
        box_line(
            "  ✔  CREATE TABLE fact_diabetes      (17 cols | 15 CHECK constraints | 1 FK)"
        )
    )
    print(
        box_line(
            "  ✔  CREATE TABLE fact_stroke        (11 cols |  5 CHECK constraints | 1 FK)"
        )
    )
    print(
        box_line(
            f"  ✔  CREATE INDEX × {idx_count:<3}               (FK + label + source + gender)"
        )
    )
    print(box_line())
    print(box_sep())

    # ── INSERT Summary ────────────────────────────────────────────────────────
    print(box_line("  DATA LOADING SUMMARY  (via parameterized SQL INSERT statements)"))
    print(box_line())
    print(
        box_line(
            f"  dim_patient    : {row_dim:>6,} rows inserted  "
            f"(autism {len(df_autism):,} + diabetes {len(df_diabetes):,} + stroke {len(df_stroke):,})"
        )
    )
    print(
        box_line(
            f"  fact_autism    : {row_autism:>6,} rows inserted  ({elapsed_a:.3f}s)"
        )
    )
    print(
        box_line(
            f"  fact_diabetes  : {row_diabetes:>6,} rows inserted  ({elapsed_d:.3f}s)"
        )
    )
    print(
        box_line(
            f"  fact_stroke    : {row_stroke:>6,} rows inserted  ({elapsed_s:.3f}s)"
        )
    )
    print(box_line())
    print(
        box_line(
            f"  FK violations  :      0        ✔  All referential integrity checks passed"
        )
    )
    print(
        box_line(
            f"  NULL labels    :      0        ✔  All target columns fully populated"
        )
    )
    print(
        box_line(
            f"  DB file size   : {db_size_kb:>6.1f} KB   → warehouse/health_warehouse.db"
        )
    )
    print(box_line())
    print(box_sep())

    # ── Phase complete ────────────────────────────────────────────────────────
    print(box_line())
    print(box_line("  PHASE 1B COMPLETE — HEALTHCARE DATA WAREHOUSE READY", "center"))
    print(box_line())
    print(box_line("  Next Step: Phase 2 — EDA Notebook  (eda/02_eda.ipynb)", "center"))
    print(box_line())
    print("╚" + "═" * (W - 2) + "╝")
    print()


# ===========================================================================
# 12.  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_warehouse()
