# =============================================================================
# Final Project Verification Script
# Project : Early Disease Prediction Using Healthcare Data Warehouse
# Run     : python final_check.py
# =============================================================================

import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent

results = []


def chk(label, ok, detail=""):
    results.append((label, ok, detail))
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}]  {label}{suffix}")


print()
print("=" * 70)
print("  FINAL PROJECT VERIFICATION")
print("  Early Disease Prediction Using Healthcare Data Warehouse")
print("=" * 70)

# ---------------------------------------------------------------------------
print("\n  [1] DOCUMENTATION")
# ---------------------------------------------------------------------------
chk("README.md", (ROOT / "README.md").exists())
chk("requirements.txt", (ROOT / "requirements.txt").exists())

# ---------------------------------------------------------------------------
print("\n  [2] RAW DATASETS")
# ---------------------------------------------------------------------------
raw = ROOT / "datasets"
chk(
    "Autism_Screening_Data_Combined.csv",
    (raw / "Autism_Screening_Data_Combined.csv").exists(),
)
chk("diabetes_data_upload.csv", (raw / "diabetes_data_upload.csv").exists())
chk(
    "healthcare-dataset-stroke-data.csv",
    (raw / "healthcare-dataset-stroke-data.csv").exists(),
)

# ---------------------------------------------------------------------------
print("\n  [3] ETL SCRIPTS")
# ---------------------------------------------------------------------------
etl = ROOT / "etl_pipeline"
chk("load_and_inspect.py", (etl / "load_and_inspect.py").exists())
chk("clean.py", (etl / "clean.py").exists())
chk("warehouse.py", (etl / "warehouse.py").exists())

# ---------------------------------------------------------------------------
print("\n  [4] CLEANED DATASETS")
# ---------------------------------------------------------------------------
cleaned = ROOT / "datasets" / "cleaned"
for ds in ["autism", "diabetes", "stroke"]:
    f = cleaned / f"{ds}_cleaned.csv"
    kb = round(f.stat().st_size / 1024) if f.exists() else 0
    chk(f"{ds}_cleaned.csv", f.exists(), f"{kb} KB")

# ---------------------------------------------------------------------------
print("\n  [5] SQLITE DATA WAREHOUSE")
# ---------------------------------------------------------------------------
db = ROOT / "warehouse" / "health_warehouse.db"
chk(
    "health_warehouse.db exists",
    db.exists(),
    f"{round(db.stat().st_size / 1024)} KB" if db.exists() else "MISSING",
)

if db.exists():
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    # Row counts
    expected = {
        "dim_patient": 11704,
        "fact_autism": 6075,
        "fact_diabetes": 520,
        "fact_stroke": 5109,
    }
    for tbl, exp in expected.items():
        n = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        chk(f"{tbl} row count", n == exp, f"{n:,} rows")

    # FK integrity
    for tbl, fk_col, fact_col in [
        ("fact_autism", "fact_autism", "fa"),
        ("fact_diabetes", "fact_diabetes", "fd"),
        ("fact_stroke", "fact_stroke", "fs"),
    ]:
        orphans = cur.execute(
            f"SELECT COUNT(*) FROM {tbl} t "
            f"LEFT JOIN dim_patient dp ON t.patient_id = dp.patient_id "
            f"WHERE dp.patient_id IS NULL"
        ).fetchone()[0]
        chk(f"{tbl} FK integrity", orphans == 0, f"{orphans} orphans")

    # Views
    views = [
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()
    ]
    for v in [
        "vw_autism_full",
        "vw_diabetes_full",
        "vw_stroke_full",
        "vw_disease_summary",
    ]:
        chk(f"View: {v}", v in views)

    # Indexes
    idx_count = cur.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchone()[0]
    chk("8 indexes created", idx_count == 8, f"{idx_count} found")

    # NULL check on key columns
    nulls_bmi = cur.execute(
        "SELECT COUNT(*) FROM fact_stroke WHERE bmi IS NULL"
    ).fetchone()[0]
    chk("fact_stroke.bmi: 0 NULLs", nulls_bmi == 0, f"{nulls_bmi} NULLs")

    conn.close()

# ---------------------------------------------------------------------------
print("\n  [6] SQL DOCUMENTATION FILES")
# ---------------------------------------------------------------------------
wh = ROOT / "warehouse"
chk(
    "schema.sql",
    (wh / "schema.sql").exists(),
    f"{round((wh / 'schema.sql').stat().st_size / 1024)} KB"
    if (wh / "schema.sql").exists()
    else "",
)
chk(
    "analytical_queries.sql",
    (wh / "analytical_queries.sql").exists(),
    f"{round((wh / 'analytical_queries.sql').stat().st_size / 1024)} KB"
    if (wh / "analytical_queries.sql").exists()
    else "",
)
chk(
    "03_sql_analysis.ipynb",
    (wh / "03_sql_analysis.ipynb").exists(),
    f"{round((wh / '03_sql_analysis.ipynb').stat().st_size / 1024)} KB"
    if (wh / "03_sql_analysis.ipynb").exists()
    else "",
)

# ---------------------------------------------------------------------------
print("\n  [7] JUPYTER NOTEBOOKS")
# ---------------------------------------------------------------------------
chk(
    "02_eda.ipynb",
    (ROOT / "eda" / "02_eda.ipynb").exists(),
    f"{round((ROOT / 'eda' / '02_eda.ipynb').stat().st_size / 1024)} KB"
    if (ROOT / "eda" / "02_eda.ipynb").exists()
    else "",
)

chk(
    "03_feature_engineering.ipynb",
    (ROOT / "models" / "03_feature_engineering.ipynb").exists(),
    f"{round((ROOT / 'models' / '03_feature_engineering.ipynb').stat().st_size / 1024)} KB"
    if (ROOT / "models" / "03_feature_engineering.ipynb").exists()
    else "",
)

chk(
    "04_modeling.ipynb",
    (ROOT / "models" / "04_modeling.ipynb").exists(),
    f"{round((ROOT / 'models' / '04_modeling.ipynb').stat().st_size / 1024)} KB"
    if (ROOT / "models" / "04_modeling.ipynb").exists()
    else "",
)

# ---------------------------------------------------------------------------
print("\n  [8] SAVED MODEL ARTIFACTS")
# ---------------------------------------------------------------------------
saved = ROOT / "models" / "saved"
for art in [
    "autism_artifacts.pkl",
    "diabetes_artifacts.pkl",
    "stroke_artifacts.pkl",
    "autism_best_model.pkl",
    "diabetes_best_model.pkl",
    "stroke_best_model.pkl",
    "autism_results.csv",
    "diabetes_results.csv",
    "stroke_results.csv",
]:
    f = saved / art
    kb = round(f.stat().st_size / 1024) if f.exists() else 0
    chk(art, f.exists(), f"{kb} KB")

# ---------------------------------------------------------------------------
print("\n  [9] FIGURES")
# ---------------------------------------------------------------------------
figs_dir = ROOT / "reports" / "figures"
all_figs = list(figs_dir.glob("*.png")) if figs_dir.exists() else []
ml_figs = [f for f in all_figs if int(f.stem.split("_")[0]) <= 28]
sql_figs = [f for f in all_figs if int(f.stem.split("_")[0]) >= 29]

chk("reports/figures/ exists", figs_dir.exists())
chk("28 ML/EDA figures (01–28)", len(ml_figs) == 28, f"{len(ml_figs)} found")
chk("11 SQL figures   (29–39)", len(sql_figs) == 11, f"{len(sql_figs)} found")
chk("39 total figures", len(all_figs) == 39, f"{len(all_figs)} found")

# ---------------------------------------------------------------------------
print("\n  [10] FOLDER STRUCTURE")
# ---------------------------------------------------------------------------
for folder in [
    "datasets/cleaned",
    "etl_pipeline",
    "warehouse",
    "eda",
    "models/saved",
    "reports/figures",
]:
    chk(f"/{folder}/", (ROOT / folder).is_dir())

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)

print()
print("=" * 70)
print(
    f"  RESULT : {passed}/{total} checks passed"
    + (f"  |  {failed} FAILED" if failed else "")
)
print("=" * 70)

if failed == 0:
    print()
    print("  ALL CHECKS PASSED — Project is complete and submission-ready.")
    print()
    print("  Deliverables summary:")
    print(f"    Scripts       : 5  (ETL + generators)")
    print(f"    SQL files     : 2  (schema.sql + analytical_queries.sql)")
    print(f"    Notebooks     : 4  (EDA, FeatEng, Modeling, SQL Analysis)")
    print(f"    Figures       : {len(all_figs)}  (EDA + FeatEng + Modeling + SQL)")
    print(f"    Models saved  : 3  (autism, diabetes, stroke best models)")
    print(f"    DB tables     : 4  (1 dimension + 3 fact)")
    print(f"    DB views      : 4  analytical views")
    print(f"    SQL queries   : 20 (6 sections: quality, dim, prevalence,")
    print(f"                        drill-down, cross-join, OLAP)")
    print()
else:
    print()
    print("  FAILED CHECKS:")
    for label, ok, detail in results:
        if not ok:
            print(f"    - {label}  {detail}")
    print()
