# =============================================================================
# ETL Pipeline — Step 1: Load & Inspect Datasets
# Project : Early Disease Prediction Using Healthcare Data Warehouse
# =============================================================================

import os
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Resolve paths relative to THIS script so the script works regardless of
#    the current working directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "..", "datasets")

# ---------------------------------------------------------------------------
# 1. Dataset registry
#    key         → friendly display name
#    file        → CSV filename inside DATASETS_DIR
#    na_values   → extra strings that should be treated as NaN
# ---------------------------------------------------------------------------
DATASETS = {
    "Autism Screening": {
        "file"     : "Autism_Screening_Data_Combined.csv",
        "na_values": ["?", "", " ", "N/A", "n/a", "NA", "na", "null", "NULL", "None"],
    },
    "Diabetes Risk": {
        "file"     : "diabetes_data_upload.csv",
        "na_values": ["?", "", " ", "N/A", "n/a", "NA", "na", "null", "NULL", "None"],
    },
    "Stroke Prediction": {
        "file"     : "healthcare-dataset-stroke-data.csv",
        # The stroke CSV stores missing BMI values as the literal string "N/A"
        "na_values": ["?", "", " ", "N/A", "n/a", "NA", "na", "null", "NULL", "None"],
    },
}

# ---------------------------------------------------------------------------
# 2. Helper — pretty section banner
# ---------------------------------------------------------------------------
def banner(title: str, width: int = 70, char: str = "=") -> None:
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


# ---------------------------------------------------------------------------
# 3. Load all datasets
# ---------------------------------------------------------------------------
banner("LOADING DATASETS")

dataframes: dict[str, pd.DataFrame] = {}

for name, meta in DATASETS.items():
    filepath = os.path.join(DATASETS_DIR, meta["file"])

    if not os.path.isfile(filepath):
        print(f"  [WARNING]  File not found — skipping '{name}'\n"
              f"             Expected path: {filepath}")
        continue

    df = pd.read_csv(filepath, na_values=meta["na_values"], keep_default_na=True)
    dataframes[name] = df
    print(f"  [OK]  Loaded '{name}'  <-  {meta['file']}")

print(f"\n  Total datasets loaded : {len(dataframes)}")


# ---------------------------------------------------------------------------
# 4. Shape summary
# ---------------------------------------------------------------------------
banner("DATASET SHAPES  (rows × columns)")

for name, df in dataframes.items():
    rows, cols = df.shape
    print(f"\n  [{name}]")
    print(f"    Rows    : {rows:,}")
    print(f"    Columns : {cols}")
    print(f"    Columns : {list(df.columns)}")


# ---------------------------------------------------------------------------
# 5. Missing-value analysis
# ---------------------------------------------------------------------------
banner("MISSING VALUE ANALYSIS")

for name, df in dataframes.items():
    rows, cols = df.shape
    total_cells = rows * cols

    missing_per_col = df.isnull().sum()
    missing_cols    = missing_per_col[missing_per_col > 0]
    total_missing   = missing_per_col.sum()

    print(f"\n  -- {name} --")
    print(f"    Total cells   : {total_cells:,}")
    print(f"    Missing cells : {total_missing:,}  "
          f"({total_missing / total_cells * 100:.2f} %)")
    print(f"    Columns with missing values : {len(missing_cols)} / {cols}")

    if missing_cols.empty:
        print("    ✔  No missing values detected in any column.")
    else:
        print()
        print(f"    {'Column':<30} {'Missing':>8} {'% of Rows':>12}")
        print(f"    {'-'*30} {'-'*8} {'-'*12}")
        for col, count in missing_cols.sort_values(ascending=False).items():
            pct = count / rows * 100
            print(f"    {col:<30} {count:>8,} {pct:>11.2f}%")


# ---------------------------------------------------------------------------
# 6. Quick dtype overview
# ---------------------------------------------------------------------------
banner("COLUMN DATA TYPES")

for name, df in dataframes.items():
    print(f"\n  ── {name} ──")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"    {str(dtype):<15} : {count} column(s)")

    # List object (string) columns explicitly — they often need encoding later
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        print(f"\n    Categorical / object columns ({len(obj_cols)}):")
        for c in obj_cols:
            unique_vals = df[c].dropna().unique()
            preview     = ", ".join(str(v) for v in unique_vals[:6])
            ellipsis    = " …" if len(unique_vals) > 6 else ""
            print(f"      • {c:<28} unique={len(unique_vals):<5}  sample: [{preview}{ellipsis}]")


# ---------------------------------------------------------------------------
# 7. Final summary table
# ---------------------------------------------------------------------------
banner("SUMMARY TABLE")

header = f"  {'Dataset':<22} {'Rows':>7} {'Cols':>5} {'Missing Cells':>14} {'Missing %':>10}"
print(header)
print("  " + "-" * (len(header) - 2))

for name, df in dataframes.items():
    rows, cols    = df.shape
    total_cells   = rows * cols
    total_missing = df.isnull().sum().sum()
    pct           = total_missing / total_cells * 100
    print(f"  {name:<22} {rows:>7,} {cols:>5} {total_missing:>14,} {pct:>9.2f}%")

banner("INSPECTION COMPLETE", char="-")
print("  Next step → data cleaning & transformation (etl_pipeline/transform.py)")
print()
