# 🥇 Gold Data Layer — Star Schema Data Warehouse (OLAP-Ready)

## Purpose
The Gold Layer is the **business-ready analytical data warehouse**. It implements
a **Star Schema** in SQLite, integrating all three disease datasets through a
shared `dim_patient` dimension table. This layer is optimised for OLAP queries,
dashboards, and cross-disease population health analysis.

## Scripts & Files
| File | Description |
|---|---|
| **`warehouse.py`** | Python ETL script that creates the schema and loads Silver data |
| **`schema.sql`** | Complete DDL documentation (CREATE TABLE, indexes, views) |
| **`analytical_queries.sql`** | 20 documented OLAP queries across 6 sections |
| **`health_warehouse.db`** | The SQLite database (~1.2 MB) |

## Star Schema Architecture

```
                    ┌──────────────────────────────┐
                    │        dim_patient            │  DIMENSION TABLE
                    │   PK: patient_id (AUTO)       │  11,704 rows
                    │       age          REAL        │
                    │       gender       INTEGER     │  0=Female, 1=Male
                    │       source_dataset TEXT      │  'autism'|'diabetes'|'stroke'
                    └──────────────┬────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼──────────┐  ┌─────▼──────────┐  ┌─────▼──────────────┐
    │    fact_autism      │  │ fact_diabetes   │  │   fact_stroke      │
    │    6,075 rows      │  │   520 rows      │  │   5,109 rows       │
    │  FK: patient_id    │  │ FK: patient_id  │  │  FK: patient_id    │
    │  a1-a10, aq_score  │  │ 14 symptoms     │  │  clinical measures │
    │  jaundice          │  │ label           │  │  smoking, work_type│
    │  family_asd, label │  │                 │  │  label             │
    └────────────────────┘  └─────────────────┘  └────────────────────┘
```

### Why Star Schema?
1. **Shared Demographics**: Age and gender are common across all 3 diseases →
   normalised into `dim_patient` to avoid redundancy
2. **Cross-Disease Queries**: JOIN through `dim_patient` enables population-level
   analysis across autism, diabetes, and stroke simultaneously
3. **OLAP Performance**: Denormalised fact tables + indexes = fast analytical queries
4. **Referential Integrity**: FK constraints with `ON DELETE CASCADE` ensure
   no orphan records can exist

### Why NOT Snowflake Schema?
Only one dimension exists (patient demographics). There are no sub-dimensions
to normalise further (e.g., no separate city/country tables needed). A snowflake
schema would add JOINs without benefit at this scale.

## Data Integrity Features

### Constraints (DDL)
| Feature | Implementation | Example |
|---|---|---|
| Primary Keys | `AUTOINCREMENT` on all tables | `patient_id INTEGER PRIMARY KEY AUTOINCREMENT` |
| Foreign Keys | `REFERENCES dim_patient(patient_id)` | All 3 fact tables |
| Cascade Delete | `ON DELETE CASCADE` | Deleting a patient removes their facts |
| NOT NULL | Every column is `NOT NULL` | Zero NULLs allowed in warehouse |
| CHECK constraints | `CHECK (col IN (0, 1))` | All binary columns validated at DB level |
| Domain checks | `CHECK (source_dataset IN (...))` | Only valid dataset names allowed |

### Indexes (8 Total)
| Index | Table | Column(s) | Purpose |
|---|---|---|---|
| `idx_patient_source` | dim_patient | source_dataset | Filter by disease cohort |
| `idx_patient_gender` | dim_patient | gender | Demographic queries |
| `idx_autism_patient` | fact_autism | patient_id | Accelerate JOINs |
| `idx_autism_label` | fact_autism | label | Filter positive/negative |
| `idx_diabetes_patient` | fact_diabetes | patient_id | Accelerate JOINs |
| `idx_diabetes_label` | fact_diabetes | label | Filter positive/negative |
| `idx_stroke_patient` | fact_stroke | patient_id | Accelerate JOINs |
| `idx_stroke_label` | fact_stroke | label | Filter positive/negative |

### Analytical Views (4 Total)
| View | Purpose |
|---|---|
| `vw_autism_full` | Denormalised autism + patient demographics, decoded labels |
| `vw_diabetes_full` | Denormalised diabetes + patient demographics, decoded labels |
| `vw_stroke_full` | Denormalised stroke + patient demographics, decoded labels |
| `vw_disease_summary` | Executive dashboard: one row per disease with prevalence stats |

## Analytical Queries (20 Queries, 6 Sections)

### Section 1: Data Quality Verification (Q01–Q04)
- Row count verification, FK orphan check, NULL audit, domain constraint check

### Section 2: Dimension Analysis (Q05–Q07)
- Age distribution per cohort, age bracket roll-up, gender balance

### Section 3: Disease Prevalence (Q08–Q10)
- Global prevalence dashboard, class imbalance classification, comparative demographics

### Section 4: Disease Deep Dives (Q11–Q14)
- AQ score statistics, AQ-10 item analysis, symptom ranking, stroke clinical profile

### Section 5: Cross-Disease Analysis (Q15–Q17)
- Stroke by smoking status, comorbidity matrix, unified warehouse summary

### Section 6: Advanced OLAP (Q18–Q20)
- **ROLL-UP**: Stroke risk by age bracket × work type
- **SLICE**: High-risk stroke patient registry with composite risk score
- **DICE**: Cross-disease high-risk subgroup identification

### OLAP Operations Demonstrated
| Operation | Queries | Description |
|---|---|---|
| **ROLL-UP** | Q-06, Q-18 | Aggregate from patient → age bracket level |
| **DRILL-DOWN** | Q-14 | From summary → specific risk factors |
| **SLICE** | Q-19 | Single-dimension filter (stroke patients only) |
| **DICE** | Q-15, Q-16, Q-20 | Multi-dimension filter (2+ criteria) |
| **PIVOT** | Q-12, Q-13 | Conditional aggregation as PIVOT alternative |

## Validation Results

| Check | Result |
|---|---|
| dim_patient row count | 11,704 (6,075 + 520 + 5,109) ✔ |
| fact_autism row count | 6,075 ✔ |
| fact_diabetes row count | 520 ✔ |
| fact_stroke row count | 5,109 ✔ |
| FK orphan check (all tables) | 0 orphans ✔ |
| NULL audit (all columns) | 0 NULLs ✔ |
| Binary domain check | All values in {0, 1} ✔ |

## SQLite PRAGMA Settings
```sql
PRAGMA foreign_keys  = ON;      -- Enforce FK constraints
PRAGMA journal_mode  = WAL;     -- Write-Ahead Logging for concurrency
PRAGMA synchronous   = NORMAL;  -- Balance safety and performance
```
