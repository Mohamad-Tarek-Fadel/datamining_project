# =============================================================================
# Ensemble Modeling Module — Voting + Stacking Ensembles
# Project : Early Disease Prediction Using Healthcare Data Warehouse
# Script  : models/ensemble_models.py
#
# ENSEMBLES:
#   1. Voting Ensemble (Hard + Soft voting)
#   2. Stacking Ensemble (Meta-learner: Logistic Regression)
#
# HOW TO RUN:
#   python models/ensemble_models.py
# =============================================================================

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("\n[WARNING] MLflow is not installed. Experiment tracking will be disabled. Run 'pip install mlflow' to enable.")

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
SAVED_DIR = SCRIPT_DIR / "saved"
FIGS_DIR = SCRIPT_DIR.parent / "reports" / "figures"

# Set up MLflow tracking URI if available
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri("file://" + str(SCRIPT_DIR.parent / "mlruns"))  
    mlflow.set_experiment("Clinical_Intelligence")


def banner(title, width=72, char="="):
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def step(msg):
    print(f"    > {msg}")


# ---------------------------------------------------------------------------
# 1.  BASE ESTIMATORS FACTORY
# ---------------------------------------------------------------------------

def get_base_estimators(dataset_name):
    """Return a list of (name, estimator) tuples for ensemble building."""
    cw = "balanced"
    # Note: SVM must have probability=True for Soft Voting and Stacking
    estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight=cw)),
        ("svm", SVC(probability=True, random_state=42, class_weight=cw)),
        ("nb", GaussianNB()),
    ]
    return estimators


# ---------------------------------------------------------------------------
# 2.  VOTING ENSEMBLE
# ---------------------------------------------------------------------------

def build_voting_ensemble(estimators, voting="soft"):
    """Build a Voting Classifier from base estimators."""
    return VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# 3.  STACKING ENSEMBLE
# ---------------------------------------------------------------------------

def build_stacking_ensemble(estimators):
    """Build a Stacking Classifier with LR as meta-learner."""
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42,
                                           class_weight="balanced"),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method="predict_proba",
        n_jobs=-1,
    )


from contextlib import contextmanager

@contextmanager
def dummy_run():
    yield

# ---------------------------------------------------------------------------
# 4.  EVALUATION & MLFLOW TRACKING
# ---------------------------------------------------------------------------

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, dataset_name):
    """Train, predict, and compute comprehensive metrics."""
    run_context = mlflow.start_run(run_name=f"{dataset_name} - {model_name}", nested=True) if MLFLOW_AVAILABLE else dummy_run()
    with run_context:
        if MLFLOW_AVAILABLE:
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("features", X_train.shape[1])
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
        except Exception:
            roc = 0.0
            pr_auc = 0.0

        # CV score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="f1_weighted", n_jobs=-1)

        from sklearn.metrics import accuracy_score
        metrics = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Features": X_train.shape[1],
            "CV F1": round(float(np.mean(cv_f1)), 4),
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "Recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
            "ROC-AUC": round(roc, 4),
            "PR-AUC": round(pr_auc, 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
        }

        for key, val in metrics.items():
            if isinstance(val, (int, float)) and MLFLOW_AVAILABLE:
                mlflow.log_metric(key.lower().replace("-", "_").replace(" ", "_"), val)

        # Error analysis
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        if MLFLOW_AVAILABLE:
            mlflow.sklearn.log_model(model, "model")

        return metrics, cm, report, model


# ---------------------------------------------------------------------------
# 5.  MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    banner("ENSEMBLE MODELING MODULE")
    banner("Voting Ensemble + Stacking Ensemble", char="-")

    all_results = []

    for ds_name in ["autism", "diabetes", "stroke"]:
        art_path = SAVED_DIR / f"{ds_name}_artifacts.pkl"
        if not art_path.exists():
            print(f"  [ERROR] Missing: {art_path}")
            sys.exit(1)

        art = joblib.load(art_path)
        X_train = art["X_train"]
        y_train = art["y_train"]
        X_test = art["X_test"]
        y_test = art["y_test"]

        X_train_fit = X_train
        y_train_fit = y_train

        banner(f"DATASET: {ds_name.upper()}", char="-")
        step(f"Train: {X_train_fit.shape}, Test: {X_test.shape}")

        estimators = get_base_estimators(ds_name)

        # ── Soft Voting Ensemble ─────────────────────────────────────────────
        step("Building Soft Voting Ensemble...")
        voting_soft = build_voting_ensemble(estimators, voting="soft")
        m1, cm1, r1, fitted_vs = evaluate_model(
            voting_soft, X_train_fit, y_train_fit, X_test, y_test,
            "Voting (Soft)", ds_name)
        all_results.append({"dataset": ds_name, **m1})
        step(f"  F1={m1['F1']:.4f}  Recall={m1['Recall']:.4f}  "
             f"ROC-AUC={m1['ROC-AUC']:.4f}")

        # ── Hard Voting Ensemble ─────────────────────────────────────────────
        step("Building Hard Voting Ensemble...")
        voting_hard = build_voting_ensemble(estimators, voting="hard")
        m2, cm2, r2, fitted_vh = evaluate_model(
            voting_hard, X_train_fit, y_train_fit, X_test, y_test,
            "Voting (Hard)", ds_name)
        all_results.append({"dataset": ds_name, **m2})
        step(f"  F1={m2['F1']:.4f}  Recall={m2['Recall']:.4f}")

        # ── Stacking Ensemble ────────────────────────────────────────────────
        step("Building Stacking Ensemble (meta-learner: LR)...")
        stacking = build_stacking_ensemble(estimators)
        m3, cm3, r3, fitted_st = evaluate_model(
            stacking, X_train_fit, y_train_fit, X_test, y_test,
            "Stacking (LR Meta)", ds_name)
        all_results.append({"dataset": ds_name, **m3})
        step(f"  F1={m3['F1']:.4f}  Recall={m3['Recall']:.4f}  "
             f"ROC-AUC={m3['ROC-AUC']:.4f}")

        # ── Error Analysis ───────────────────────────────────────────────────
        print(f"\n  Confusion Matrix (Stacking — {ds_name}):")
        print(f"    {cm3}")

        # ── Save best ensemble ───────────────────────────────────────────────
        best_m = max([m1, m2, m3], key=lambda x: x["F1"])
        best_name = best_m["Model"]
        if best_name == m1["Model"]:
            best_model = fitted_vs
        elif best_name == m2["Model"]:
            best_model = fitted_vh
        else:
            best_model = fitted_st

        ens_path = SAVED_DIR / f"{ds_name}_ensemble_model.pkl"
        joblib.dump(best_model, ens_path)
        step(f"Saved best ensemble ({best_name}): {ens_path.name}")

    # ── Final Summary ────────────────────────────────────────────────────────
    banner("ENSEMBLE RESULTS SUMMARY", char="-")

    df_results = pd.DataFrame(all_results)
    print()
    print(df_results.to_string(index=False))
    print()

    results_path = SAVED_DIR / "ensemble_results.csv"
    df_results.to_csv(results_path, index=False)
    step(f"Saved: {results_path.name}")

    banner("ENSEMBLE MODELING COMPLETE", char="-")
    print()


if __name__ == "__main__":
    main()
