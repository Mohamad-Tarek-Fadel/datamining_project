# =============================================================================
# Feature Selection Module — Heuristic + Genetic Algorithm Methods
# Project : Early Disease Prediction Using Healthcare Data Warehouse
# Script  : models/feature_selection.py
#
# METHODS:
#   1. Heuristic: Mutual Information, Chi-Square, Correlation-based
#   2. Genetic Algorithm: Evolutionary feature subset selection
#
# HOW TO RUN:
#   python models/feature_selection.py
# =============================================================================

import os
import sys
import warnings
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import (
    mutual_info_classif,
    chi2,
    SelectKBest,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
SAVED_DIR = SCRIPT_DIR / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR = ROOT_DIR / "reports" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def banner(title, width=72, char="="):
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def step(msg):
    print(f"    > {msg}")


# ---------------------------------------------------------------------------
# 1.  HEURISTIC FEATURE SELECTION
# ---------------------------------------------------------------------------

class HeuristicSelector:
    """Feature selection using three heuristic scoring methods:
       Mutual Information, Chi-Square, and Correlation-based ranking."""

    def __init__(self, X_train, y_train, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = list(feature_names)
        self.scores = {}

    def mutual_information(self):
        """Rank features by Mutual Information with the target."""
        mi = mutual_info_classif(self.X_train, self.y_train, random_state=42)
        self.scores["mutual_info"] = dict(zip(self.feature_names, mi))
        return pd.Series(mi, index=self.feature_names).sort_values(ascending=False)

    def chi_square(self):
        """Rank features by Chi-Square statistic (requires non-negative data)."""
        scaler = MinMaxScaler()
        X_pos = scaler.fit_transform(self.X_train)
        chi_scores, p_values = chi2(X_pos, self.y_train)
        self.scores["chi2"] = dict(zip(self.feature_names, chi_scores))
        return pd.Series(chi_scores, index=self.feature_names).sort_values(ascending=False)

    def correlation_based(self):
        """Rank features by absolute Pearson correlation with the target."""
        df = pd.DataFrame(self.X_train, columns=self.feature_names)
        df["target"] = self.y_train
        corr = df.corr()["target"].drop("target").abs()
        self.scores["correlation"] = corr.to_dict()
        return corr.sort_values(ascending=False)

    def aggregate_ranking(self, top_k=None):
        """Aggregate rankings from all three methods using average rank."""
        if not self.scores:
            self.mutual_information()
            self.chi_square()
            self.correlation_based()

        ranks = {}
        for method, scores in self.scores.items():
            sorted_feats = sorted(scores.keys(), key=lambda f: scores[f], reverse=True)
            for rank, feat in enumerate(sorted_feats):
                ranks.setdefault(feat, []).append(rank)

        avg_ranks = {f: np.mean(r) for f, r in ranks.items()}
        sorted_feats = sorted(avg_ranks.keys(), key=lambda f: avg_ranks[f])
        if top_k:
            sorted_feats = sorted_feats[:top_k]
        return sorted_feats, avg_ranks

    def select_top_k(self, k):
        """Return indices of top-k features by aggregate ranking."""
        top_feats, _ = self.aggregate_ranking(top_k=k)
        return [self.feature_names.index(f) for f in top_feats]


# ---------------------------------------------------------------------------
# 2.  GENETIC ALGORITHM FEATURE SELECTION
# ---------------------------------------------------------------------------

class GeneticAlgorithmSelector:
    """Feature selection using a Genetic Algorithm (GA).
    Each individual is a binary chromosome where 1 = feature selected."""

    def __init__(self, X_train, y_train, feature_names,
                 pop_size=30, n_generations=25, crossover_rate=0.8,
                 mutation_rate=0.1, elite_size=2, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = list(feature_names)
        self.n_features = X_train.shape[1]
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.rng = random.Random(random_state)
        np.random.seed(random_state)
        self.history = []

    def _init_population(self):
        """Initialize population with random binary chromosomes."""
        pop = []
        for _ in range(self.pop_size):
            # Ensure at least 2 features selected
            chrom = [self.rng.randint(0, 1) for _ in range(self.n_features)]
            while sum(chrom) < 2:
                idx = self.rng.randint(0, self.n_features - 1)
                chrom[idx] = 1
            pop.append(chrom)
        return pop

    def _fitness(self, chromosome):
        """Evaluate fitness using 3-fold CV F1 score with selected features."""
        selected = [i for i, g in enumerate(chromosome) if g == 1]
        if len(selected) < 2:
            return 0.0
        X_sub = self.X_train[:, selected]
        clf = LogisticRegression(max_iter=500, random_state=42,
                                 class_weight="balanced")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_sub, self.y_train, cv=cv,
                                 scoring="f1_weighted", n_jobs=-1)
        # Penalize using too many features (parsimony pressure)
        penalty = 1.0 - 0.005 * sum(chromosome)
        return float(np.mean(scores)) * max(penalty, 0.5)

    def _select_parents(self, population, fitnesses):
        """Tournament selection."""
        def tournament():
            i, j = self.rng.sample(range(len(population)), 2)
            return population[i] if fitnesses[i] > fitnesses[j] else population[j]
        return tournament(), tournament()

    def _crossover(self, p1, p2):
        """Single-point crossover."""
        if self.rng.random() < self.crossover_rate:
            pt = self.rng.randint(1, self.n_features - 1)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
            return c1, c2
        return p1[:], p2[:]

    def _mutate(self, chromosome):
        """Bit-flip mutation."""
        for i in range(len(chromosome)):
            if self.rng.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        # Ensure at least 2 features
        while sum(chromosome) < 2:
            idx = self.rng.randint(0, self.n_features - 1)
            chromosome[idx] = 1
        return chromosome

    def run(self):
        """Execute the GA and return the best chromosome."""
        population = self._init_population()
        best_chrom = None
        best_fit = -1.0

        for gen in range(self.n_generations):
            fitnesses = [self._fitness(c) for c in population]
            gen_best = max(fitnesses)
            gen_avg = np.mean(fitnesses)

            if gen_best > best_fit:
                best_fit = gen_best
                best_chrom = population[fitnesses.index(gen_best)][:]

            self.history.append({
                "generation": gen + 1,
                "best_fitness": gen_best,
                "avg_fitness": gen_avg,
                "n_features": sum(population[fitnesses.index(gen_best)]),
            })

            if (gen + 1) % 5 == 0 or gen == 0:
                step(f"Gen {gen+1:>3}/{self.n_generations}  "
                     f"best={gen_best:.4f}  avg={gen_avg:.4f}  "
                     f"features={sum(population[fitnesses.index(gen_best)])}")

            # Elitism
            sorted_pop = sorted(zip(fitnesses, population),
                                key=lambda x: x[0], reverse=True)
            new_pop = [ind[:] for _, ind in sorted_pop[:self.elite_size]]

            # Breed next generation
            while len(new_pop) < self.pop_size:
                p1, p2 = self._select_parents(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))

            population = new_pop

        selected_features = [self.feature_names[i]
                             for i, g in enumerate(best_chrom) if g == 1]
        return best_chrom, best_fit, selected_features


# ---------------------------------------------------------------------------
# 3.  COMPARISON: baseline vs heuristic vs GA
# ---------------------------------------------------------------------------

def evaluate_subset(X_train, y_train, feature_indices, method_name):
    """Evaluate a feature subset using 5-fold CV with RF."""
    X_sub = X_train[:, feature_indices]
    clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(clf, X_sub, y_train, cv=cv,
                         scoring="f1_weighted", n_jobs=-1)
    return {
        "Method": method_name,
        "Features": len(feature_indices),
        "CV F1 (mean)": round(float(np.mean(f1)), 4),
        "CV F1 (std)": round(float(np.std(f1)), 4),
    }


# ---------------------------------------------------------------------------
# 4.  MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    banner("FEATURE SELECTION MODULE")
    banner("Heuristic Methods + Genetic Algorithm", char="-")

    # Load artifacts
    datasets = {}
    for name in ["autism", "diabetes", "stroke"]:
        art_path = SAVED_DIR / f"{name}_artifacts.pkl"
        if not art_path.exists():
            print(f"  [ERROR] Missing: {art_path}")
            print("  Run models/03_feature_engineering.ipynb first.")
            sys.exit(1)
        art = joblib.load(art_path)
        datasets[name] = art
        step(f"Loaded {name} artifacts")

    all_results = []

    for ds_name, art in datasets.items():
        banner(f"DATASET: {ds_name.upper()}", char="-")

        X_train = art["X_train"]
        y_train = art["y_train"]

        # Use SMOTE data for stroke if available
        if "X_train_smote" in art and ds_name == "stroke":
            X_train = art["X_train_smote"]
            y_train = art["y_train_smote"]

        feature_names = art.get("feature_names", [f"f{i}" for i in range(X_train.shape[1])])
        n_features = X_train.shape[1]
        top_k = max(3, n_features // 2)  # Select top 50% features

        step(f"Total features: {n_features}, selecting top-{top_k}")

        # ── Baseline (all features) ──────────────────────────────────────────
        all_idx = list(range(n_features))
        baseline = evaluate_subset(X_train, y_train, all_idx, "Baseline (All)")
        all_results.append({"dataset": ds_name, **baseline})
        step(f"Baseline: CV F1 = {baseline['CV F1 (mean)']:.4f}")

        # ── Heuristic Selection ──────────────────────────────────────────────
        print()
        step("Running Heuristic Feature Selection...")
        hs = HeuristicSelector(X_train, y_train, feature_names)
        mi_scores = hs.mutual_information()
        chi_scores = hs.chi_square()
        corr_scores = hs.correlation_based()

        print(f"\n      {'Feature':<25} {'MI':>8} {'Chi2':>10} {'Corr':>8}")
        print(f"      {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
        for feat in feature_names[:min(10, len(feature_names))]:
            mi = hs.scores["mutual_info"].get(feat, 0)
            ch = hs.scores["chi2"].get(feat, 0)
            co = hs.scores["correlation"].get(feat, 0)
            print(f"      {feat:<25} {mi:>8.4f} {ch:>10.2f} {co:>8.4f}")

        heuristic_idx = hs.select_top_k(top_k)
        selected_names = [feature_names[i] for i in heuristic_idx]
        step(f"Selected features: {selected_names}")

        heuristic_result = evaluate_subset(X_train, y_train, heuristic_idx,
                                           f"Heuristic (top-{top_k})")
        all_results.append({"dataset": ds_name, **heuristic_result})
        step(f"Heuristic: CV F1 = {heuristic_result['CV F1 (mean)']:.4f}")

        # ── Genetic Algorithm ────────────────────────────────────────────────
        print()
        step("Running Genetic Algorithm Feature Selection...")
        t0 = time.perf_counter()
        ga = GeneticAlgorithmSelector(
            X_train, y_train, feature_names,
            pop_size=20, n_generations=15,
            crossover_rate=0.8, mutation_rate=0.15,
            random_state=42,
        )
        best_chrom, best_fit, ga_features = ga.run()
        ga_time = time.perf_counter() - t0

        ga_idx = [i for i, g in enumerate(best_chrom) if g == 1]
        step(f"GA selected {len(ga_features)} features in {ga_time:.1f}s: {ga_features}")

        ga_result = evaluate_subset(X_train, y_train, ga_idx,
                                    f"GA ({len(ga_features)} features)")
        all_results.append({"dataset": ds_name, **ga_result})
        step(f"GA: CV F1 = {ga_result['CV F1 (mean)']:.4f}")

        # Save GA history
        ga_history_path = SAVED_DIR / f"{ds_name}_ga_history.pkl"
        joblib.dump({
            "history": ga.history,
            "best_chromosome": best_chrom,
            "best_fitness": best_fit,
            "selected_features": ga_features,
            "heuristic_features": selected_names,
            "heuristic_scores": hs.scores,
        }, ga_history_path)
        step(f"Saved: {ga_history_path.name}")

    # ── Final Comparison Table ───────────────────────────────────────────────
    banner("FEATURE SELECTION — COMPARISON SUMMARY", char="-")

    df_results = pd.DataFrame(all_results)
    print()
    print(df_results.to_string(index=False))
    print()

    results_path = SAVED_DIR / "feature_selection_results.csv"
    df_results.to_csv(results_path, index=False)
    step(f"Saved: {results_path.name}")

    banner("FEATURE SELECTION COMPLETE", char="-")
    print()


if __name__ == "__main__":
    main()
