"""Model 6 / GA-Evolved Gate: a genetic algorithm searches over which of the
9 gate context features to use and how much capacity the gate needs
(hidden_size), rather than hand-picking Model 4's full feature set -- the
proposal's Unit 1 (evolutionary computing) tie-in and the direct test of
whether feature/architecture selection can match or beat the hand-picked
Model 4 baseline.

Each candidate genome is fitted with atg.gates.nn.GateNet -- the exact same
model class Model 4 uses -- on a short training budget purely to rank
candidates (fitness = -early-stop hybrid MSE on the same val gate-train/
gate-early-stop split Model 4 uses, so genomes are compared fairly against
each other and against Model 4 on identical data). The winning genome is
then retrained to full convergence for the final gate, mirroring the
proposal's Week 5 milestone ("genetic algorithm for feature/architecture
search; retrain evolved gate").
"""
import time

import numpy as np
import pandas as pd

from atg.gates.features import FEATURE_COLUMNS, build_gate_features
from atg.gates.nn import GateNet

HIDDEN_SIZE_CHOICES = (0, 4, 8)


def _random_genome(rng, n_features):
    mask = rng.random(n_features) < 0.6
    if not mask.any():
        mask[rng.integers(n_features)] = True
    return {"mask": mask, "hidden_size": int(rng.choice(HIDDEN_SIZE_CHOICES))}


def _fit_genome(genome, X_tr, cf_tr, cb_tr, y_tr, X_es, cf_es, cb_es, y_es, epochs, patience, seed):
    cols = np.where(genome["mask"])[0]
    l2 = 1.0 if genome["hidden_size"] == 0 else 1e-3
    net = GateNet(n_features=len(cols), hidden_size=genome["hidden_size"], l2=l2, lr=0.05, seed=seed)
    net.fit(X_tr[:, cols], cf_tr, cb_tr, y_tr, X_es[:, cols], cf_es, cb_es, y_es,
            epochs=epochs, patience=patience, log_every=max(epochs // 10, 1))
    es_loss = net.history_[-1][2]
    return es_loss, net


def _crossover(rng, p1, p2):
    mask = np.where(rng.random(len(p1["mask"])) < 0.5, p1["mask"], p2["mask"])
    if not mask.any():
        mask[rng.integers(len(mask))] = True
    hidden = p1["hidden_size"] if rng.random() < 0.5 else p2["hidden_size"]
    return {"mask": mask, "hidden_size": hidden}


def _mutate(rng, genome, feature_flip_prob=0.15, hidden_jump_prob=0.2):
    mask = genome["mask"].copy()
    flips = rng.random(len(mask)) < feature_flip_prob
    mask[flips] = ~mask[flips]
    if not mask.any():
        mask[rng.integers(len(mask))] = True
    hidden = genome["hidden_size"]
    if rng.random() < hidden_jump_prob:
        hidden = int(rng.choice(HIDDEN_SIZE_CHOICES))
    return {"mask": mask, "hidden_size": hidden}


def _tournament_select(rng, population, fitness, k=3):
    idx = rng.integers(0, len(population), size=k)
    best = idx[np.argmin([fitness[i] for i in idx])]  # lower held-out MSE = better
    return population[best]


def run_ga(X_tr, cf_tr, cb_tr, y_tr, X_es, cf_es, cb_es, y_es,
           population_size=16, generations=12, elitism=2, seed=42,
           search_epochs=800, search_patience=100):
    rng = np.random.default_rng(seed)
    n_features = X_tr.shape[1]

    population = [_random_genome(rng, n_features) for _ in range(population_size)]
    history = []
    best_genome, best_loss = None, np.inf

    for gen in range(generations):
        scored = []
        for genome in population:
            loss, _ = _fit_genome(genome, X_tr, cf_tr, cb_tr, y_tr, X_es, cf_es, cb_es, y_es,
                                   search_epochs, search_patience, seed)
            scored.append(loss)
            if loss < best_loss:
                best_loss, best_genome = loss, genome

        history.append({
            "generation": gen,
            "best_es_mse": float(min(scored)),
            "mean_es_mse": float(np.mean(scored)),
        })

        order = np.argsort(scored)
        new_population = [population[i] for i in order[:elitism]]  # elitism
        while len(new_population) < population_size:
            p1 = _tournament_select(rng, population, scored)
            p2 = _tournament_select(rng, population, scored)
            new_population.append(_mutate(rng, _crossover(rng, p1, p2)))
        population = new_population

    return best_genome, best_loss, history


class GAEvolvedGate:
    def __init__(self, es_frac: float = 0.2, seed: int = 42,
                 population_size: int = 16, generations: int = 12,
                 search_epochs: int = 800, search_patience: int = 100,
                 final_epochs: int = 3000, final_patience: int = 200):
        self.es_frac = es_frac
        self.seed = seed
        self.population_size = population_size
        self.generations = generations
        self.search_epochs = search_epochs
        self.search_patience = search_patience
        self.final_epochs = final_epochs
        self.final_patience = final_patience

        self.net = None
        self.selected_mask_ = None
        self.selected_features_ = None
        self.ga_history_ = None
        self.train_time_sec = None

    def fit(self, val_df: pd.DataFrame, item_popularity: dict, cf_expert, cb_expert) -> "GAEvolvedGate":
        start = time.perf_counter()

        feats = build_gate_features(val_df, item_popularity, cf_expert, cb_expert)
        X = feats.to_numpy(dtype=float)
        cf = val_df["cf_pred"].to_numpy(dtype=float)
        cb = val_df["cb_pred"].to_numpy(dtype=float)
        y = val_df["rating"].to_numpy(dtype=float)

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(val_df))
        n_es = int(round(len(val_df) * self.es_frac))
        es_idx, tr_idx = idx[:n_es], idx[n_es:]
        X_tr, X_es = X[tr_idx], X[es_idx]
        cf_tr, cf_es = cf[tr_idx], cf[es_idx]
        cb_tr, cb_es = cb[tr_idx], cb[es_idx]
        y_tr, y_es = y[tr_idx], y[es_idx]

        best_genome, best_loss, history = run_ga(
            X_tr, cf_tr, cb_tr, y_tr, X_es, cf_es, cb_es, y_es,
            population_size=self.population_size, generations=self.generations, seed=self.seed,
            search_epochs=self.search_epochs, search_patience=self.search_patience,
        )
        self.ga_history_ = history
        self.selected_mask_ = best_genome["mask"]
        self.selected_features_ = [f for f, keep in zip(FEATURE_COLUMNS, self.selected_mask_) if keep]

        cols = np.where(self.selected_mask_)[0]
        l2 = 1.0 if best_genome["hidden_size"] == 0 else 1e-3
        self.net = GateNet(n_features=len(cols), hidden_size=best_genome["hidden_size"],
                            l2=l2, lr=0.05, seed=self.seed)
        self.net.fit(X_tr[:, cols], cf_tr, cb_tr, y_tr, X_es[:, cols], cf_es, cb_es, y_es,
                      epochs=self.final_epochs, patience=self.final_patience)

        self.train_time_sec = time.perf_counter() - start
        return self

    def g(self, df: pd.DataFrame, item_popularity: dict, cf_expert, cb_expert) -> np.ndarray:
        feats = build_gate_features(df, item_popularity, cf_expert, cb_expert)
        X = feats.to_numpy(dtype=float)
        cols = np.where(self.selected_mask_)[0]
        return self.net.g(X[:, cols])

    def n_params(self) -> int:
        return self.net.n_params()
