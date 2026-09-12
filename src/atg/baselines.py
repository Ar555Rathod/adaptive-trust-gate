"""External baselines: published combination strategies Models 3-7 are measured
against.

Models 1-7 form an internal ablation -- every arm is ours, so the comparison
establishes which gating mechanism is best but not whether gating is the right
tool at all. These four are the standard alternatives:

  B1 Switching Hybrid (Burke, 2002)   -- the hard version of a gate: g in {0,1},
     chosen by a threshold on the user's train rating count.
  B2 Feature-Weighted Linear Stacking (Sill et al., 2009) -- blend weights are
     linear in the same meta-features Model 4 sees:
     r_hat = (a.x)*cf + (b.x)*cb. A strict SUPERSET of Model 4, which
     additionally forces the two weights to sum to 1.
  B3 Ridge Stacking -- linear regression on the two expert predictions alone,
     no context. Isolates how much comes from context versus from merely being
     allowed an intercept and weights that need not sum to 1.
  B4 Gradient-Boosted Stacking -- nonlinear stacker over [cf, cb] + the 9 gate
     features; the strongest off-the-shelf competitor.

Every fitter takes the SAME val gate-train/early-stop partition the learned
gates use (see gate_split), so no baseline sees a rating a gate did not.

This lives in the package rather than in scripts/11 so that
scripts/10_multiseed_full.py can reuse it without importing a numerically
prefixed module -- the baselines need seed variance reported the same way the
gates do, otherwise the headline table compares "0.8952 +/- s" against a bare
point estimate.
"""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from atg import config

CLIP = (config.RATING_MIN, config.RATING_MAX)

BASELINE_NAMES = ["B1_SwitchingHybrid", "B2_FWLS", "B3_RidgeStack", "B4_GBMStack"]


def clip(pred):
    return np.clip(pred, *CLIP)


def gate_split(n, es_frac=0.2, seed=config.RANDOM_SEED):
    """The exact val partition atg.gates.learned.LearnedGate uses, so every
    baseline is fit on the same rows the learned gates were fit on."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_es = int(round(n * es_frac))
    return idx[n_es:], idx[:n_es]   # (fit_idx, holdout_idx)


# ---------------------------------------------------------------- B1
def fit_switching(counts, cf, cb, y, fit_idx, max_tau=50):
    start = time.perf_counter()
    best_tau, best_score, curve = None, np.inf, []
    for tau in range(0, max_tau + 1):
        pred = clip(np.where(counts[fit_idx] >= tau, cf[fit_idx], cb[fit_idx]))
        score = float(np.sqrt(np.mean((y[fit_idx] - pred) ** 2)))
        curve.append((int(tau), score))
        if score < best_score:
            best_score, best_tau = score, int(tau)
    return {"tau": best_tau, "grid": curve,
            "train_time_sec": time.perf_counter() - start, "n_params": 1}


def predict_switching(m, counts, cf, cb):
    return clip(np.where(counts >= m["tau"], cf, cb))


# ---------------------------------------------------------------- B2
def _fwls_design(X, cf, cb):
    """[x*cf | x*cb] with an intercept folded into x, so the fitted weights are
    exactly the (a, b) of r_hat = (a.x)*cf + (b.x)*cb."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    return np.hstack([Xa * cf[:, None], Xa * cb[:, None]])


def fit_fwls(X, cf, cb, y, fit_idx, mean, std):
    start = time.perf_counter()
    Z = _fwls_design((X - mean) / std, cf, cb)
    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(Z[fit_idx], y[fit_idx])
    return {"model": model, "mean": mean, "std": std,
            "train_time_sec": time.perf_counter() - start, "n_params": int(Z.shape[1])}


def predict_fwls(m, X, cf, cb):
    return clip(m["model"].predict(_fwls_design((X - m["mean"]) / m["std"], cf, cb)))


# ---------------------------------------------------------------- B3
def fit_ridge_stack(cf, cb, y, fit_idx):
    start = time.perf_counter()
    Z = np.column_stack([cf, cb])
    model = Ridge(alpha=1.0)
    model.fit(Z[fit_idx], y[fit_idx])
    return {"model": model, "train_time_sec": time.perf_counter() - start, "n_params": 3}


def predict_ridge_stack(m, cf, cb):
    return clip(m["model"].predict(np.column_stack([cf, cb])))


# ---------------------------------------------------------------- B4
def fit_gbm_stack(X, cf, cb, y, fit_idx, hold_idx, seed=config.RANDOM_SEED):
    start = time.perf_counter()
    Z = np.column_stack([cf, cb, X])
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=None, n_iter_no_change=20,
        random_state=seed,
    )
    model.fit(Z[fit_idx], y[fit_idx])
    hold_pred = clip(model.predict(Z[hold_idx]))
    return {"model": model,
            "holdout_rmse": float(np.sqrt(np.mean((y[hold_idx] - hold_pred) ** 2))),
            "train_time_sec": time.perf_counter() - start,
            "n_params": int(len(model._predictors) * 31)}   # trees x max_leaf_nodes


def predict_gbm_stack(m, X, cf, cb):
    return clip(m["model"].predict(np.column_stack([cf, cb, X])))


# ---------------------------------------------------------------- all four
def fit_all(X_val, cf_val, cb_val, y_val, counts_val, seed=config.RANDOM_SEED, es_frac=0.2):
    """Fit all four baselines on VAL. Returns {name: fitted_model_dict}."""
    fit_idx, hold_idx = gate_split(len(y_val), es_frac=es_frac, seed=seed)
    mean = X_val[fit_idx].mean(axis=0)
    std = X_val[fit_idx].std(axis=0)
    std[std < 1e-8] = 1.0
    return {
        "B1_SwitchingHybrid": fit_switching(counts_val, cf_val, cb_val, y_val, fit_idx),
        "B2_FWLS": fit_fwls(X_val, cf_val, cb_val, y_val, fit_idx, mean, std),
        "B3_RidgeStack": fit_ridge_stack(cf_val, cb_val, y_val, fit_idx),
        "B4_GBMStack": fit_gbm_stack(X_val, cf_val, cb_val, y_val, fit_idx, hold_idx, seed=seed),
    }, (fit_idx, hold_idx)


def predict_all(fitted, X, cf, cb, counts):
    """Apply all four fitted baselines to a split. Returns {name: predictions}."""
    return {
        "B1_SwitchingHybrid": predict_switching(fitted["B1_SwitchingHybrid"], counts, cf, cb),
        "B2_FWLS": predict_fwls(fitted["B2_FWLS"], X, cf, cb),
        "B3_RidgeStack": predict_ridge_stack(fitted["B3_RidgeStack"], cf, cb),
        "B4_GBMStack": predict_gbm_stack(fitted["B4_GBMStack"], X, cf, cb),
    }
