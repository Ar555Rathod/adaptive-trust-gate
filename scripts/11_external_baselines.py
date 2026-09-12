"""External baselines: published combination strategies that Models 3-7 must
be measured against, not just against each other.

Models 1-7 form an internal ablation -- every arm is ours, so the comparison
establishes which gating mechanism is best but says nothing about whether
gating is the right tool at all. These four baselines answer the questions a
reviewer asks first:

  B1 Switching Hybrid (Burke, 2002)   -- the classic hard alternative to a soft
     gate: use CF above a user-count threshold, content below it. If a step
     function matches a learned gate, the gate's extra machinery is unjustified.

  B2 Feature-Weighted Linear Stacking (Sill et al., 2009) -- the canonical
     learned-blend baseline. Blend weights are linear in the same meta-features
     Model 4 sees: r_hat = (a.x)*cf + (b.x)*cb. This is a strict SUPERSET of
     Model 4, which additionally constrains the two weights to sum to 1, so it
     upper-bounds what a linear gate can do.

  B3 Ridge Stacking -- the simplest possible learned combiner, on the two expert
     predictions alone with no context features. Isolates how much of any gain
     comes from context rather than from merely learning a combination.

  B4 Gradient-Boosted Stacking -- a nonlinear stacker over [cf, cb] + the 9 gate
     features. The strongest off-the-shelf competitor; if it beats every gate,
     the contribution has to be reframed around interpretability and compute
     cost rather than accuracy.

Protocol: identical to the gates. Everything is fit on VAL (the experts were
fit on TRAIN, so only their VAL predictions are honest out-of-sample scores),
using the SAME val gate-train / early-stop partition Model 4 and Model 6 use --
same seed, same permutation, same 80/20 -- and scored on the untouched TEST
split. No baseline sees a rating the gates did not also see.
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from atg import config
from atg.data.splits import load_splits
from atg.eval.metrics import segmented_rating_metrics, rmse
from atg.gates.features import build_item_popularity, build_gate_features, FEATURE_COLUMNS

CLIP = (config.RATING_MIN, config.RATING_MAX)


def print_segment_table(name, metrics):
    print(f"\n[{name}]")
    for seg in ("overall", "cold", "warm", "power"):
        if seg in metrics:
            m = metrics[seg]
            print(f"  {seg:8s} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  n={m['n']}")


def clip(pred):
    return np.clip(pred, *CLIP)


def gate_split(n, es_frac=0.2, seed=config.RANDOM_SEED):
    """The exact val partition atg.gates.learned.LearnedGate uses, so every
    baseline is fit on the same rows the learned gates were fit on."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_es = int(round(n * es_frac))
    return idx[n_es:], idx[:n_es]   # (fit_idx, holdout_idx)


# --------------------------------------------------------------------------
# B1 -- Switching Hybrid (Burke, 2002)
# --------------------------------------------------------------------------
def fit_switching(val_df, fit_idx):
    """Grid-search the user-count threshold tau: predict CF when the user has
    at least tau train ratings, content otherwise."""
    start = time.perf_counter()
    counts = val_df["train_rating_count"].to_numpy(dtype=float)
    cf = val_df["cf_pred"].to_numpy(dtype=float)
    cb = val_df["cb_pred"].to_numpy(dtype=float)
    y = val_df["rating"].to_numpy(dtype=float)

    best_tau, best_score, curve = None, np.inf, []
    for tau in range(0, 51):
        pred = clip(np.where(counts[fit_idx] >= tau, cf[fit_idx], cb[fit_idx]))
        score = rmse(y[fit_idx], pred)
        curve.append((int(tau), float(score)))
        if score < best_score:
            best_score, best_tau = score, int(tau)

    return {"tau": best_tau, "grid": curve, "train_time_sec": time.perf_counter() - start,
            "n_params": 1}


def predict_switching(model, df):
    counts = df["train_rating_count"].to_numpy(dtype=float)
    cf = df["cf_pred"].to_numpy(dtype=float)
    cb = df["cb_pred"].to_numpy(dtype=float)
    return clip(np.where(counts >= model["tau"], cf, cb))


# --------------------------------------------------------------------------
# B2 -- Feature-Weighted Linear Stacking (Sill et al., 2009)
# --------------------------------------------------------------------------
def _fwls_design(X, cf, cb):
    """[x*cf | x*cb] with an intercept column folded into x, so the fitted
    weights are exactly the (a, b) of r_hat = (a.x)*cf + (b.x)*cb."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    return np.hstack([Xa * cf[:, None], Xa * cb[:, None]])


def fit_fwls(X, cf, cb, y, fit_idx, mean, std):
    start = time.perf_counter()
    Xn = (X - mean) / std
    Z = _fwls_design(Xn, cf, cb)
    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(Z[fit_idx], y[fit_idx])
    return {"model": model, "mean": mean, "std": std,
            "train_time_sec": time.perf_counter() - start,
            "n_params": int(Z.shape[1])}


def predict_fwls(m, X, cf, cb):
    Xn = (X - m["mean"]) / m["std"]
    return clip(m["model"].predict(_fwls_design(Xn, cf, cb)))


# --------------------------------------------------------------------------
# B3 -- Ridge stacking on the two expert predictions alone
# --------------------------------------------------------------------------
def fit_ridge_stack(cf, cb, y, fit_idx):
    start = time.perf_counter()
    Z = np.column_stack([cf, cb])
    model = Ridge(alpha=1.0)
    model.fit(Z[fit_idx], y[fit_idx])
    return {"model": model, "train_time_sec": time.perf_counter() - start, "n_params": 3}


def predict_ridge_stack(m, cf, cb):
    return clip(m["model"].predict(np.column_stack([cf, cb])))


# --------------------------------------------------------------------------
# B4 -- Gradient-boosted stacking over [cf, cb] + the 9 gate features
# --------------------------------------------------------------------------
def fit_gbm_stack(X, cf, cb, y, fit_idx, hold_idx):
    start = time.perf_counter()
    Z = np.column_stack([cf, cb, X])
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=None, n_iter_no_change=20,
        random_state=config.RANDOM_SEED,
    )
    # Early stopping uses the SAME held-out val slice the learned gates use, by
    # concatenating it last and pointing sklearn's internal split at it is not
    # possible -- so fit on fit_idx only and report holdout RMSE separately.
    model.fit(Z[fit_idx], y[fit_idx])
    n_trees = len(model._predictors)
    return {"model": model,
            "holdout_rmse": float(rmse(y[hold_idx], clip(model.predict(Z[hold_idx])))),
            "train_time_sec": time.perf_counter() - start,
            "n_params": int(n_trees * 31)}   # upper bound: trees x max_leaf_nodes


def predict_gbm_stack(m, X, cf, cb):
    return clip(m["model"].predict(np.column_stack([cf, cb, X])))


# --------------------------------------------------------------------------
def main():
    train_df, _, _ = load_splits()
    val_df = pd.read_csv(config.PREDICTIONS_DIR / "experts_val.csv")
    test_df = pd.read_csv(config.PREDICTIONS_DIR / "experts_test.csv")

    with open(config.MODELS_DIR / "cf_svdpp.pkl", "rb") as f:
        cf_expert = pickle.load(f)
    with open(config.MODELS_DIR / "content_based.pkl", "rb") as f:
        cb_expert = pickle.load(f)

    item_popularity = build_item_popularity(train_df)

    print("Building gate context features (val/test)...")
    X_val = build_gate_features(val_df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=float)
    X_test = build_gate_features(test_df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=float)

    cf_val, cb_val, y_val = (val_df[c].to_numpy(dtype=float) for c in ("cf_pred", "cb_pred", "rating"))
    cf_test, cb_test = (test_df[c].to_numpy(dtype=float) for c in ("cf_pred", "cb_pred"))

    fit_idx, hold_idx = gate_split(len(val_df))
    print(f"  val fit rows={len(fit_idx)}  holdout rows={len(hold_idx)}"
          f"  (same partition as Models 4 and 6)")

    mean = X_val[fit_idx].mean(axis=0)
    std = X_val[fit_idx].std(axis=0)
    std[std < 1e-8] = 1.0

    results, summary = {}, []

    specs = []

    print("\nB1  Switching Hybrid (Burke, 2002)...")
    m = fit_switching(val_df, fit_idx)
    print(f"  best tau={m['tau']} train ratings  (train_time_sec={m['train_time_sec']:.3f})")
    specs.append(("B1_SwitchingHybrid", "b1_switching_hybrid",
                  predict_switching(m, val_df), predict_switching(m, test_df),
                  m, {"tau": m["tau"]}))

    print("\nB2  Feature-Weighted Linear Stacking (Sill et al., 2009)...")
    m = fit_fwls(X_val, cf_val, cb_val, y_val, fit_idx, mean, std)
    print(f"  n_params={m['n_params']}  (train_time_sec={m['train_time_sec']:.3f})")
    specs.append(("B2_FWLS", "b2_fwls",
                  predict_fwls(m, X_val, cf_val, cb_val),
                  predict_fwls(m, X_test, cf_test, cb_test), m, {}))

    print("\nB3  Ridge Stacking (expert predictions only)...")
    m = fit_ridge_stack(cf_val, cb_val, y_val, fit_idx)
    coef = m["model"].coef_
    print(f"  w_cf={coef[0]:.4f}  w_cb={coef[1]:.4f}  intercept={m['model'].intercept_:.4f}")
    specs.append(("B3_RidgeStack", "b3_ridge_stack",
                  predict_ridge_stack(m, cf_val, cb_val),
                  predict_ridge_stack(m, cf_test, cb_test), m,
                  {"w_cf": float(coef[0]), "w_cb": float(coef[1]),
                   "intercept": float(m["model"].intercept_)}))

    print("\nB4  Gradient-Boosted Stacking ([cf, cb] + 9 gate features)...")
    m = fit_gbm_stack(X_val, cf_val, cb_val, y_val, fit_idx, hold_idx)
    print(f"  holdout RMSE={m['holdout_rmse']:.4f}  (train_time_sec={m['train_time_sec']:.2f})")
    specs.append(("B4_GBMStack", "b4_gbm_stack",
                  predict_gbm_stack(m, X_val, cf_val, cb_val),
                  predict_gbm_stack(m, X_test, cf_test, cb_test), m,
                  {"holdout_rmse": m["holdout_rmse"]}))

    for name, stub, val_pred, test_pred, m, extra in specs:
        v = val_df.copy()
        v["hybrid_pred"] = val_pred
        val_metrics = segmented_rating_metrics(v, true_col="rating", pred_col="hybrid_pred")
        print_segment_table(f"{name} [val]", val_metrics)

        t = test_df.copy()
        t["hybrid_pred"] = test_pred
        test_metrics = segmented_rating_metrics(t, true_col="rating", pred_col="hybrid_pred")
        print_segment_table(f"{name} [test]", test_metrics)

        preds_path = config.PREDICTIONS_DIR / f"{stub}_test.csv"
        t.to_csv(preds_path, index=False)
        print(f"  saved -> {preds_path}")

        results[name] = {
            **extra,
            "val": val_metrics,
            "test": test_metrics,
            "compute_cost": {"train_time_sec": m["train_time_sec"], "n_params": m["n_params"]},
        }
        summary.append((name, test_metrics["overall"]["rmse"]))

    metrics_path = config.METRICS_DIR / "external_baselines.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")

    print("\nExternal baselines, overall TEST RMSE:")
    for name, r in sorted(summary, key=lambda kv: kv[1]):
        print(f"  {name:24s} {r:.4f}")
    print("\nRe-run scripts/08_full_comparison.py to fold these into the main table.")


if __name__ == "__main__":
    main()
