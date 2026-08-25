"""Statistical significance testing between two models' predictions on the
SAME held-out examples (paired comparison) -- needed because a 0.89 vs 0.90
RMSE gap is meaningless without knowing whether it's larger than sampling
noise. Two complementary tests are provided:

  - paired_bootstrap_rmse_diff: resamples test rows with replacement,
    recomputes each model's RMSE on the resample, and reports the
    distribution of (RMSE_a - RMSE_b). A 95% CI that excludes 0 means the
    difference is unlikely to be noise. Non-parametric, no distributional
    assumptions, directly interpretable in RMSE units.
  - wilcoxon_paired: a non-parametric paired test on each example's
    squared error under model A vs model B, testing whether one model's
    per-example errors are systematically smaller. More standard/citable
    in a paper, but only says "significant" vs not, not by how much.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from atg.eval.metrics import rmse


def paired_bootstrap_rmse_diff(y_true, pred_a, pred_b, n_boot: int = 2000,
                                seed: int = 42, ci: float = 0.95) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = rmse(y_true[idx], pred_a[idx]) - rmse(y_true[idx], pred_b[idx])

    alpha = (1 - ci) / 2
    lo, hi = np.percentile(diffs, [alpha * 100, (1 - alpha) * 100])
    p_value = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return {
        "rmse_a": rmse(y_true, pred_a),
        "rmse_b": rmse(y_true, pred_b),
        "mean_diff_a_minus_b": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_level": ci,
        "significant": bool(lo > 0 or hi < 0),
        "p_value_approx": float(min(p_value, 1.0)),
        "n_boot": n_boot,
        "n": n,
    }


def wilcoxon_paired(y_true, pred_a, pred_b) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    err_a = (y_true - np.asarray(pred_a, dtype=float)) ** 2
    err_b = (y_true - np.asarray(pred_b, dtype=float)) ** 2
    diff = err_a - err_b
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "p_value": 1.0, "n": int(len(y_true))}
    stat, p = wilcoxon(err_a, err_b)
    return {"statistic": float(stat), "p_value": float(p), "n": int(len(y_true))}


def compare_models(df: pd.DataFrame, true_col: str, model_preds: dict, segment_col: str | None = None,
                    n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """All-pairs comparison across model_preds = {model_name: pred_col_name},
    optionally repeated per segment. Returns one row per (segment, model_a,
    model_b) with both tests' results.
    """
    rows = []
    groups = [("overall", df)] if segment_col is None else [("overall", df)] + list(df.groupby(segment_col))
    names = list(model_preds.keys())
    for seg_name, sub in groups:
        y = sub[true_col].to_numpy(dtype=float)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pa = sub[model_preds[a]].to_numpy(dtype=float)
                pb = sub[model_preds[b]].to_numpy(dtype=float)
                boot = paired_bootstrap_rmse_diff(y, pa, pb, n_boot=n_boot, seed=seed)
                wil = wilcoxon_paired(y, pa, pb)
                rows.append({
                    "segment": seg_name, "model_a": a, "model_b": b,
                    "rmse_a": boot["rmse_a"], "rmse_b": boot["rmse_b"],
                    "mean_diff": boot["mean_diff_a_minus_b"],
                    "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                    "bootstrap_significant": boot["significant"],
                    "wilcoxon_p": wil["p_value"],
                    "wilcoxon_significant": wil["p_value"] < 0.05,
                    "n": boot["n"],
                })
    return pd.DataFrame(rows)
