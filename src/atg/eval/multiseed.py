"""Aggregate segmented RMSE/MAE metrics collected across multiple random
seeds into mean +/- std. Point estimates from a single seed can't
distinguish a real effect from noise in the (stochastic) simulated
cold-start split and in gate training -- this turns a list of per-seed
segmented_rating_metrics() outputs into a single mean/std summary per
segment.
"""
import numpy as np


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    rmses = [m["rmse"] for m in metrics_list]
    maes = [m["mae"] for m in metrics_list]
    ns = [m["n"] for m in metrics_list]
    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "n_mean": float(np.mean(ns)),
        "n_seeds": len(metrics_list),
    }


def aggregate_segmented_metrics(list_of_segmented: list[dict]) -> dict:
    segments = set()
    for d in list_of_segmented:
        segments.update(d.keys())
    out = {}
    for seg in segments:
        present = [d[seg] for d in list_of_segmented if seg in d]
        out[seg] = aggregate_metrics(present)
    return out
