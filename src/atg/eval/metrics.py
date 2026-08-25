"""Rating-prediction accuracy metrics.

Only RMSE/MAE are implemented for now (breadth-first pass). Ranking metrics
(Spearman, Kendall, DCG, ARHR, ROC) are added once all 7 models are wired up
end-to-end, per the evaluation plan in the project proposal (Unit 5).
"""
import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rating_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n": int(len(y_true)),
    }


def segmented_rating_metrics(df: pd.DataFrame, true_col: str, pred_col: str,
                              segment_col: str = "segment") -> dict:
    """Overall metrics plus a breakdown per sparsity segment.

    `df` must already carry the segment label for each row (see
    atg.utils.segments.attach_segments).
    """
    out = {"overall": rating_metrics(df[true_col], df[pred_col])}
    for seg, sub in df.groupby(segment_col):
        out[seg] = rating_metrics(sub[true_col], sub[pred_col])
    return out
