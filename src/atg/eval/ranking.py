"""Ranking metrics (Unit 5): Spearman/Kendall rank correlation, DCG/NDCG,
ARHR, and ROC-AUC. All are inherently per-user list metrics (except ROC-AUC,
computed globally as a binary relevant/irrelevant discrimination score) so
each takes a dataframe with one row per (user, item) test example and
groups by user internally.
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score


def mean_rank_correlation(df: pd.DataFrame, true_col: str, pred_col: str,
                           user_col: str = "userId", min_items: int = 2) -> dict:
    """Spearman rho and Kendall tau between true and predicted ratings,
    computed within each user's test items then averaged across users who
    have enough items (>=2, distinct values) for a correlation to be
    defined. Measures whether the *ranking* of a user's items is right,
    not just the raw score.
    """
    rhos, taus, n_users = [], [], 0
    for _, g in df.groupby(user_col):
        if len(g) < min_items or g[true_col].nunique() < 2 or g[pred_col].nunique() < 2:
            continue
        rho, _ = spearmanr(g[true_col], g[pred_col])
        tau, _ = kendalltau(g[true_col], g[pred_col])
        if np.isnan(rho) or np.isnan(tau):
            continue
        rhos.append(rho)
        taus.append(tau)
        n_users += 1
    return {
        "spearman": float(np.mean(rhos)) if rhos else float("nan"),
        "kendall": float(np.mean(taus)) if taus else float("nan"),
        "n_users": n_users,
    }


def _dcg(relevances: np.ndarray, k: int, gain: str = "linear") -> float:
    relevances = relevances[:k]
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    gains = relevances if gain == "linear" else (2.0 ** relevances - 1.0)
    return float(np.sum(gains / discounts))


def ndcg_at_k(df: pd.DataFrame, true_col: str, pred_col: str, user_col: str = "userId",
              k: int = 10, gain: str = "linear", min_items: int = 2) -> dict:
    """NDCG@k: are the items the user actually rated highly ranked near the
    top of the list ordered by predicted score? Graded relevance = the true
    rating itself (linear gain) by default.
    """
    scores, n_users = [], 0
    for _, g in df.groupby(user_col):
        if len(g) < min_items:
            continue
        ranked_by_pred = g.sort_values(pred_col, ascending=False)[true_col].to_numpy(dtype=float)
        ideal = np.sort(g[true_col].to_numpy(dtype=float))[::-1]
        idcg = _dcg(ideal, k, gain)
        if idcg <= 0:
            continue
        dcg = _dcg(ranked_by_pred, k, gain)
        scores.append(dcg / idcg)
        n_users += 1
    return {
        "ndcg": float(np.mean(scores)) if scores else float("nan"),
        "k": k,
        "n_users": n_users,
    }


def arhr_at_k(df: pd.DataFrame, true_col: str, pred_col: str, user_col: str = "userId",
              k: int = 10, relevance_threshold: float = 4.0) -> dict:
    """Average Reciprocal Hit Rank: for each user, rank their test items by
    predicted score, look at the top k, and score 1/rank of the first
    truly-relevant item found (relevant = true rating >= threshold). Users
    with no relevant items in their test set are excluded (there is nothing
    to "hit"). Measures how quickly a relevant item surfaces, not just
    whether it's present.
    """
    scores, n_users = [], 0
    for _, g in df.groupby(user_col):
        relevant_mask = g[true_col].to_numpy(dtype=float) >= relevance_threshold
        if not relevant_mask.any():
            continue
        order = g[pred_col].to_numpy(dtype=float).argsort()[::-1][:k]
        ranked_relevant = relevant_mask[order]
        hit_positions = np.flatnonzero(ranked_relevant)
        rr = 1.0 / (hit_positions[0] + 1) if len(hit_positions) > 0 else 0.0
        scores.append(rr)
        n_users += 1
    return {
        "arhr": float(np.mean(scores)) if scores else float("nan"),
        "k": k,
        "relevance_threshold": relevance_threshold,
        "n_users": n_users,
    }


def roc_auc(df: pd.DataFrame, true_col: str, pred_col: str,
            relevance_threshold: float = 4.0) -> dict:
    """Global ROC-AUC treating predicted score as the ranking score and
    true rating >= threshold as the positive class -- overall
    relevant-vs-irrelevant discrimination ability (proposal Sec 5.2)."""
    y = (df[true_col].to_numpy(dtype=float) >= relevance_threshold).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"roc_auc": float("nan"), "relevance_threshold": relevance_threshold, "n": int(len(y))}
    auc = roc_auc_score(y, df[pred_col].to_numpy(dtype=float))
    return {"roc_auc": float(auc), "relevance_threshold": relevance_threshold, "n": int(len(y))}


def ranking_metrics(df: pd.DataFrame, true_col: str, pred_col: str, user_col: str = "userId",
                     k: int = 10, relevance_threshold: float = 4.0) -> dict:
    out = {}
    out.update(mean_rank_correlation(df, true_col, pred_col, user_col))
    out.update(ndcg_at_k(df, true_col, pred_col, user_col, k=k))
    out.update(arhr_at_k(df, true_col, pred_col, user_col, k=k, relevance_threshold=relevance_threshold))
    out.update(roc_auc(df, true_col, pred_col, relevance_threshold=relevance_threshold))
    return out


def segmented_ranking_metrics(df: pd.DataFrame, true_col: str, pred_col: str,
                               user_col: str = "userId", segment_col: str = "segment",
                               k: int = 10, relevance_threshold: float = 4.0) -> dict:
    out = {"overall": ranking_metrics(df, true_col, pred_col, user_col, k, relevance_threshold)}
    for seg, sub in df.groupby(segment_col):
        out[seg] = ranking_metrics(sub, true_col, pred_col, user_col, k, relevance_threshold)
    return out
