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
    # Filter users with < min_items
    counts = df[user_col].value_counts()
    valid_users = counts[counts >= min_items].index
    if len(valid_users) == 0:
        return {"spearman": float("nan"), "kendall": float("nan"), "n_users": 0}

    valid_df = df[df[user_col].isin(valid_users)]
    
    # Filter users who don't have variance in true or pred ratings
    std = valid_df.groupby(user_col)[[true_col, pred_col]].std()
    variance_users = std[(std[true_col] > 0) & (std[pred_col] > 0)].index
    valid_df = valid_df[valid_df[user_col].isin(variance_users)]
    
    if len(valid_df) == 0:
        return {"spearman": float("nan"), "kendall": float("nan"), "n_users": 0}

    # Vectorized Spearman via Pandas corr
    spearman_corrs = valid_df.groupby(user_col)[[true_col, pred_col]].corr(method="spearman")
    # corr() returns a MultiIndex (userId, true_col/pred_col) x (true_col, pred_col)
    # We grab the cross-correlation values
    rhos = spearman_corrs.xs(true_col, level=1)[pred_col].dropna()
    
    # Vectorized Kendall via Pandas corr
    kendall_corrs = valid_df.groupby(user_col)[[true_col, pred_col]].corr(method="kendall")
    taus = kendall_corrs.xs(true_col, level=1)[pred_col].dropna()
    
    return {
        "spearman": float(rhos.mean()) if len(rhos) else float("nan"),
        "kendall": float(taus.mean()) if len(taus) else float("nan"),
        "n_users": len(variance_users),
    }


def ndcg_at_k(df: pd.DataFrame, true_col: str, pred_col: str, user_col: str = "userId",
              k: int = 10, gain: str = "linear", min_items: int = 2) -> dict:
    """NDCG@k: are the items the user actually rated highly ranked near the
    top of the list ordered by predicted score? Graded relevance = the true
    rating itself (linear gain) by default.
    """
    counts = df[user_col].value_counts()
    valid_users = counts[counts >= min_items].index
    if len(valid_users) == 0:
        return {"ndcg": float("nan"), "k": k, "n_users": 0}
        
    df = df[df[user_col].isin(valid_users)].copy()
    
    # --- Compute Ideal DCG (IDCG) ---
    ideal = df.sort_values([user_col, true_col], ascending=[True, False])
    ideal["rank"] = ideal.groupby(user_col).cumcount() + 1
    ideal = ideal[ideal["rank"] <= k]
    
    gains = ideal[true_col] if gain == "linear" else (2.0 ** ideal[true_col] - 1.0)
    discounts = np.log2(ideal["rank"] + 1)
    ideal["dcg"] = gains / discounts
    idcg = ideal.groupby(user_col)["dcg"].sum()
    
    # Filter users with IDCG > 0
    valid_idcg_users = idcg[idcg > 0].index
    if len(valid_idcg_users) == 0:
        return {"ndcg": float("nan"), "k": k, "n_users": 0}
        
    # --- Compute Actual DCG ---
    actual = df[df[user_col].isin(valid_idcg_users)].sort_values([user_col, pred_col], ascending=[True, False])
    actual["rank"] = actual.groupby(user_col).cumcount() + 1
    actual = actual[actual["rank"] <= k]
    
    gains = actual[true_col] if gain == "linear" else (2.0 ** actual[true_col] - 1.0)
    discounts = np.log2(actual["rank"] + 1)
    actual["dcg"] = gains / discounts
    dcg = actual.groupby(user_col)["dcg"].sum()
    
    # Realign index and calculate NDCG
    dcg = dcg.reindex(idcg.index).fillna(0.0)
    ndcg = (dcg / idcg).dropna()
    
    return {
        "ndcg": float(ndcg.mean()) if len(ndcg) else float("nan"),
        "k": k,
        "n_users": len(valid_idcg_users),
    }


def arhr_at_k(df: pd.DataFrame, true_col: str, pred_col: str, user_col: str = "userId",
              k: int = 10, relevance_threshold: float = 4.0) -> dict:
    """Average Reciprocal Hit Rank: for each user, rank their test items by
    predicted score, look at the top k, and score 1/rank of the first
    truly-relevant item found (relevant = true rating >= threshold).
    """
    relevant_mask = df[true_col] >= relevance_threshold
    has_relevant_users = df[relevant_mask][user_col].unique()
    if len(has_relevant_users) == 0:
        return {"arhr": float("nan"), "k": k, "relevance_threshold": relevance_threshold, "n_users": 0}
        
    df = df[df[user_col].isin(has_relevant_users)].copy()
    
    # Sort by predicted score descending to get rankings
    actual = df.sort_values([user_col, pred_col], ascending=[True, False])
    actual["rank"] = actual.groupby(user_col).cumcount() + 1
    
    # Consider only top-k for hits
    top_k = actual[actual["rank"] <= k]
    
    # Filter only relevant items in top-k
    hits = top_k[top_k[true_col] >= relevance_threshold]
    
    if len(hits) == 0:
        return {"arhr": 0.0, "k": k, "relevance_threshold": relevance_threshold, "n_users": len(has_relevant_users)}
    
    # Find the best (minimum) rank of a relevant item per user
    first_hit_rank = hits.groupby(user_col)["rank"].min()
    
    scores = pd.Series(0.0, index=has_relevant_users)
    scores.loc[first_hit_rank.index] = 1.0 / first_hit_rank
    
    return {
        "arhr": float(scores.mean()),
        "k": k,
        "relevance_threshold": relevance_threshold,
        "n_users": len(has_relevant_users),
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
