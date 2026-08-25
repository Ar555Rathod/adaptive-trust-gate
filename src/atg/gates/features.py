"""Gate context features: sparsity + similarity + CF-confidence signals that
Models 4-6 condition g(u,i) on (proposal Sec 4.2/4.3, Week 2 milestone).

Leakage check: every feature here is derived either from (a) TRAIN-only
counts/history (user_train_count, item_train_count, the CB expert's
similarity neighbors, which are restricted to train ratings in
ContentBasedExpert.user_train_history_), or (b) the frozen experts' own
predictions on the row being scored (cf_pred/cb_pred), which are themselves
computed by experts fit on train only. No val/test rating ever contributes
to a feature value -- val/test rows only ever appear as the (u,i) query,
never as a source of aggregate statistics.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_user_count",
    "log_item_count",
    "cf_known_user",
    "cf_known_item",
    "cb_max_sim",
    "cb_support",
    "cb_sim_weight_sum",
    "pred_gap",
    "abs_pred_gap",
]


def build_item_popularity(train_df: pd.DataFrame) -> dict:
    return train_df.groupby("movieId").size().to_dict()


def build_gate_features(df: pd.DataFrame, item_popularity: dict, cf_expert, cb_expert,
                         user_col: str = "userId", item_col: str = "movieId") -> pd.DataFrame:
    """`df` must already carry cf_pred, cb_pred, and train_rating_count
    (see atg.utils.segments.attach_segments + scripts/02_train_experts.py).
    """
    user_counts = df["train_rating_count"].to_numpy(dtype=float)
    item_counts = np.array([item_popularity.get(i, 0) for i in df[item_col]], dtype=float)

    cf_known_user = np.array([cf_expert.is_known_user(u) for u in df[user_col]], dtype=float)
    cf_known_item = np.array([cf_expert.is_known_item(i) for i in df[item_col]], dtype=float)

    diagnostics = [cb_expert.similarity_diagnostics(u, i) for u, i in zip(df[user_col], df[item_col])]
    cb_max_sim = np.array([d[0] for d in diagnostics], dtype=float)
    cb_support = np.array([d[1] for d in diagnostics], dtype=float)
    cb_sim_weight_sum = np.array([d[2] for d in diagnostics], dtype=float)

    cf_pred = df["cf_pred"].to_numpy(dtype=float)
    cb_pred = df["cb_pred"].to_numpy(dtype=float)

    feats = pd.DataFrame({
        "log_user_count": np.log1p(user_counts),
        "log_item_count": np.log1p(item_counts),
        "cf_known_user": cf_known_user,
        "cf_known_item": cf_known_item,
        "cb_max_sim": cb_max_sim,
        "cb_support": cb_support,
        "cb_sim_weight_sum": cb_sim_weight_sum,
        "pred_gap": cf_pred - cb_pred,
        "abs_pred_gap": np.abs(cf_pred - cb_pred),
    })
    return feats[FEATURE_COLUMNS]
