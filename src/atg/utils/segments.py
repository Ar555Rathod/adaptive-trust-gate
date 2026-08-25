"""User sparsity-segment assignment (cold / warm / power)."""
import numpy as np
import pandas as pd

from atg import config


def segment_for_count(count: int) -> str:
    if count < config.COLD_MAX:
        return "cold"
    if count < config.WARM_MAX:
        return "warm"
    return "power"


def build_user_segments(train_df: pd.DataFrame) -> pd.DataFrame:
    """One row per user who appears in TRAIN, with their train rating count
    and sparsity segment. Users with zero train ratings (all their ratings
    fell into val/test) are handled separately by callers as segment='cold'
    with count=0 -- they are the hardest cold-start case.
    """
    counts = train_df.groupby("userId").size().rename("train_rating_count").reset_index()
    counts["segment"] = counts["train_rating_count"].apply(segment_for_count)
    return counts


def attach_segments(df: pd.DataFrame, user_segments: pd.DataFrame) -> pd.DataFrame:
    """Left-join segment info onto any ratings dataframe (e.g. val/test).
    Users absent from train (train_rating_count=0) are labeled 'cold'.
    """
    out = df.merge(user_segments, on="userId", how="left")
    out["train_rating_count"] = out["train_rating_count"].fillna(0).astype(int)
    out["segment"] = out["segment"].fillna("cold")
    return out
