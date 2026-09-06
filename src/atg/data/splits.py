"""Build the single 70/15/15 train/val/test split shared by all 7 models.

The split is stratified per-user (each user's own ratings are divided
70/15/15) rather than one global random draw, EXCEPT for a designated quota
of "cold" users, whose train allocation is deliberately capped to a handful
of ratings -- see COLD_USER_FRAC/COLD_TRAIN_MIN/COLD_TRAIN_MAX in
atg.config. This is a simulated cold-start split: MovieLens guarantees every
user has >=20 total ratings, so an unmodified per-user (or global) random
split essentially never produces a user with <5 train ratings, which would
leave the "cold" sparsity segment structurally empty and silently defeat the
project's central cold-vs-power hypothesis. Capping a real quota of users'
train ratings (not fabricating any ratings) reproduces genuine cold-start
conditions using only real interaction data. The same resulting split is
cached to disk and reused by every one of the 7 models.
"""
import numpy as np
import pandas as pd

from atg import config


def build_splits(seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading normalized interactions...")
    ratings = pd.read_csv(config.NORMALIZED_INTERACTIONS_CSV)
    rng = np.random.default_rng(seed)

    print("Identifying cold users...")
    users = ratings["userId"].unique()
    shuffled_users = rng.permutation(users)
    n_cold_users = int(round(len(shuffled_users) * config.COLD_USER_FRAC))
    
    cold_users_array = shuffled_users[:n_cold_users]
    is_cold = ratings["userId"].isin(cold_users_array)

    print("Sorting interactions chronologically...")
    ratings = ratings.sort_values(["userId", "timestamp"])
    
    print("Computing split boundaries...")
    ratings["user_total_ratings"] = ratings.groupby("userId")["userId"].transform("size")
    ratings["chronological_rank"] = ratings.groupby("userId").cumcount()
    
    # Calculate bounds for cold users: they get a random cap between min and max
    random_caps = rng.integers(config.COLD_TRAIN_MIN, config.COLD_TRAIN_MAX + 1, size=n_cold_users)
    cold_cap_series = pd.Series(random_caps, index=cold_users_array)
    
    # Map the cap back to the ratings dataframe
    ratings["cold_cap"] = ratings["userId"].map(cold_cap_series).fillna(0).astype(int)
    cold_n_train = np.minimum(ratings["cold_cap"], ratings["user_total_ratings"])
    cold_n_val = (ratings["user_total_ratings"] - cold_n_train) // 2
    
    # Calculate bounds for regular users
    reg_n_train = np.round(ratings["user_total_ratings"] * config.TRAIN_FRAC).astype(int)
    reg_n_val = np.round(ratings["user_total_ratings"] * config.VAL_FRAC).astype(int)
    
    # Combine based on is_cold
    n_train = np.where(is_cold, cold_n_train, reg_n_train)
    n_val = np.where(is_cold, cold_n_val, reg_n_val)
    
    print("Filtering datasets...")
    rank = ratings["chronological_rank"]
    train_mask = rank < n_train
    val_mask = (rank >= n_train) & (rank < (n_train + n_val))
    test_mask = rank >= (n_train + n_val)
    
    cols = ["userId", "itemId", "rating", "timestamp"]
    train_df = ratings.loc[train_mask, cols].sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = ratings.loc[val_mask, cols].sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = ratings.loc[test_mask, cols].sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Created Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    return train_df, val_df, test_df
