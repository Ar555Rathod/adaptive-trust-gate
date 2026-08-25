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
    ratings = pd.read_csv(config.RATINGS_CSV)
    rng = np.random.default_rng(seed)

    users = ratings["userId"].unique()
    shuffled_users = rng.permutation(users)
    n_cold_users = int(round(len(shuffled_users) * config.COLD_USER_FRAC))
    cold_users = set(shuffled_users[:n_cold_users])

    train_parts, val_parts, test_parts = [], [], []
    for uid, group in ratings.groupby("userId", sort=True):
        order = rng.permutation(len(group))
        group = group.iloc[order]
        n = len(group)

        if uid in cold_users:
            n_train = int(rng.integers(config.COLD_TRAIN_MIN, config.COLD_TRAIN_MAX + 1))
            n_train = min(n_train, n)
            remaining = group.iloc[n_train:]
            n_val = len(remaining) // 2
            train_g = group.iloc[:n_train]
            val_g = remaining.iloc[:n_val]
            test_g = remaining.iloc[n_val:]
        else:
            n_train = int(round(n * config.TRAIN_FRAC))
            n_val = int(round(n * config.VAL_FRAC))
            train_g = group.iloc[:n_train]
            val_g = group.iloc[n_train : n_train + n_val]
            test_g = group.iloc[n_train + n_val :]

        train_parts.append(train_g)
        val_parts.append(val_g)
        test_parts.append(test_g)

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

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
