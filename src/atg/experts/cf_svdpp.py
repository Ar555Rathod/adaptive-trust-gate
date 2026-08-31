"""Model 1 / CF expert: SVD++ (Koren, 2008) via scikit-surprise.

This is the shared collaborative-filtering signal used by every one of the
7 models (Unit 3 - Model-Based CF). It is trained exactly once on the train
split; only the gating mechanism differs downstream.
"""
import time

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVDpp

from atg import config


class CFExpertSVDpp:
    def __init__(self, n_factors: int = 20, n_epochs: int = 20, random_state: int = config.RANDOM_SEED):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.algo = SVDpp(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
        self.train_time_sec = None
        self.global_mean_ = None

    def fit(self, train_df: pd.DataFrame) -> "CFExpertSVDpp":
        reader = Reader(rating_scale=(config.RATING_MIN, config.RATING_MAX))
        data = Dataset.load_from_df(train_df[["userId", "itemId", "rating"]], reader)
        trainset = data.build_full_trainset()

        start = time.perf_counter()
        self.algo.fit(trainset)
        self.train_time_sec = time.perf_counter() - start

        self.global_mean_ = trainset.global_mean
        return self

    def predict(self, user_id, item_id) -> float:
        """Raw CF score, clipped to the valid rating range. Unknown
        user/item pairs fall back to surprise's baseline (global mean +
        bias), which is a deliberately weak prediction for cold users --
        that weakness is exactly what the gate is meant to detect and
        compensate for.
        """
        pred = self.algo.predict(user_id, item_id, clip=True)
        return float(pred.est)

    def predict_batch(self, df: pd.DataFrame, user_col: str = "userId", item_col: str = "itemId") -> np.ndarray:
        return np.array([self.predict(u, i) for u, i in zip(df[user_col], df[item_col])])

    def is_known_user(self, user_id) -> bool:
        try:
            self.algo.trainset.to_inner_uid(user_id)
            return True
        except ValueError:
            return False

    def is_known_item(self, item_id) -> bool:
        try:
            self.algo.trainset.to_inner_iid(item_id)
            return True
        except ValueError:
            return False

    def n_params(self) -> int:
        """Total learned scalar parameters -- used later for the
        compute-cost comparison across all 7 models."""
        trainset = self.algo.trainset
        n_users, n_items, f = trainset.n_users, trainset.n_items, self.n_factors
        # bu (n_users) + bi (n_items) + pu (n_users*f) + qi (n_items*f) + yj (n_items*f)
        return n_users + n_items + n_users * f + n_items * f + n_items * f
