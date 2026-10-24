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

    def _fast_tables(self):
        """Per-user augmented factors for vectorised scoring, built once.

        SVD++ scores r_hat = mu + b_u + b_i + q_i . (p_u + |I_u|^-1/2 sum_j y_j),
        and the bracketed term depends only on u -- but surprise recomputes that
        sum on every predict() call, so scoring a split costs one pass over the
        user's rated items per row. Precomputing it once per user turns the whole
        batch into a lookup plus one dot product per row.
        """
        cached = getattr(self, "_fast_", None)
        if cached is not None:
            return cached
        algo, ts = self.algo, self.algo.trainset
        p_aug = np.array(algo.pu, dtype=float, copy=True)
        for inner_u in range(ts.n_users):
            rated = ts.ur[inner_u]
            if rated:
                acc = algo.yj[[j for j, _ in rated]].sum(axis=0)
                p_aug[inner_u] += acc / np.sqrt(len(rated))
        self._fast_ = (p_aug, np.asarray(algo.bu, dtype=float), np.asarray(algo.bi, dtype=float),
                       np.asarray(algo.qi, dtype=float), float(ts.global_mean),
                       ts._raw2inner_id_users, ts._raw2inner_id_items)
        return self._fast_

    def predict_batch(self, df: pd.DataFrame, user_col: str = "userId", item_col: str = "itemId") -> np.ndarray:
        p_aug, bu, bi, qi, mu, u_map, i_map = self._fast_tables()
        iu = np.fromiter((u_map.get(u, -1) for u in df[user_col]), dtype=np.int64, count=len(df))
        ii = np.fromiter((i_map.get(i, -1) for i in df[item_col]), dtype=np.int64, count=len(df))
        known_u, known_i = iu >= 0, ii >= 0

        est = np.full(len(df), mu, dtype=float)
        est[known_u] += bu[iu[known_u]]
        est[known_i] += bi[ii[known_i]]
        both = known_u & known_i
        if both.any():
            est[both] += np.einsum("ij,ij->i", qi[ii[both]], p_aug[iu[both]])
        return np.clip(est, config.RATING_MIN, config.RATING_MAX)

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

    def known_masks(self, users, items):
        """(known_user, known_item) as float arrays, without raising per row.

        is_known_user/is_known_item catch a ValueError from surprise for every
        unknown id, and atg.gates.features called them once per row -- which on
        a cold-start-heavy split means millions of exceptions used as control
        flow. These are plain dict lookups.
        """
        _, _, _, _, _, u_map, i_map = self._fast_tables()
        ku = np.fromiter((u in u_map for u in users), dtype=bool, count=len(users))
        ki = np.fromiter((i in i_map for i in items), dtype=bool, count=len(items))
        return ku.astype(float), ki.astype(float)

    def n_params(self) -> int:
        """Total learned scalar parameters -- used later for the
        compute-cost comparison across all 7 models."""
        trainset = self.algo.trainset
        n_users, n_items, f = trainset.n_users, trainset.n_items, self.n_factors
        # bu (n_users) + bi (n_items) + pu (n_users*f) + qi (n_items*f) + yj (n_items*f)
        return n_users + n_items + n_users * f + n_items * f + n_items * f
