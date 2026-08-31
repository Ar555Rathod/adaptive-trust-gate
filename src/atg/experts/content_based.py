"""Model 2 / Content-Based expert (Unit 4).

Item profiles are built from genres (movies.csv) plus aggregated
user-applied tags (tags.csv) -- both describe the *item*, not any specific
rating, so folding tags in does not leak train/val/test rating labels.
Item-item cosine similarity over TF-IDF vectors drives a memory-based
content-KNN: a user's score for item i is the similarity-weighted average
of their own TRAIN-only ratings on the most similar items they've rated.
Restricting that history to train-only is the leakage check called out in
the proposal's Week 2 milestone.
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from atg import config


def _soup(genres: str, tags: str) -> str:
    genre_tokens = genres.replace("|", " ").replace("(no genres listed)", "")
    return f"{genre_tokens} {tags}".strip()


class ContentBasedExpert:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.train_time_sec = None
        self.global_mean_ = None
        self.user_mean_ = {}
        self.item_index_ = {}   # itemId -> row index in self.item_vectors
        self.item_vectors = None
        self.user_train_history_ = {}  # userId -> list[(itemId, rating)]

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame) -> "ContentBasedExpert":
        start = time.perf_counter()

        items = items_df.reset_index(drop=True)
        self.item_index_ = {mid: idx for idx, mid in enumerate(items["itemId"])}
        corpus = items["metadata_text"].fillna("").tolist()
        self.item_vectors = self.vectorizer.fit_transform(corpus)  # rows are L2-normalized by default

        self.global_mean_ = float(train_df["rating"].mean())
        self.user_mean_ = train_df.groupby("userId")["rating"].mean().to_dict()
        self.user_train_history_ = {
            uid: list(zip(g["itemId"], g["rating"]))
            for uid, g in train_df.groupby("userId")
        }

        self.train_time_sec = time.perf_counter() - start
        return self

    def _fallback(self, user_id) -> float:
        return self.user_mean_.get(user_id, self.global_mean_)

    def _selected_neighbors(self, user_id, item_id):
        """Top-k (by cosine sim) of the user's TRAIN-rated items relative to
        item_id, restricted to positive similarity. Returns (sims, ratings)
        arrays, both possibly empty. Shared by predict() and
        similarity_diagnostics() so the two never drift out of sync.
        """
        empty = (np.array([]), np.array([]))
        if item_id not in self.item_index_:
            return empty
        history = self.user_train_history_.get(user_id)
        if not history:
            return empty

        target_vec = self.item_vectors[self.item_index_[item_id]]
        hist_item_ids = [mid for mid, _ in history]
        hist_ratings = np.array([r for _, r in history], dtype=float)
        keep = [i for i, mid in enumerate(hist_item_ids) if mid in self.item_index_]
        if not keep:
            return empty
        hist_ratings = hist_ratings[keep]
        hist_rows = [self.item_index_[hist_item_ids[i]] for i in keep]

        hist_vecs = self.item_vectors[hist_rows]
        sims = np.asarray(hist_vecs.dot(target_vec.T).todense()).ravel()  # cosine, since rows are L2-normalized

        if self.top_k is not None and len(sims) > self.top_k:
            top_idx = np.argpartition(sims, -self.top_k)[-self.top_k:]
        else:
            top_idx = np.arange(len(sims))

        sel_sims = sims[top_idx]
        sel_ratings = hist_ratings[top_idx]
        mask = sel_sims > 0
        return sel_sims[mask], sel_ratings[mask]

    def predict(self, user_id, item_id) -> float:
        w, r = self._selected_neighbors(user_id, item_id)
        if len(w) == 0:
            return self._fallback(user_id)
        score = float(np.dot(w, r) / w.sum())
        return float(np.clip(score, config.RATING_MIN, config.RATING_MAX))

    def predict_batch(self, df: pd.DataFrame, user_col: str = "userId", item_col: str = "itemId") -> np.ndarray:
        return np.array([self.predict(u, i) for u, i in zip(df[user_col], df[item_col])])

    def similarity_diagnostics(self, user_id, item_id) -> tuple[float, int, float]:
        """(max_similarity, n_positive_neighbors, sum_of_similarity_weights)
        for the (user, item) pair -- used as CB-confidence gate features:
        a high max-sim/support means the content prediction rests on
        genuinely similar items the user has rated, not just a fallback
        mean.
        """
        w, _ = self._selected_neighbors(user_id, item_id)
        if len(w) == 0:
            return 0.0, 0, 0.0
        return float(w.max()), int(len(w)), float(w.sum())

    def n_params(self) -> int:
        """Not a learned parametric model -- report representation size
        (vocab size x n_items) as the compute-cost proxy."""
        return int(self.item_vectors.shape[0] * self.item_vectors.shape[1])
