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

    # ---- batched scoring -------------------------------------------------
    #
    # predict() and similarity_diagnostics() each rebuild the user's history
    # and run one sparse matmul PER ROW. Scoring a split therefore costs one
    # matmul per rating, and the gate features pay the same cost a second
    # time. score_batch groups the rows by user instead: a user's history is
    # assembled once, and all of that user's candidate items are scored with a
    # single matmul. The per-row top-k is still done row by row with the same
    # np.argpartition call as _selected_neighbors, so the selected neighbours
    # -- and therefore the numbers -- are unchanged.

    def _history_arrays(self, user_id):
        """(hist_rows, hist_ratings) for a user, cached. Mirrors the keep/order
        logic of _selected_neighbors exactly."""
        cache = getattr(self, "_hist_cache_", None)
        if cache is None:
            cache = self._hist_cache_ = {}
        hit = cache.get(user_id)
        if hit is not None:
            return hit
        history = self.user_train_history_.get(user_id)
        rows, ratings = [], []
        if history:
            for mid, r in history:
                row = self.item_index_.get(mid)
                if row is not None:
                    rows.append(row)
                    ratings.append(r)
        out = (np.asarray(rows, dtype=np.int64), np.asarray(ratings, dtype=float))
        cache[user_id] = out
        return out

    def score_batch(self, df: pd.DataFrame, user_col: str = "userId", item_col: str = "itemId",
                    chunk_pairs: int = 1_000_000):
        """Predictions and similarity diagnostics for a whole frame in one pass.

        Returns (pred, max_sim, support, sim_weight_sum), each length len(df).

        Every (row, history item) pair in the frame is scored with a handful of
        bulk sparse operations rather than one matmul per row. Users here hold
        very few training ratings -- a median of 1 on Goodreads poetry and about
        2 on Amazon -- so per-user batching amortises nothing; what costs is the
        sheer number of tiny sparse row-slices, and this removes them.
        """
        n = len(df)
        users = df[user_col].to_numpy()
        items = df[item_col].to_numpy()
        item_rows = np.array([self.item_index_.get(i, -1) for i in items], dtype=np.int64)

        gm = self.global_mean_
        pred = np.array([self.user_mean_.get(u, gm) for u in users], dtype=float)  # fallback
        max_sim = np.zeros(n, dtype=float)
        support = np.zeros(n, dtype=float)
        wsum = np.zeros(n, dtype=float)

        counts = np.zeros(n, dtype=np.int64)
        hist_rows, hist_ratings = [], []
        for idx in range(n):
            if item_rows[idx] < 0:
                continue
            hr, hrat = self._history_arrays(users[idx])
            if len(hr) == 0:
                continue
            counts[idx] = len(hr)
            hist_rows.append(hr)
            hist_ratings.append(hrat)
        if not hist_rows:
            return pred, max_sim, support, wsum

        scored = np.flatnonzero(counts)
        pair_row = np.repeat(scored, counts[scored])
        pair_hist = np.concatenate(hist_rows)
        pair_rating = np.concatenate(hist_ratings)
        pair_cand = item_rows[pair_row]

        sims = np.empty(len(pair_row), dtype=float)
        for lo in range(0, len(pair_row), chunk_pairs):
            hi = min(lo + chunk_pairs, len(pair_row))
            a = self.item_vectors[pair_cand[lo:hi]]
            b = self.item_vectors[pair_hist[lo:hi]]
            sims[lo:hi] = np.asarray(a.multiply(b).sum(axis=1)).ravel()

        # Top-k within each row: sort by (row, descending sim) and keep the
        # first k of each row, then the same positive-similarity mask.
        order = np.lexsort((-sims, pair_row))
        pr, ps, prat = pair_row[order], sims[order], pair_rating[order]
        starts = np.flatnonzero(np.r_[True, pr[1:] != pr[:-1]])
        rank = np.arange(len(pr)) - np.repeat(starts, np.diff(np.r_[starts, len(pr)]))
        keep = (ps > 0) if self.top_k is None else ((rank < self.top_k) & (ps > 0))

        kr, kw, krat = pr[keep], ps[keep], prat[keep]
        den = np.bincount(kr, weights=kw, minlength=n)
        num = np.bincount(kr, weights=kw * krat, minlength=n)
        cnt = np.bincount(kr, weights=np.ones(len(kr)), minlength=n)

        has = den > 0
        vals = np.divide(num, den, out=np.zeros(n), where=has)
        pred[has] = np.clip(vals[has], config.RATING_MIN, config.RATING_MAX)
        wsum[:] = den
        support[:] = cnt
        np.maximum.at(max_sim, kr, kw)
        return pred, max_sim, support, wsum

    def predict_batch(self, df: pd.DataFrame, user_col: str = "userId", item_col: str = "itemId") -> np.ndarray:
        return self.score_batch(df, user_col, item_col)[0]

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
