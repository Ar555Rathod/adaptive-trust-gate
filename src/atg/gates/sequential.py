"""Model 7 / Sequential Gate: a BiLSTM over each user's long-term and
short-term TRAIN-only rating history, conditioned on the candidate item,
repurposing the long-/short-term interest split from Ahmadian Yazdi et al.
(2024) into a gating mechanism g(u,i,t). This is the temporal-adaptation
axis, independent of the sparsity/similarity features Models 4-6 use --
though the same 9 gate features are still concatenated into the final head,
so the sequence signal is additive on top of, not a replacement for, what
Model 4 already sees.

Leakage/causality: for a query (u,i,t), "history" means only that user's
TRAIN ratings with timestamp STRICTLY BEFORE t, sorted chronologically --
never anything from val/test, and never a future rating relative to the
query even if it happens to be in train. Fallback: per the proposal's build
note, a query with fewer than SEQ_MIN_HISTORY such prior ratings has no
meaningful sequence to encode, so it falls back to Model 3's fixed alpha
rather than running the LSTM on a near-empty/padded sequence.

Per history step, the model sees 4 features: the rating value (normalized),
an exponential recency decay relative to the query time (half-life 180
days), the item's log popularity, and the item's content-similarity to the
CANDIDATE item (so the sequence is genuinely conditioned on i, not just on
u and t) -- reusing the same TF-IDF item vectors as the content-based
expert and Model 4-6's cb_max_sim feature, via ContentBasedExpert.item_vectors.
"""
import bisect
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from atg import config
from atg.gates.features import FEATURE_COLUMNS, build_gate_features

RECENCY_HALF_LIFE_DAYS = 180.0


def build_user_train_sequences(train_df: pd.DataFrame) -> dict:
    """userId -> (timestamps: np.ndarray sorted asc, movieIds: np.ndarray, ratings: np.ndarray),
    all aligned by index. bisect_left on `timestamps` finds the prefix
    strictly before any query timestamp.
    """
    seqs = {}
    for uid, g in train_df.sort_values("timestamp").groupby("userId"):
        seqs[uid] = (
            g["timestamp"].to_numpy(),
            g["movieId"].to_numpy(),
            g["rating"].to_numpy(dtype=float),
        )
    return seqs


def _prior_slice(seqs: dict, user_id, before_ts, max_len: int):
    entry = seqs.get(user_id)
    if entry is None:
        return np.array([]), np.array([]), np.array([])
    timestamps, movie_ids, ratings = entry
    cut = bisect.bisect_left(timestamps, before_ts)
    lo = max(0, cut - max_len)
    return timestamps[lo:cut], movie_ids[lo:cut], ratings[lo:cut]


def prior_history_len(seqs: dict, user_id, before_ts) -> int:
    entry = seqs.get(user_id)
    if entry is None:
        return 0
    return int(bisect.bisect_left(entry[0], before_ts))


def _step_features(cb_expert, item_popularity, query_ts, query_item_id, timestamps, movie_ids, ratings):
    """(len, 4) array: [rating/5, recency_decay, log1p(item_popularity)/10, sim_to_candidate]."""
    n = len(ratings)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    recency_days = (query_ts - timestamps) / 86400.0
    recency_decay = 0.5 ** (np.clip(recency_days, 0, None) / RECENCY_HALF_LIFE_DAYS)
    pop = np.array([item_popularity.get(m, 0) for m in movie_ids], dtype=float)
    log_pop = np.log1p(pop) / 10.0

    sims = np.zeros(n, dtype=float)
    if query_item_id in cb_expert.item_index_:
        target_vec = cb_expert.item_vectors[cb_expert.item_index_[query_item_id]]
        rows = [cb_expert.item_index_[m] for m in movie_ids if m in cb_expert.item_index_]
        keep = [i for i, m in enumerate(movie_ids) if m in cb_expert.item_index_]
        if rows:
            hist_vecs = cb_expert.item_vectors[rows]
            sims[keep] = np.asarray(hist_vecs.dot(target_vec.T).todense()).ravel()

    return np.stack([ratings / 5.0, recency_decay, log_pop, sims], axis=1).astype(np.float32)


def build_sequence_batch(df: pd.DataFrame, seqs: dict, cb_expert, item_popularity,
                          user_col="userId", item_col="movieId", ts_col="timestamp"):
    """Returns padded (left-zero-padded) long/short tensors + lengths + a
    fallback mask (prior history shorter than SEQ_MIN_HISTORY).
    """
    n = len(df)
    long_len, short_len = config.SEQ_LONG_LEN, config.SEQ_SHORT_LEN

    long_feats = np.zeros((n, long_len, 4), dtype=np.float32)
    short_feats = np.zeros((n, short_len, 4), dtype=np.float32)
    long_lens = np.zeros(n, dtype=np.int64)
    short_lens = np.zeros(n, dtype=np.int64)
    fallback_mask = np.zeros(n, dtype=bool)

    for row_i, (uid, iid, ts) in enumerate(zip(df[user_col], df[item_col], df[ts_col])):
        prior_len = prior_history_len(seqs, uid, ts)
        fallback_mask[row_i] = prior_len < config.SEQ_MIN_HISTORY
        if fallback_mask[row_i]:
            continue

        timestamps, movie_ids, ratings = _prior_slice(seqs, uid, ts, long_len)
        feats = _step_features(cb_expert, item_popularity, ts, iid, timestamps, movie_ids, ratings)
        L = len(feats)
        long_feats[row_i, long_len - L:, :] = feats
        long_lens[row_i] = L

        s_timestamps, s_movie_ids, s_ratings = _prior_slice(seqs, uid, ts, short_len)
        s_feats = _step_features(cb_expert, item_popularity, ts, iid, s_timestamps, s_movie_ids, s_ratings)
        S = len(s_feats)
        short_feats[row_i, short_len - S:, :] = s_feats
        short_lens[row_i] = S

    return long_feats, long_lens, short_feats, short_lens, fallback_mask


class BiLSTMGateNet(nn.Module):
    def __init__(self, n_gate_features: int, step_dim: int = 4, hidden_size: int = 16, mlp_hidden: int = 16):
        super().__init__()
        self.long_lstm = nn.LSTM(step_dim, hidden_size, batch_first=True, bidirectional=True)
        self.short_lstm = nn.LSTM(step_dim, hidden_size, batch_first=True, bidirectional=True)
        rep_dim = 4 * hidden_size + n_gate_features  # 2*hidden (long, bidir) + 2*hidden (short, bidir) + gate feats
        self.head = nn.Sequential(
            nn.Linear(rep_dim, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, 1),
        )

    @staticmethod
    def _encode(lstm, x, lengths):
        lengths_clamped = lengths.clamp(min=1)  # avoid pack_padded_sequence choking on 0; masked out below
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_clamped.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = lstm(packed)
        # h_n: (num_directions, batch, hidden) for a 1-layer LSTM -> concat fwd/back
        rep = torch.cat([h_n[0], h_n[1]], dim=-1)
        rep = rep * (lengths > 0).float().unsqueeze(-1)  # zero out truly-empty sequences
        return rep

    def forward(self, long_x, long_len, short_x, short_len, gate_feats):
        long_rep = self._encode(self.long_lstm, long_x, long_len)
        short_rep = self._encode(self.short_lstm, short_x, short_len)
        z = torch.cat([long_rep, short_rep, gate_feats], dim=-1)
        g = torch.sigmoid(self.head(z)).squeeze(-1)
        return g


class SequentialGate:
    """Wraps BiLSTMGateNet + the Model-3 fallback into a single g(u,i,t)."""

    def __init__(self, fallback_alpha: float, seed: int = 42, hidden_size: int = 16,
                 lr: float = 1e-3, weight_decay: float = 1e-4):
        self.fallback_alpha = fallback_alpha
        self.seed = seed
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.net = None
        self.gate_feat_mean_ = None
        self.gate_feat_std_ = None
        self.train_time_sec = None
        self.history_ = []

    def _prep(self, df, seqs, cb_expert, item_popularity, gate_feats_np):
        long_feats, long_lens, short_feats, short_lens, fallback_mask = build_sequence_batch(
            df, seqs, cb_expert, item_popularity)
        return long_feats, long_lens, short_feats, short_lens, fallback_mask, gate_feats_np

    def fit(self, val_df: pd.DataFrame, seqs: dict, item_popularity: dict, cf_expert, cb_expert,
            es_frac: float = 0.2, epochs: int = 60, patience: int = 10, batch_size: int = 512) -> "SequentialGate":
        start = time.perf_counter()
        torch.manual_seed(self.seed)

        gate_feats = build_gate_features(val_df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=np.float32)
        long_feats, long_lens, short_feats, short_lens, fallback_mask = build_sequence_batch(
            val_df, seqs, cb_expert, item_popularity)

        keep = ~fallback_mask  # only rows with enough history actually train the LSTM
        cf = val_df["cf_pred"].to_numpy(dtype=np.float32)[keep]
        cb = val_df["cb_pred"].to_numpy(dtype=np.float32)[keep]
        y = val_df["rating"].to_numpy(dtype=np.float32)[keep]
        gate_feats = gate_feats[keep]
        long_feats, long_lens = long_feats[keep], long_lens[keep]
        short_feats, short_lens = short_feats[keep], short_lens[keep]

        self.gate_feat_mean_ = gate_feats.mean(axis=0)
        self.gate_feat_std_ = gate_feats.std(axis=0)
        self.gate_feat_std_[self.gate_feat_std_ < 1e-8] = 1.0
        gate_feats_n = (gate_feats - self.gate_feat_mean_) / self.gate_feat_std_

        n = len(y)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        n_es = int(round(n * es_frac))
        es_idx, tr_idx = idx[:n_es], idx[n_es:]

        def to_t(arr):
            return torch.as_tensor(arr)

        Xtr = dict(long_x=to_t(long_feats[tr_idx]), long_len=to_t(long_lens[tr_idx]),
                   short_x=to_t(short_feats[tr_idx]), short_len=to_t(short_lens[tr_idx]),
                   gate=to_t(gate_feats_n[tr_idx]).float(), cf=to_t(cf[tr_idx]), cb=to_t(cb[tr_idx]), y=to_t(y[tr_idx]))
        Xes = dict(long_x=to_t(long_feats[es_idx]), long_len=to_t(long_lens[es_idx]),
                   short_x=to_t(short_feats[es_idx]), short_len=to_t(short_lens[es_idx]),
                   gate=to_t(gate_feats_n[es_idx]).float(), cf=to_t(cf[es_idx]), cb=to_t(cb[es_idx]), y=to_t(y[es_idx]))

        self.net = BiLSTMGateNet(n_gate_features=len(FEATURE_COLUMNS), hidden_size=self.hidden_size)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n_tr = len(tr_idx)
        best_es, best_state, no_improve = np.inf, None, 0

        for epoch in range(1, epochs + 1):
            self.net.train()
            perm = torch.randperm(n_tr)
            for start_i in range(0, n_tr, batch_size):
                b = perm[start_i:start_i + batch_size]
                opt.zero_grad()
                g = self.net(Xtr["long_x"][b], Xtr["long_len"][b], Xtr["short_x"][b], Xtr["short_len"][b], Xtr["gate"][b])
                pred = g * Xtr["cf"][b] + (1 - g) * Xtr["cb"][b]
                loss = torch.mean((pred - Xtr["y"][b]) ** 2)
                loss.backward()
                opt.step()

            self.net.eval()
            with torch.no_grad():
                g_tr = self.net(Xtr["long_x"], Xtr["long_len"], Xtr["short_x"], Xtr["short_len"], Xtr["gate"])
                train_mse = torch.mean((g_tr * Xtr["cf"] + (1 - g_tr) * Xtr["cb"] - Xtr["y"]) ** 2).item()
                g_es = self.net(Xes["long_x"], Xes["long_len"], Xes["short_x"], Xes["short_len"], Xes["gate"])
                es_mse = torch.mean((g_es * Xes["cf"] + (1 - g_es) * Xes["cb"] - Xes["y"]) ** 2).item()
            self.history_.append((epoch, train_mse, es_mse))

            if es_mse < best_es - 1e-5:
                best_es, best_state, no_improve = es_mse, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

        self.train_time_sec = time.perf_counter() - start
        return self

    def g(self, df: pd.DataFrame, seqs: dict, item_popularity: dict, cf_expert, cb_expert) -> tuple[np.ndarray, np.ndarray]:
        """Returns (g_values, used_fallback_mask)."""
        gate_feats = build_gate_features(df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=np.float32)
        long_feats, long_lens, short_feats, short_lens, fallback_mask = build_sequence_batch(
            df, seqs, cb_expert, item_popularity)

        g_out = np.full(len(df), self.fallback_alpha, dtype=np.float32)
        keep = ~fallback_mask
        if keep.any():
            gate_feats_n = (gate_feats[keep] - self.gate_feat_mean_) / self.gate_feat_std_
            with torch.no_grad():
                g_lstm = self.net(
                    torch.as_tensor(long_feats[keep]), torch.as_tensor(long_lens[keep]),
                    torch.as_tensor(short_feats[keep]), torch.as_tensor(short_lens[keep]),
                    torch.as_tensor(gate_feats_n).float(),
                ).numpy()
            g_out[keep] = g_lstm
        return g_out, fallback_mask

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))
