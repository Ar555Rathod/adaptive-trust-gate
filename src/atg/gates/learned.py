"""Model 4 / Learned Gate: a GateNet (logistic-regression-capacity by
default) trained on the sparsity + similarity + CF-confidence features from
atg.gates.features, fit end-to-end against the hybrid blend loss.

Split discipline: the CF/CB experts are fit on TRAIN only (Week 1). Their
predictions on VAL are therefore honest, out-of-sample scores -- unlike
their predictions on TRAIN, which SVD++ in particular has already partly
memorized. So the gate itself is fit on VAL (further subdivided internally
into a gate-train/gate-early-stop split), never on TRAIN, and is evaluated
only on the fully held-out TEST split. This mirrors how Model 3's alpha was
already tuned on VAL and scored on TEST.
"""
import time

import numpy as np
import pandas as pd

from atg.gates.features import FEATURE_COLUMNS, build_gate_features
from atg.gates.nn import GateNet


class LearnedGate:
    def __init__(self, hidden_size: int = 0, l2: float = 1.0, lr: float = 0.05,
                 es_frac: float = 0.2, seed: int = 42):
        self.hidden_size = hidden_size
        self.l2 = l2
        self.lr = lr
        self.es_frac = es_frac
        self.seed = seed
        self.net = None
        self.train_time_sec = None

    def fit(self, val_df: pd.DataFrame, item_popularity: dict, cf_expert, cb_expert,
            epochs: int = 3000, patience: int = 200) -> "LearnedGate":
        start = time.perf_counter()

        feats = build_gate_features(val_df, item_popularity, cf_expert, cb_expert)
        X = feats.to_numpy(dtype=float)
        cf = val_df["cf_pred"].to_numpy(dtype=float)
        cb = val_df["cb_pred"].to_numpy(dtype=float)
        y = val_df["rating"].to_numpy(dtype=float)

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(val_df))
        n_es = int(round(len(val_df) * self.es_frac))
        es_idx, tr_idx = idx[:n_es], idx[n_es:]

        self.net = GateNet(n_features=X.shape[1], hidden_size=self.hidden_size,
                            l2=self.l2, lr=self.lr, seed=self.seed)
        self.net.fit(
            X[tr_idx], cf[tr_idx], cb[tr_idx], y[tr_idx],
            X[es_idx], cf[es_idx], cb[es_idx], y[es_idx],
            epochs=epochs, patience=patience,
        )

        self.train_time_sec = time.perf_counter() - start
        return self

    def g(self, df: pd.DataFrame, item_popularity: dict, cf_expert, cb_expert) -> np.ndarray:
        feats = build_gate_features(df, item_popularity, cf_expert, cb_expert)
        return self.net.g(feats.to_numpy(dtype=float))

    def n_params(self) -> int:
        return self.net.n_params()
