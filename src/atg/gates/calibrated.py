"""Model 8 / Calibrated Trust Gate: the Learned Gate plus a global correction.

    r_hat(u,i) = a + s * [ g(x)*cf + (1 - g(x))*cb ],     g(x) in [0, 1]

Why: the multi-seed decomposition on Goodreads poetry found two roughly equal,
additive sources of gain over the static blend -- conditioning the weight on
context (what Model 4 captures, -0.0012 RMSE) and escaping the convex
constraint, i.e. allowing an intercept and weights that need not sum to 1
(what Ridge Stack captures, -0.0011). Model 4 can only have the first; FWLS
gets both but its weights no longer form a share. Model 8 keeps g as a
bounded trust share -- "how much this prediction trusts CF" -- and moves the
shrinkage gain into two GLOBAL scalars: a shift a and a scale s. One number
each, applied identically to every prediction, so the reading of g survives.

Optionally the gate also sees user- and item-count bins and their log
interaction. These were added to target an observed miscalibration of Model
4's weight by rating count; on the seed-42 prototype they contributed only
0.0002 of the 0.0014 gain, so the multi-seed run records both variants and
three seeds decide whether they earn their place.

Fitting: for fixed (a, s) the loss is quadratic in the gate's weights, so g is
solved in closed form exactly as Model 4 does (see atg.gates.nn); for fixed g,
(a, s) is ordinary least squares. The two steps alternate. l2 is chosen on the
same 20% val holdout Model 4 uses for early stopping, and the final model is
the one fit on the remaining 80% -- the same rows Model 4 is fit on.
"""
import time

import numpy as np

from atg import config
from atg.gates.features import FEATURE_COLUMNS, build_gate_features

# Upper-exclusive edges; np.digitize maps a count to its bin index.
USER_COUNT_EDGES = (1, 2, 3, 5, 10, 50)       # 0 | 1 | 2 | 3-4 | 5-9 | 10-49 | 50+
ITEM_COUNT_EDGES = (1, 5, 20, 100, 500)       # 0 | 1-4 | 5-19 | 20-99 | 100-499 | 500+

_I_USER = FEATURE_COLUMNS.index("log_user_count")
_I_ITEM = FEATURE_COLUMNS.index("log_item_count")

# Guards the g-step target (y - a) / s against a degenerate scale fit.
_MIN_SCALE = 0.05


def count_features(X_base: np.ndarray) -> np.ndarray:
    """One-hot user- and item-count bins (first level dropped; the gate's bias
    covers it) plus log(1+user count) * log(1+item count).

    Counts are recovered from the base features rather than passed in, because
    atg.gates.features defines log_user_count = log1p(train_rating_count) and
    log_item_count = log1p(item train count) exactly -- so this works on any
    precomputed feature matrix without the source frame.
    """
    lu, li = X_base[:, _I_USER], X_base[:, _I_ITEM]
    u, i = np.rint(np.expm1(lu)), np.rint(np.expm1(li))
    U = np.eye(len(USER_COUNT_EDGES) + 1)[np.digitize(u, USER_COUNT_EDGES)][:, 1:]
    I = np.eye(len(ITEM_COUNT_EDGES) + 1)[np.digitize(i, ITEM_COUNT_EDGES)][:, 1:]
    return np.hstack([U, I, (lu * li)[:, None]])


class CalibratedGate:
    def __init__(self, use_count_features: bool = True, l2_grid=(0.1, 1.0, 10.0),
                 iters: int = 10, es_frac: float = 0.2, seed: int = config.RANDOM_SEED):
        self.use_count_features = use_count_features
        self.l2_grid = tuple(l2_grid)
        self.iters = iters
        self.es_frac = es_frac
        self.seed = seed
        self.l2 = None
        self.a, self.s = 0.0, 1.0
        self.mu_ = self.sd_ = self.beta_ = None
        self.holdout_rmse_ = None
        self.train_time_sec = None

    # ---- design --------------------------------------------------------------
    def _design(self, X_base):
        X_base = np.asarray(X_base, dtype=float)
        return np.hstack([X_base, count_features(X_base)]) if self.use_count_features else X_base

    def _g_from_design(self, X, mu, sd, beta):
        Xd = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
        return np.clip(Xd @ beta, 0.0, 1.0)

    # ---- fitting -------------------------------------------------------------
    def _fit_one(self, X, cf, cb, y, l2):
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xd = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
        diff = cf - cb
        Xu = Xd * diff[:, None]          # rows scaled by (cf - cb); see atg.gates.nn
        A = Xu.T @ Xu + l2 * np.eye(Xd.shape[1])
        a, s = 0.0, 1.0
        for _ in range(self.iters):
            # g-step: (a + s*(g*diff + cb) - y)^2 = s^2 * (g*diff - t)^2
            t = (y - a) / s - cb
            beta = np.linalg.solve(A, Xu.T @ t)
            # (a, s)-step: ordinary least squares of y on the clipped blend
            h = self._g_from_design(X, mu, sd, beta) * diff + cb
            a, s = np.linalg.lstsq(np.column_stack([np.ones_like(h), h]), y, rcond=None)[0]
            s = max(float(s), _MIN_SCALE)
        return mu, sd, beta, float(a), float(s)

    def fit_arrays(self, X_base, cf, cb, y) -> "CalibratedGate":
        """Fit from a precomputed base feature matrix (FEATURE_COLUMNS order)."""
        start = time.perf_counter()
        X = self._design(X_base)
        cf, cb, y = (np.asarray(v, dtype=float) for v in (cf, cb, y))

        # Same partition as atg.gates.learned.LearnedGate.
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        n_es = int(round(len(y) * self.es_frac))
        hold, fit = idx[:n_es], idx[n_es:]

        best = None
        for l2 in self.l2_grid:
            mu, sd, beta, a, s = self._fit_one(X[fit], cf[fit], cb[fit], y[fit], l2)
            g = self._g_from_design(X[hold], mu, sd, beta)
            pred = np.clip(a + s * (g * cf[hold] + (1 - g) * cb[hold]),
                           config.RATING_MIN, config.RATING_MAX)
            err = float(np.sqrt(np.mean((pred - y[hold]) ** 2)))
            if best is None or err < best[0]:
                best = (err, l2, mu, sd, beta, a, s)

        self.holdout_rmse_, self.l2, self.mu_, self.sd_, self.beta_, self.a, self.s = best
        self.train_time_sec = time.perf_counter() - start
        return self

    def fit(self, val_df, item_popularity: dict, cf_expert, cb_expert) -> "CalibratedGate":
        start = time.perf_counter()
        X = build_gate_features(val_df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=float)
        self.fit_arrays(X, val_df["cf_pred"], val_df["cb_pred"], val_df["rating"])
        # Report the cost including feature construction, as Model 4 does.
        self.train_time_sec = time.perf_counter() - start
        return self

    # ---- inference -----------------------------------------------------------
    def g_arrays(self, X_base) -> np.ndarray:
        return self._g_from_design(self._design(X_base), self.mu_, self.sd_, self.beta_)

    def predict_arrays(self, X_base, cf, cb):
        """Returns (prediction, g). g is the trust share before the global correction."""
        g = self.g_arrays(X_base)
        cf, cb = np.asarray(cf, dtype=float), np.asarray(cb, dtype=float)
        pred = np.clip(self.a + self.s * (g * cf + (1 - g) * cb),
                       config.RATING_MIN, config.RATING_MAX)
        return pred, g

    def predict(self, df, item_popularity: dict, cf_expert, cb_expert):
        X = build_gate_features(df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=float)
        return self.predict_arrays(X, df["cf_pred"], df["cb_pred"])

    def n_params(self) -> int:
        # gate weights + bias, plus the global shift and scale
        return int(len(self.beta_) + 2)
