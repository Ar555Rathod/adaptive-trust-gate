"""Model 3 / Static Hybrid gate: g(u,i) = alpha, a single constant found by
grid search on the validation set. This is the textbook hybrid baseline
(proposal Sec. 1) that Models 4-7 are meant to improve on, specifically in
the cold/warm segments where one fixed blend ratio can't be optimal for
both a 2-rating user and a 300-rating user.
"""
import time

import numpy as np

from atg.eval.metrics import mae, rmse


class StaticGate:
    def __init__(self, alpha: float | None = None):
        self.alpha = alpha
        self.train_time_sec = None
        self.grid_curve_ = None  # list[(alpha, rmse, mae)]

    def g(self, n: int) -> np.ndarray:
        """Constant gate value, broadcast to n predictions."""
        return np.full(n, self.alpha, dtype=float)

    def fit(self, val_df, cf_col: str = "cf_pred", cb_col: str = "cb_pred",
            true_col: str = "rating", step: float = 0.01, metric: str = "rmse") -> "StaticGate":
        start = time.perf_counter()
        alphas = np.arange(0.0, 1.0 + step / 2, step)
        cf = val_df[cf_col].to_numpy(dtype=float)
        cb = val_df[cb_col].to_numpy(dtype=float)
        y = val_df[true_col].to_numpy(dtype=float)

        curve = []
        best_alpha, best_score = None, np.inf
        for a in alphas:
            pred = np.clip(a * cf + (1 - a) * cb, 0.5, 5.0)
            r, m = rmse(y, pred), mae(y, pred)
            curve.append((float(a), r, m))
            score = r if metric == "rmse" else m
            if score < best_score:
                best_score, best_alpha = score, float(a)

        self.alpha = best_alpha
        self.grid_curve_ = curve
        self.train_time_sec = time.perf_counter() - start
        return self

    def n_params(self) -> int:
        return 1
