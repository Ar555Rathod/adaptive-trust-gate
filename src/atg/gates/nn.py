"""A small gate network: g(u,i) = f(x), with f either a plain linear model
clipped to [0,1] (hidden_size=0 -- logistic-regression-style, the Model 4
default) or one tanh hidden layer + sigmoid output (a genuine MLP).

Both are trained end-to-end against the *hybrid* prediction loss

    L = mean((g*cf + (1-g)*cb - y)^2) + l2 * ||weights||^2

rather than against some proxy target for g itself -- there is no ground
truth for "the right blend weight" for a single rating, only for the
resulting blended prediction, so the loss has to flow through the blend
formula.

For hidden_size=0 the loss is exactly quadratic in the weights (no sigmoid
in the way), since g is linear and clipping only ever binds at the loss
optimum's boundary in rare cases -- so it is solved in one shot via ridge
weighted least squares instead of gradient descent. This matters in
practice: an earlier Adam-based version of this gate converged to a
near-constant g regardless of input (i.e. it collapsed to Model 3's fixed
alpha) because the gradient signal through sigmoid(linear(x)) is very flat
near typical initializations here, and no amount of extra epochs escaped
it. The closed-form solve has no such local-optimum problem and recovers
real per-segment structure (see README/notes on Model 4 for the numbers).
Rewriting the loss: for row i with diff_i = cf_i - cb_i and target
r_i = y_i - cb_i, (g_i*diff_i + cb_i - y_i)^2 = (diff_i * g_i - r_i)^2, so
fitting linear g = Xd @ beta reduces to ordinary least squares of
(diff * Xd) against r -- no division by diff is ever needed, so it's
numerically safe even when cf and cb nearly agree.

For hidden_size>0 there is no closed form (the tanh layer makes g
nonlinear in the weights), so Adam gradient descent is used instead. This
path is reused by Model 6, which searches over hidden_size and the feature
subset via a genetic algorithm.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class GateNet:
    def __init__(self, n_features: int, hidden_size: int = 0, l2: float = 1.0,
                 lr: float = 0.05, seed: int = 42):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.l2 = l2
        self.lr = lr
        rng = np.random.default_rng(seed)

        scale1 = np.sqrt(2.0 / max(n_features, 1))
        if hidden_size > 0:
            self.W1 = rng.normal(0, scale1, size=(n_features, hidden_size))
            self.b1 = np.zeros(hidden_size)
            scale2 = np.sqrt(2.0 / max(hidden_size, 1))
            self.W2 = rng.normal(0, scale2, size=(hidden_size, 1))
        else:
            self.W1 = None
            self.b1 = None
            self.W2 = np.zeros((n_features, 1))
        self.b2 = np.zeros(1)

        self.feat_mean_ = np.zeros(n_features)
        self.feat_std_ = np.ones(n_features)
        self._adam_state = None
        self.history_ = []  # list[(epoch, train_loss, es_loss)]

    def _normalize(self, X):
        return (X - self.feat_mean_) / self.feat_std_

    def _forward(self, Xn):
        if self.hidden_size > 0:
            Z1 = Xn @ self.W1 + self.b1
            A1 = np.tanh(Z1)
            Z2 = A1 @ self.W2 + self.b2
            g = _sigmoid(Z2).ravel()
        else:
            A1 = None
            Z2 = Xn @ self.W2 + self.b2
            g = np.clip(Z2, 0.0, 1.0).ravel()
        return g, A1

    def g(self, X) -> np.ndarray:
        Xn = self._normalize(np.asarray(X, dtype=float))
        g, _ = self._forward(Xn)
        return g

    def n_params(self) -> int:
        n = self.W2.size + self.b2.size
        if self.hidden_size > 0:
            n += self.W1.size + self.b1.size
        return int(n)

    def _hybrid_loss(self, X, cf, cb, y):
        g = self.g(X)
        pred = g * cf + (1 - g) * cb
        return float(np.mean((pred - y) ** 2))

    # ---- linear (hidden_size=0): exact ridge weighted least squares ----

    def _fit_linear_closed_form(self, X, cf, cb, y, X_es, cf_es, cb_es, y_es):
        Xn = self._normalize(X)
        Xd = np.hstack([Xn, np.ones((len(Xn), 1))])

        diff = cf - cb
        r = y - cb
        Xu = Xd * diff[:, None]  # rows scaled by diff -- see module docstring

        A = Xu.T @ Xu + self.l2 * np.eye(Xd.shape[1])
        b = Xu.T @ r
        beta = np.linalg.solve(A, b)

        self.W2 = beta[:-1].reshape(-1, 1)
        self.b2 = beta[-1:]

        train_loss = self._hybrid_loss(X, cf, cb, y)
        es_loss = self._hybrid_loss(X_es, cf_es, cb_es, y_es)
        self.history_ = [(1, train_loss, es_loss)]

    # ---- MLP (hidden_size>0): Adam gradient descent ----

    def _params_and_grads(self, Xn, A1, dL_dz2):
        n = Xn.shape[0]
        grads = {
            "W2": A1.T @ dL_dz2 / n + self.l2 * self.W2,
            "b2": dL_dz2.mean(axis=0),
        }
        dL_dA1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_dA1 * (1 - A1 ** 2)
        grads["W1"] = Xn.T @ dL_dz1 / n + self.l2 * self.W1
        grads["b1"] = dL_dz1.mean(axis=0)
        return grads

    def _init_adam(self):
        self._adam_state = {p: {"m": np.zeros_like(getattr(self, p)), "v": np.zeros_like(getattr(self, p)), "t": 0}
                             for p in ["W1", "b1", "W2", "b2"]}

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        for name, grad in grads.items():
            st = self._adam_state[name]
            st["t"] += 1
            st["m"] = beta1 * st["m"] + (1 - beta1) * grad
            st["v"] = beta2 * st["v"] + (1 - beta2) * (grad ** 2)
            m_hat = st["m"] / (1 - beta1 ** st["t"])
            v_hat = st["v"] / (1 - beta2 ** st["t"])
            update = self.lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, getattr(self, name) - update)

    def _fit_mlp_gd(self, X, cf, cb, y, X_es, cf_es, cb_es, y_es, epochs, patience, log_every):
        Xn = self._normalize(X)
        self._init_adam()

        best_es_loss, best_state, no_improve = np.inf, None, 0

        for epoch in range(1, epochs + 1):
            g, A1 = self._forward(Xn)
            pred = g * cf + (1 - g) * cb
            n = len(y)
            dL_dpred = 2.0 * (pred - y) / n
            dpred_dg = cf - cb
            dL_dg = dL_dpred * dpred_dg
            dg_dz2 = g * (1 - g)
            dL_dz2 = (dL_dg * dg_dz2).reshape(-1, 1)

            grads = self._params_and_grads(Xn, A1, dL_dz2)
            self._adam_step(grads)

            if epoch % log_every == 0 or epoch == epochs:
                train_loss = self._hybrid_loss(X, cf, cb, y)
                es_loss = self._hybrid_loss(X_es, cf_es, cb_es, y_es)
                self.history_.append((epoch, train_loss, es_loss))
                if es_loss < best_es_loss - 1e-6:
                    best_es_loss, best_state, no_improve = es_loss, self._snapshot(), 0
                else:
                    no_improve += 1
                    if no_improve * log_every >= patience:
                        break

        if best_state is not None:
            self._restore(best_state)

    # ---- shared entry point ----

    def fit(self, X, cf, cb, y, X_es, cf_es, cb_es, y_es,
            epochs: int = 3000, patience: int = 200, log_every: int = 100) -> "GateNet":
        X = np.asarray(X, dtype=float)
        cf = np.asarray(cf, dtype=float)
        cb = np.asarray(cb, dtype=float)
        y = np.asarray(y, dtype=float)
        X_es = np.asarray(X_es, dtype=float)
        cf_es = np.asarray(cf_es, dtype=float)
        cb_es = np.asarray(cb_es, dtype=float)
        y_es = np.asarray(y_es, dtype=float)

        self.feat_mean_ = X.mean(axis=0)
        self.feat_std_ = X.std(axis=0)
        self.feat_std_[self.feat_std_ < 1e-8] = 1.0

        if self.hidden_size == 0:
            self._fit_linear_closed_form(X, cf, cb, y, X_es, cf_es, cb_es, y_es)
        else:
            self._fit_mlp_gd(X, cf, cb, y, X_es, cf_es, cb_es, y_es, epochs, patience, log_every)
        return self

    def _snapshot(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(), "W2": self.W2.copy(), "b2": self.b2.copy()}

    def _restore(self, state):
        for k, v in state.items():
            setattr(self, k, v)
