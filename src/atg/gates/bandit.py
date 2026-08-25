"""Model 5 / Bandit Gate: a contextual multi-armed bandit that selects a
blend weight alpha from a small discrete set of "arms" per (u,i), using the
same sparsity/similarity/CF-confidence context features as Model 4
(atg.gates.features). Two standard strategies are implemented -- disjoint
LinUCB (Li et al., 2010) and an epsilon-greedy variant sharing the same
per-arm ridge reward model -- directly tying to the RL/bandit coursework
(UCB, epsilon-greedy, regret) called out in the proposal's escalation
rationale.

Unlike Model 4 (a single batch fit on val), this gate updates online: every
interaction it processes produces a prediction AND immediately updates that
arm's linear reward model from the observed true rating, one interaction at
a time. That sequential, continues-after-deployment update loop is the
actual escalation over Model 4, not just having a bigger context.

Reward for arm k on context x, having produced blended prediction
pred_k = alpha_k*cf + (1-alpha_k)*cb, is reward = -(pred_k - y)^2 (negative
squared error, so maximizing reward means minimizing rating-prediction
error). Each arm keeps a disjoint ridge-regression reward model (A_k, b_k);
LinUCB's arm choice adds an uncertainty bonus (sqrt(x^T A_k^-1 x)) on top of
the point estimate so under-explored arms still get tried occasionally,
rather than the bandit locking onto an early lucky guess.
"""
import numpy as np


class ContextualBanditGate:
    def __init__(self, n_features: int, arms=None, strategy: str = "ucb",
                 ucb_alpha: float = 1.0, epsilon: float = 0.1,
                 ridge_lambda: float = 1.0, seed: int = 42):
        self.arms = np.asarray(arms if arms is not None else np.linspace(0.0, 1.0, 11))
        self.n_arms = len(self.arms)
        self.d = n_features + 1  # +1 for the intercept term
        self.strategy = strategy
        self.ucb_alpha = ucb_alpha
        self.epsilon = epsilon
        self.ridge_lambda = ridge_lambda
        self.rng = np.random.default_rng(seed)

        self.A_inv = np.stack([np.eye(self.d) / ridge_lambda for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.d))
        self.n_pulls = np.zeros(self.n_arms, dtype=int)

    def _augment(self, x):
        return np.concatenate([np.asarray(x, dtype=float), [1.0]])

    def _theta(self, k):
        return self.A_inv[k] @ self.b[k]

    def select_arm(self, x, explore: bool = True) -> int:
        xa = self._augment(x)
        if self.strategy == "epsilon_greedy":
            if explore and self.rng.random() < self.epsilon:
                return int(self.rng.integers(self.n_arms))
            scores = np.array([xa @ self._theta(k) for k in range(self.n_arms)])
            return int(np.argmax(scores))

        scores = np.empty(self.n_arms)
        for k in range(self.n_arms):
            mean_k = xa @ self._theta(k)
            bonus_k = self.ucb_alpha * np.sqrt(max(float(xa @ self.A_inv[k] @ xa), 0.0)) if explore else 0.0
            scores[k] = mean_k + bonus_k
        return int(np.argmax(scores))

    def update(self, k: int, x, reward: float) -> None:
        xa = self._augment(x)
        Ainv = self.A_inv[k]
        Ainv_x = Ainv @ xa
        denom = 1.0 + xa @ Ainv_x
        self.A_inv[k] = Ainv - np.outer(Ainv_x, Ainv_x) / denom  # Sherman-Morrison rank-1 update
        self.b[k] += reward * xa
        self.n_pulls[k] += 1

    def n_params(self) -> int:
        return int(self.A_inv.size + self.b.size)


def run_stream(bandit: ContextualBanditGate, X, cf, cb, y, explore: bool = True,
               clip=(0.5, 5.0)):
    """Feed (context, cf_pred, cb_pred, true_rating) rows through the bandit
    one at a time: select an arm, form the blended prediction, observe the
    true rating, update. Returns per-step predictions, chosen alphas, and
    squared errors -- the raw material for both the frozen-eval metrics and
    the learning-curve plot.
    """
    n = len(y)
    preds = np.empty(n)
    chosen_alpha = np.empty(n)
    sq_err = np.empty(n)
    for t in range(n):
        x = X[t]
        k = bandit.select_arm(x, explore=explore)
        alpha = bandit.arms[k]
        pred = float(np.clip(alpha * cf[t] + (1 - alpha) * cb[t], *clip))
        reward = -((pred - y[t]) ** 2)
        if explore:
            bandit.update(k, x, reward)
        preds[t] = pred
        chosen_alpha[t] = alpha
        sq_err[t] = (pred - y[t]) ** 2
    return preds, chosen_alpha, sq_err
