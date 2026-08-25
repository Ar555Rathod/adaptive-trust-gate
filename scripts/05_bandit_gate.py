"""Model 5 / Bandit Gate.

Two runs, per the proposal's evaluation plan:

1. "Deployed" eval, comparable to Models 1-4: a bandit learns online by
   streaming through VAL in timestamp order, is then frozen (greedy, no
   more exploration/updates) and scored on TEST with the same segmented
   RMSE/MAE table as every other model.

2. Sequential simulation for the learning-curve plot: a FRESH bandit
   (cold start) streams through VAL+TEST combined, sorted by timestamp,
   continuing to explore/update the whole way -- this is what "genuine
   online adaptation" actually looks like, and is evaluated separately
   from (1) because letting it keep exploring on what would otherwise be
   the frozen comparison set breaks the apples-to-apples table.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atg import config
from atg.data.splits import load_splits
from atg.gates.features import build_item_popularity, build_gate_features, FEATURE_COLUMNS
from atg.gates.bandit import ContextualBanditGate, run_stream
from atg.eval.metrics import segmented_rating_metrics, rmse


def print_segment_table(name, metrics):
    print(f"\n[{name}]")
    for seg in ("overall", "cold", "warm", "power"):
        if seg in metrics:
            m = metrics[seg]
            print(f"  {seg:8s} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  n={m['n']}")


def rolling_rmse(sq_err: np.ndarray, window: int = 500) -> np.ndarray:
    cum = np.cumsum(np.insert(sq_err, 0, 0.0))
    out = np.full(len(sq_err), np.nan)
    for i in range(len(sq_err)):
        lo = max(0, i + 1 - window)
        out[i] = np.sqrt((cum[i + 1] - cum[lo]) / (i + 1 - lo))
    return out


def main():
    train_df, _, _ = load_splits()
    val_df = pd.read_csv(config.PREDICTIONS_DIR / "experts_val.csv")
    test_df = pd.read_csv(config.PREDICTIONS_DIR / "experts_test.csv")

    with open(config.MODELS_DIR / "cf_svdpp.pkl", "rb") as f:
        cf_expert = pickle.load(f)
    with open(config.MODELS_DIR / "content_based.pkl", "rb") as f:
        cb_expert = pickle.load(f)
    item_popularity = build_item_popularity(train_df)

    # ---- Phase 1: deployed eval (fit on VAL stream, freeze, score on TEST) ----
    val_sorted = val_df.sort_values("timestamp").reset_index(drop=True)
    X_val = build_gate_features(val_sorted, item_popularity, cf_expert, cb_expert).to_numpy(dtype=float)
    cf_val = val_sorted["cf_pred"].to_numpy(dtype=float)
    cb_val = val_sorted["cb_pred"].to_numpy(dtype=float)
    y_val = val_sorted["rating"].to_numpy(dtype=float)

    print("Training Bandit Gate (LinUCB) online over VAL (timestamp order)...")
    import time
    start = time.perf_counter()
    bandit = ContextualBanditGate(n_features=len(FEATURE_COLUMNS), strategy="ucb", ucb_alpha=1.0, seed=config.RANDOM_SEED)
    run_stream(bandit, X_val, cf_val, cb_val, y_val, explore=True)
    train_time_sec = time.perf_counter() - start
    print(f"  train_time_sec={train_time_sec:.2f}  n_params={bandit.n_params()}  arm_pulls={bandit.n_pulls.tolist()}")

    X_test = build_gate_features(test_df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=float)
    cf_test = test_df["cf_pred"].to_numpy(dtype=float)
    cb_test = test_df["cb_pred"].to_numpy(dtype=float)
    y_test = test_df["rating"].to_numpy(dtype=float)

    test_preds, test_alphas, _ = run_stream(bandit, X_test, cf_test, cb_test, y_test, explore=False)
    test_df["hybrid_pred"] = test_preds
    test_df["gate_g"] = test_alphas
    test_metrics = segmented_rating_metrics(test_df, true_col="rating", pred_col="hybrid_pred")
    print_segment_table("Bandit Gate (frozen, LinUCB) [test]", test_metrics)
    print(f"  mean g by segment: {test_df.groupby('segment')['gate_g'].mean().to_dict()}")

    preds_path = config.PREDICTIONS_DIR / "model5_bandit_gate_test.csv"
    test_df.to_csv(preds_path, index=False)
    print(f"\nSaved test predictions -> {preds_path}")

    with open(config.MODELS_DIR / "model5_bandit_gate.pkl", "wb") as f:
        pickle.dump(bandit, f)

    results = {
        "test": test_metrics,
        "compute_cost": {"train_time_sec": train_time_sec, "n_params": bandit.n_params()},
        "arm_alphas": bandit.arms.tolist(),
        "arm_pulls_on_val": bandit.n_pulls.tolist(),
    }

    # ---- Phase 2: sequential simulation across VAL+TEST for the learning curve ----
    print("\nRunning sequential simulation (VAL+TEST, timestamp order, cold-start bandits)...")
    stream_df = pd.concat([val_df, test_df.drop(columns=["hybrid_pred", "gate_g"])], ignore_index=True)
    stream_df = stream_df.sort_values("timestamp").reset_index(drop=True)
    X_stream = build_gate_features(stream_df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=float)
    cf_stream = stream_df["cf_pred"].to_numpy(dtype=float)
    cb_stream = stream_df["cb_pred"].to_numpy(dtype=float)
    y_stream = stream_df["rating"].to_numpy(dtype=float)

    curves = {}
    for strategy, label in (("ucb", "Bandit (LinUCB)"), ("epsilon_greedy", "Bandit (epsilon-greedy)")):
        b = ContextualBanditGate(n_features=len(FEATURE_COLUMNS), strategy=strategy, ucb_alpha=1.0,
                                  epsilon=0.1, seed=config.RANDOM_SEED)
        _, _, sq_err = run_stream(b, X_stream, cf_stream, cb_stream, y_stream, explore=True)
        curves[label] = sq_err

    # Reference baselines over the SAME stream (no learning, deterministic).
    cf_only_err = (cf_stream - y_stream) ** 2
    cb_only_err = (cb_stream - y_stream) ** 2
    static_pred = np.clip(0.84 * cf_stream + 0.16 * cb_stream, 0.5, 5.0)
    static_err = (static_pred - y_stream) ** 2
    curves["CF/SVD++ only"] = cf_only_err
    curves["Content-Based only"] = cb_only_err
    curves["Static Hybrid (a=0.84)"] = static_err

    window = 500
    curve_df = pd.DataFrame({"step": np.arange(1, len(y_stream) + 1)})
    for label, sq_err in curves.items():
        curve_df[label] = rolling_rmse(sq_err, window=window)
    curve_path = config.METRICS_DIR / "model5_learning_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    print(f"Saved learning curve data -> {curve_path}")

    final_rmse = {label: float(np.sqrt(np.mean(sq_err))) for label, sq_err in curves.items()}
    print("\nFull-stream RMSE (sequential simulation, VAL+TEST):")
    for label, r in final_rmse.items():
        print(f"  {label:28s} {r:.4f}")
    results["sequential_simulation_full_stream_rmse"] = final_rmse
    results["sequential_simulation_window"] = window
    results["sequential_simulation_n_steps"] = int(len(y_stream))

    plt.figure(figsize=(9, 5.5))
    for label in ["Bandit (LinUCB)", "Bandit (epsilon-greedy)", "Static Hybrid (a=0.84)", "CF/SVD++ only", "Content-Based only"]:
        plt.plot(curve_df["step"], curve_df[label], label=label, linewidth=1.3)
    plt.xlabel(f"Interaction step (chronological, VAL+TEST, n={len(y_stream)})")
    plt.ylabel(f"Rolling RMSE (window={window})")
    plt.title("Model 5: Bandit Gate online learning curve")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path = config.METRICS_DIR / "model5_learning_curve.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved learning curve plot -> {plot_path}")

    metrics_path = config.METRICS_DIR / "model5_bandit_gate.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
