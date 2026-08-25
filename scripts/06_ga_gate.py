"""Model 6 / GA-Evolved Gate: a genetic algorithm searches the gate's
feature subset + architecture (hidden_size) on VAL (same split discipline
as Model 4 -- see atg.gates.learned for the leakage rationale), then the
winning genome is retrained to convergence and scored on TEST.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atg import config
from atg.data.splits import load_splits
from atg.gates.features import build_item_popularity
from atg.gates.ga import GAEvolvedGate
from atg.models.hybrid import blend_scores
from atg.eval.metrics import segmented_rating_metrics


def print_segment_table(name, metrics):
    print(f"\n[{name}]")
    for seg in ("overall", "cold", "warm", "power"):
        if seg in metrics:
            m = metrics[seg]
            print(f"  {seg:8s} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  n={m['n']}")


def main():
    train_df, _, _ = load_splits()
    val_df = pd.read_csv(config.PREDICTIONS_DIR / "experts_val.csv")
    test_df = pd.read_csv(config.PREDICTIONS_DIR / "experts_test.csv")

    with open(config.MODELS_DIR / "cf_svdpp.pkl", "rb") as f:
        cf_expert = pickle.load(f)
    with open(config.MODELS_DIR / "content_based.pkl", "rb") as f:
        cb_expert = pickle.load(f)
    item_popularity = build_item_popularity(train_df)

    print("Running GA search (feature subset + hidden_size) on VAL...")
    gate = GAEvolvedGate(population_size=16, generations=12, seed=config.RANDOM_SEED)
    gate.fit(val_df, item_popularity, cf_expert, cb_expert)
    print(f"  train_time_sec={gate.train_time_sec:.2f}  n_params={gate.n_params()}")
    print(f"  selected features: {gate.selected_features_}")
    print(f"  hidden_size: {gate.net.hidden_size}")
    print("  GA fitness by generation (best / mean early-stop MSE):")
    for row in gate.ga_history_:
        print(f"    gen {row['generation']:2d}  best={row['best_es_mse']:.4f}  mean={row['mean_es_mse']:.4f}")

    val_g = gate.g(val_df, item_popularity, cf_expert, cb_expert)
    val_df["gate_g"] = val_g
    val_df["hybrid_pred"] = blend_scores(val_df["cf_pred"], val_df["cb_pred"], val_g)
    val_metrics = segmented_rating_metrics(val_df, true_col="rating", pred_col="hybrid_pred")
    print_segment_table("GA-Evolved Gate [val]", val_metrics)

    test_g = gate.g(test_df, item_popularity, cf_expert, cb_expert)
    test_df["gate_g"] = test_g
    test_df["hybrid_pred"] = blend_scores(test_df["cf_pred"], test_df["cb_pred"], test_g)
    test_metrics = segmented_rating_metrics(test_df, true_col="rating", pred_col="hybrid_pred")
    print_segment_table("GA-Evolved Gate [test]", test_metrics)
    print(f"  mean g by segment: {test_df.groupby('segment')['gate_g'].mean().to_dict()}")

    preds_path = config.PREDICTIONS_DIR / "model6_ga_gate_test.csv"
    test_df.to_csv(preds_path, index=False)
    print(f"\nSaved test predictions -> {preds_path}")

    with open(config.MODELS_DIR / "model6_ga_gate.pkl", "wb") as f:
        pickle.dump(gate, f)

    ga_curve_path = config.METRICS_DIR / "model6_ga_fitness_curve.csv"
    pd.DataFrame(gate.ga_history_).to_csv(ga_curve_path, index=False)
    print(f"Saved GA fitness-over-generations curve -> {ga_curve_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = pd.DataFrame(gate.ga_history_)
        plt.figure(figsize=(7, 4.5))
        plt.plot(hist["generation"], hist["best_es_mse"], marker="o", label="best (elite) held-out MSE")
        plt.plot(hist["generation"], hist["mean_es_mse"], marker="o", label="population mean held-out MSE")
        plt.xlabel("GA generation")
        plt.ylabel("Held-out hybrid MSE (lower is better)")
        plt.title("Model 6: GA feature/architecture search convergence")
        plt.legend()
        plt.tight_layout()
        plot_path = config.METRICS_DIR / "model6_ga_fitness_curve.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved GA fitness plot -> {plot_path}")
    except ImportError:
        pass

    results = {
        "val": val_metrics,
        "test": test_metrics,
        "compute_cost": {"train_time_sec": gate.train_time_sec, "n_params": gate.n_params()},
        "selected_features": gate.selected_features_,
        "hidden_size": gate.net.hidden_size,
        "ga_history": gate.ga_history_,
    }
    metrics_path = config.METRICS_DIR / "model6_ga_gate.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
