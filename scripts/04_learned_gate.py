"""Model 4 / Learned Gate: a small gate network (logistic-regression
capacity by default) trained on sparsity + similarity + CF-confidence
features, fit on VAL (the experts' honest out-of-sample predictions) and
evaluated on TEST -- see atg.gates.learned for the split-discipline
rationale.
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
from atg.gates.learned import LearnedGate
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

    print("Fitting Learned Gate (logistic-regression capacity) on VAL...")
    gate = LearnedGate(hidden_size=0, l2=1e-3, lr=0.05, es_frac=0.2)
    gate.fit(val_df, item_popularity, cf_expert, cb_expert, epochs=3000, patience=200)
    print(f"  train_time_sec={gate.train_time_sec:.2f}  n_params={gate.n_params()}")
    last_epoch, last_train_loss, last_es_loss = gate.net.history_[-1]
    print(f"  stopped at epoch={last_epoch}  train_mse={last_train_loss:.4f}  es_mse={last_es_loss:.4f}")

    val_g = gate.g(val_df, item_popularity, cf_expert, cb_expert)
    val_df["gate_g"] = val_g
    val_df["hybrid_pred"] = blend_scores(val_df["cf_pred"], val_df["cb_pred"], val_g)
    val_metrics = segmented_rating_metrics(val_df, true_col="rating", pred_col="hybrid_pred")
    print_segment_table("Learned Gate [val]", val_metrics)
    print(f"  mean g by segment: {val_df.groupby('segment')['gate_g'].mean().to_dict()}")

    test_g = gate.g(test_df, item_popularity, cf_expert, cb_expert)
    test_df["gate_g"] = test_g
    test_df["hybrid_pred"] = blend_scores(test_df["cf_pred"], test_df["cb_pred"], test_g)
    test_metrics = segmented_rating_metrics(test_df, true_col="rating", pred_col="hybrid_pred")
    print_segment_table("Learned Gate [test]", test_metrics)
    print(f"  mean g by segment: {test_df.groupby('segment')['gate_g'].mean().to_dict()}")

    preds_path = config.PREDICTIONS_DIR / "model4_learned_gate_test.csv"
    test_df.to_csv(preds_path, index=False)
    print(f"\nSaved test predictions -> {preds_path}")

    with open(config.MODELS_DIR / "model4_learned_gate.pkl", "wb") as f:
        pickle.dump(gate, f)

    results = {
        "val": val_metrics,
        "test": test_metrics,
        "compute_cost": {"train_time_sec": gate.train_time_sec, "n_params": gate.n_params()},
        "training_curve_tail": gate.net.history_[-5:],
    }
    metrics_path = config.METRICS_DIR / "model4_learned_gate.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
