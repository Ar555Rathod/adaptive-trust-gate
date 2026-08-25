"""Model 7 / Sequential Gate (BiLSTM): trained on VAL (same split discipline
as Models 4/6), evaluated on TEST with the standard segmented table, PLUS a
breakdown by whether each row used the actual BiLSTM or fell back to Model
3's fixed alpha (fewer than SEQ_MIN_HISTORY prior TRAIN ratings) -- this
isolates genuine sequential adaptation from the fallback behaviour, per the
proposal's Model 7 evaluation note.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from atg import config
from atg.data.splits import load_splits
from atg.gates.features import build_item_popularity
from atg.gates.sequential import SequentialGate, build_user_train_sequences
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

    with open(config.METRICS_DIR / "model3_static_hybrid.json") as f:
        model3 = json.load(f)
    fallback_alpha = model3["alpha"]
    print(f"Fallback alpha (Model 3): {fallback_alpha:.2f}")
    print(f"SEQ_MIN_HISTORY={config.SEQ_MIN_HISTORY}  SEQ_LONG_LEN={config.SEQ_LONG_LEN}  SEQ_SHORT_LEN={config.SEQ_SHORT_LEN}")

    seqs = build_user_train_sequences(train_df)

    print("\nTraining Sequential Gate (BiLSTM) on VAL...")
    gate = SequentialGate(fallback_alpha=fallback_alpha, seed=config.RANDOM_SEED)
    gate.fit(val_df, seqs, item_popularity, cf_expert, cb_expert, epochs=60, patience=10)
    print(f"  train_time_sec={gate.train_time_sec:.2f}  n_params={gate.n_params()}")
    last = gate.history_[-1]
    print(f"  stopped at epoch={last[0]}  train_mse={last[1]:.4f}  es_mse={last[2]:.4f}")

    val_g, val_fallback = gate.g(val_df, seqs, item_popularity, cf_expert, cb_expert)
    val_df["gate_g"] = val_g
    val_df["used_fallback"] = val_fallback
    val_df["hybrid_pred"] = blend_scores(val_df["cf_pred"], val_df["cb_pred"], val_g)
    val_metrics = segmented_rating_metrics(val_df, true_col="rating", pred_col="hybrid_pred")
    print_segment_table("Sequential Gate [val]", val_metrics)

    test_g, test_fallback = gate.g(test_df, seqs, item_popularity, cf_expert, cb_expert)
    test_df["gate_g"] = test_g
    test_df["used_fallback"] = test_fallback
    test_df["hybrid_pred"] = blend_scores(test_df["cf_pred"], test_df["cb_pred"], test_g)
    test_metrics = segmented_rating_metrics(test_df, true_col="rating", pred_col="hybrid_pred")
    print_segment_table("Sequential Gate [test]", test_metrics)
    print(f"  mean g by segment: {test_df.groupby('segment')['gate_g'].mean().to_dict()}")
    print(f"  fallback rate: {test_fallback.mean():.1%}  ({int(test_fallback.sum())}/{len(test_fallback)} rows)")

    fallback_group = test_df.groupby("used_fallback")
    fallback_metrics = {}
    for used_fallback, sub in fallback_group:
        key = "fallback_to_static" if used_fallback else "genuine_bilstm"
        m = segmented_rating_metrics(sub, true_col="rating", pred_col="hybrid_pred")
        fallback_metrics[key] = m
        print_segment_table(f"Sequential Gate [test, {key}, n={len(sub)}]", m)

    preds_path = config.PREDICTIONS_DIR / "model7_sequential_gate_test.csv"
    test_df.to_csv(preds_path, index=False)
    print(f"\nSaved test predictions -> {preds_path}")

    with open(config.MODELS_DIR / "model7_sequential_gate.pkl", "wb") as f:
        pickle.dump(gate, f)

    results = {
        "val": val_metrics,
        "test": test_metrics,
        "test_by_fallback": fallback_metrics,
        "test_fallback_rate": float(test_fallback.mean()),
        "compute_cost": {"train_time_sec": gate.train_time_sec, "n_params": gate.n_params()},
        "training_curve_tail": gate.history_[-5:],
        "seq_min_history": config.SEQ_MIN_HISTORY,
        "fallback_alpha": fallback_alpha,
    }
    metrics_path = config.METRICS_DIR / "model7_sequential_gate.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
