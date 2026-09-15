"""Model 8 / Calibrated Trust Gate: fit on VAL, score on TEST.

    r_hat = a + s * [ g(x)*cf + (1 - g(x))*cb ]

See atg.gates.calibrated for the rationale. Same split discipline as Model 4:
fit on VAL (the experts' honest out-of-sample predictions), evaluate on TEST.
Must run after scripts/02_train_experts.py and before 08_full_comparison.py.

Also fits the variant without count features, so a single-seed run shows both;
the multi-seed run is what decides between them.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atg import config
from atg.data.splits import load_splits
from atg.eval.metrics import segmented_rating_metrics
from atg.gates.calibrated import CalibratedGate
from atg.gates.features import build_item_popularity, build_gate_features


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

    print("Building gate context features (val/test)...")
    X_val = build_gate_features(val_df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=float)
    X_test = build_gate_features(test_df, item_popularity, cf_expert, cb_expert).to_numpy(dtype=float)

    results = {}
    for label, use_counts, stub in (("Calibrated Gate", True, "model8_calibrated_gate"),
                                    ("Calibrated Gate, no count features", False, "model8a_calibrated_no_counts")):
        print(f"\nFitting {label} on VAL...")
        gate = CalibratedGate(use_count_features=use_counts).fit_arrays(
            X_val, val_df["cf_pred"], val_df["cb_pred"], val_df["rating"])
        print(f"  l2={gate.l2}  shift a={gate.a:+.4f}  scale s={gate.s:.4f}  "
              f"holdout RMSE={gate.holdout_rmse_:.4f}  n_params={gate.n_params()}  "
              f"train_time_sec={gate.train_time_sec:.2f}")

        v = val_df.copy()
        v["hybrid_pred"], v["gate_g"] = gate.predict_arrays(X_val, v["cf_pred"], v["cb_pred"])
        val_metrics = segmented_rating_metrics(v, true_col="rating", pred_col="hybrid_pred")
        print_segment_table(f"{label} [val]", val_metrics)

        t = test_df.copy()
        t["hybrid_pred"], t["gate_g"] = gate.predict_arrays(X_test, t["cf_pred"], t["cb_pred"])
        test_metrics = segmented_rating_metrics(t, true_col="rating", pred_col="hybrid_pred")
        print_segment_table(f"{label} [test]", test_metrics)
        g_by_seg = t.groupby("segment")["gate_g"].mean().to_dict()
        print(f"  mean g by segment: {g_by_seg}")

        preds_path = config.PREDICTIONS_DIR / f"{stub}_test.csv"
        t.to_csv(preds_path, index=False)
        with open(config.MODELS_DIR / f"{stub}.pkl", "wb") as f:
            pickle.dump(gate, f)
        print(f"  saved -> {preds_path}")

        results[stub] = {
            "use_count_features": use_counts,
            "l2": gate.l2, "shift_a": gate.a, "scale_s": gate.s,
            "holdout_rmse": gate.holdout_rmse_,
            "mean_g_by_segment_test": g_by_seg,
            "val": val_metrics,
            "test": test_metrics,
            "compute_cost": {"train_time_sec": gate.train_time_sec, "n_params": gate.n_params()},
        }

    metrics_path = config.METRICS_DIR / "model8_calibrated_gate.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")
    print("Re-run scripts/08_full_comparison.py to fold Model 8 into the main table.")


if __name__ == "__main__":
    main()
