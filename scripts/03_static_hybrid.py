"""Model 3 / Static Hybrid: grid-search a single fixed alpha on val (reusing
the CF/CB predictions cached by 02_train_experts.py -- no retraining of the
experts needed), then evaluate the resulting blend on test, overall and by
sparsity segment.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atg import config
from atg.gates.static import StaticGate
from atg.models.hybrid import blend_scores
from atg.eval.metrics import segmented_rating_metrics


def print_segment_table(name, metrics):
    print(f"\n[{name}]")
    for seg in ("overall", "cold", "warm", "power"):
        if seg in metrics:
            m = metrics[seg]
            print(f"  {seg:8s} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  n={m['n']}")


def main():
    val_df = pd.read_csv(config.PREDICTIONS_DIR / "experts_val.csv")
    test_df = pd.read_csv(config.PREDICTIONS_DIR / "experts_test.csv")

    gate = StaticGate()
    gate.fit(val_df, step=0.01, metric="rmse")
    print(f"Grid search complete: best alpha={gate.alpha:.2f}  "
          f"(train_time_sec={gate.train_time_sec:.3f}, n_params={gate.n_params()})")

    grid_path = config.METRICS_DIR / "model3_alpha_grid.csv"
    pd.DataFrame(gate.grid_curve_, columns=["alpha", "rmse", "mae"]).to_csv(grid_path, index=False)
    print(f"Saved alpha grid curve -> {grid_path}")

    val_pred = blend_scores(val_df["cf_pred"], val_df["cb_pred"], gate.g(len(val_df)))
    val_df["hybrid_pred"] = val_pred
    val_metrics = segmented_rating_metrics(val_df, true_col="rating", pred_col="hybrid_pred")
    print_segment_table("Static Hybrid [val]", val_metrics)

    test_pred = blend_scores(test_df["cf_pred"], test_df["cb_pred"], gate.g(len(test_df)))
    test_df["hybrid_pred"] = test_pred
    test_metrics = segmented_rating_metrics(test_df, true_col="rating", pred_col="hybrid_pred")
    print_segment_table("Static Hybrid [test]", test_metrics)

    preds_path = config.PREDICTIONS_DIR / "model3_static_hybrid_test.csv"
    test_df.to_csv(preds_path, index=False)
    print(f"\nSaved test predictions -> {preds_path}")

    results = {
        "alpha": gate.alpha,
        "val": val_metrics,
        "test": test_metrics,
        "compute_cost": {"train_time_sec": gate.train_time_sec, "n_params": gate.n_params()},
    }
    metrics_path = config.METRICS_DIR / "model3_static_hybrid.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
