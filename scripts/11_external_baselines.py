"""External baselines on the cached expert predictions.

Thin driver around atg.baselines -- see that module for what B1-B4 are and why
they are here. This script fits them on VAL, scores them on TEST, and writes
predictions + metrics in the same layout every model script uses, so
scripts/08_full_comparison.py can fold them into the main table.

Protocol matches the gates exactly: fit on VAL (the experts were fit on TRAIN,
so only their VAL predictions are honest out-of-sample scores), using the SAME
gate-train/early-stop partition Models 4 and 6 use, and score on the untouched
TEST split.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atg import config
from atg import baselines
from atg.data.splits import load_splits
from atg.eval.metrics import segmented_rating_metrics
from atg.gates.features import build_item_popularity, build_gate_features

STUBS = {
    "B1_SwitchingHybrid": "b1_switching_hybrid",
    "B2_FWLS": "b2_fwls",
    "B3_RidgeStack": "b3_ridge_stack",
    "B4_GBMStack": "b4_gbm_stack",
}


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

    def cols(df):
        return (df["cf_pred"].to_numpy(dtype=float),
                df["cb_pred"].to_numpy(dtype=float),
                df["rating"].to_numpy(dtype=float),
                df["train_rating_count"].to_numpy(dtype=float))

    cf_val, cb_val, y_val, counts_val = cols(val_df)
    cf_test, cb_test, _, counts_test = cols(test_df)

    fitted, (fit_idx, hold_idx) = baselines.fit_all(
        X_val, cf_val, cb_val, y_val, counts_val, seed=config.RANDOM_SEED)
    print(f"  val fit rows={len(fit_idx)}  holdout rows={len(hold_idx)}"
          f"  (same partition as Models 4 and 6)")

    b1 = fitted["B1_SwitchingHybrid"]
    b3 = fitted["B3_RidgeStack"]["model"]
    print(f"  B1 best tau={b1['tau']} train ratings")
    print(f"  B2 n_params={fitted['B2_FWLS']['n_params']}")
    print(f"  B3 w_cf={b3.coef_[0]:.4f}  w_cb={b3.coef_[1]:.4f}  intercept={b3.intercept_:.4f}"
          f"  (weights sum to {b3.coef_.sum():.4f})")
    print(f"  B4 holdout RMSE={fitted['B4_GBMStack']['holdout_rmse']:.4f}")

    val_preds = baselines.predict_all(fitted, X_val, cf_val, cb_val, counts_val)
    test_preds = baselines.predict_all(fitted, X_test, cf_test, cb_test, counts_test)

    results, summary = {}, []
    for name in baselines.BASELINE_NAMES:
        v = val_df.copy()
        v["hybrid_pred"] = val_preds[name]
        val_metrics = segmented_rating_metrics(v, true_col="rating", pred_col="hybrid_pred")
        print_segment_table(f"{name} [val]", val_metrics)

        t = test_df.copy()
        t["hybrid_pred"] = test_preds[name]
        test_metrics = segmented_rating_metrics(t, true_col="rating", pred_col="hybrid_pred")
        print_segment_table(f"{name} [test]", test_metrics)

        preds_path = config.PREDICTIONS_DIR / f"{STUBS[name]}_test.csv"
        t.to_csv(preds_path, index=False)
        print(f"  saved -> {preds_path}")

        m = fitted[name]
        extra = {k: v2 for k, v2 in m.items() if k in ("tau", "holdout_rmse")}
        if name == "B3_RidgeStack":
            extra.update({"w_cf": float(b3.coef_[0]), "w_cb": float(b3.coef_[1]),
                          "intercept": float(b3.intercept_)})
        results[name] = {
            **extra,
            "val": val_metrics,
            "test": test_metrics,
            "compute_cost": {"train_time_sec": m["train_time_sec"], "n_params": m["n_params"]},
        }
        summary.append((name, test_metrics["overall"]["rmse"]))

    metrics_path = config.METRICS_DIR / "external_baselines.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")

    print("\nExternal baselines, overall TEST RMSE:")
    for name, r in sorted(summary, key=lambda kv: kv[1]):
        print(f"  {name:24s} {r:.4f}")
    print("\nRe-run scripts/08_full_comparison.py to fold these into the main table.")


if __name__ == "__main__":
    main()
