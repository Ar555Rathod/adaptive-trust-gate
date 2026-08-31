"""Train the two shared experts (CF/SVD++ and Content-Based) once on the
cached train split, evaluate both on val and test with RMSE/MAE, overall and
segmented by cold/warm/power. This is the Week 1 milestone: baselines +
split, before any gating mechanism is introduced.
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atg import config
from atg.data.splits import load_splits
from atg.experts.cf_svdpp import CFExpertSVDpp
from atg.experts.content_based import ContentBasedExpert
from atg.eval.metrics import segmented_rating_metrics
from atg.utils.segments import build_user_segments, attach_segments


def evaluate(name, expert, df, user_segments, pred_col):
    preds = expert.predict_batch(df)
    scored = df.copy()
    scored[pred_col] = preds
    scored = attach_segments(scored, user_segments)
    metrics = segmented_rating_metrics(scored, true_col="rating", pred_col=pred_col)
    print(f"\n[{name}]")
    for seg in ("overall", "cold", "warm", "power"):
        if seg in metrics:
            m = metrics[seg]
            print(f"  {seg:8s} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  n={m['n']}")
    return scored, metrics


def main():
    train_df, val_df, test_df = load_splits()
    items_df = pd.read_csv(config.NORMALIZED_ITEMS_CSV)
    user_segments = build_user_segments(train_df)

    print("Training CF expert (SVD++)...")
    cf = CFExpertSVDpp()
    cf.fit(train_df)
    print(f"  train_time_sec={cf.train_time_sec:.2f}  n_params={cf.n_params()}")

    print("\nTraining Content-Based expert...")
    cb = ContentBasedExpert()
    cb.fit(train_df, items_df)
    print(f"  train_time_sec={cb.train_time_sec:.2f}  n_params(repr size)={cb.n_params()}")

    with open(config.MODELS_DIR / "cf_svdpp.pkl", "wb") as f:
        pickle.dump(cf, f)
    with open(config.MODELS_DIR / "content_based.pkl", "wb") as f:
        pickle.dump(cb, f)
    print(f"Saved fitted experts -> {config.MODELS_DIR}")

    results = {}
    for split_name, df in (("val", val_df), ("test", test_df)):
        cf_scored, cf_metrics = evaluate(f"CF/SVD++ [{split_name}]", cf, df, user_segments, "cf_pred")

        start = time.perf_counter()
        cb_scored, cb_metrics = evaluate(f"Content-Based [{split_name}]", cb, df, user_segments, "cb_pred")
        cb_infer_time = time.perf_counter() - start

        results[split_name] = {
            "cf_svdpp": cf_metrics,
            "content_based": cb_metrics,
        }

        preds_path = config.PREDICTIONS_DIR / f"experts_{split_name}.csv"
        merged = df.copy()
        merged["cf_pred"] = cf_scored["cf_pred"]
        merged["cb_pred"] = cb_scored["cb_pred"]
        merged = attach_segments(merged, user_segments)
        merged.to_csv(preds_path, index=False)
        print(f"\nSaved {split_name} predictions -> {preds_path}")

    results["compute_cost"] = {
        "cf_svdpp": {"train_time_sec": cf.train_time_sec, "n_params": cf.n_params()},
        "content_based": {"train_time_sec": cb.train_time_sec, "n_params_repr_size": cb.n_params()},
    }

    metrics_path = config.METRICS_DIR / "experts_baseline.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
