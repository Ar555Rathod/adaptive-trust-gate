"""Multi-seed variance check for Model 3 (Static Hybrid): re-run the whole
split -> experts -> grid-search pipeline under several seeds and report
mean +/- std RMSE/MAE per segment, instead of a single point estimate.
Demonstrates the methodology from atg.eval.multiseed; the same pattern
applies to every other model once wired up.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atg import config
from atg.data.splits import build_splits
from atg.experts.cf_svdpp import CFExpertSVDpp
from atg.experts.content_based import ContentBasedExpert
from atg.gates.static import StaticGate
from atg.models.hybrid import blend_scores
from atg.eval.metrics import segmented_rating_metrics
from atg.eval.multiseed import aggregate_segmented_metrics
from atg.utils.segments import build_user_segments, attach_segments

SEEDS = [42, 1, 2, 3, 4]


def run_one_seed(seed: int, items_df: pd.DataFrame) -> dict:
    train_df, val_df, test_df = build_splits(seed=seed)
    user_segments = build_user_segments(train_df)

    cf = CFExpertSVDpp(random_state=seed).fit(train_df)
    cb = ContentBasedExpert().fit(train_df, items_df)

    val_scored = attach_segments(val_df.copy(), user_segments)
    val_scored["cf_pred"] = cf.predict_batch(val_scored)
    val_scored["cb_pred"] = cb.predict_batch(val_scored)

    gate = StaticGate().fit(val_scored, step=0.02)

    test_scored = attach_segments(test_df.copy(), user_segments)
    test_scored["cf_pred"] = cf.predict_batch(test_scored)
    test_scored["cb_pred"] = cb.predict_batch(test_scored)
    test_scored["hybrid_pred"] = blend_scores(test_scored["cf_pred"], test_scored["cb_pred"], gate.g(len(test_scored)))

    metrics = segmented_rating_metrics(test_scored, true_col="rating", pred_col="hybrid_pred")
    metrics["_alpha"] = gate.alpha
    print(f"  seed={seed}  alpha={gate.alpha:.2f}  "
          f"overall_rmse={metrics['overall']['rmse']:.4f}  "
          f"cold_rmse={metrics.get('cold', {}).get('rmse', float('nan')):.4f}")
    return metrics


def main():
    items_df = pd.read_csv(config.NORMALIZED_ITEMS_CSV)

    print(f"Running Model 3 across {len(SEEDS)} seeds: {SEEDS}")
    per_seed = [run_one_seed(s, items_df) for s in SEEDS]
    alphas = [m.pop("_alpha") for m in per_seed]

    agg = aggregate_segmented_metrics(per_seed)
    print("\n[Model 3 multi-seed summary, test set]")
    for seg in ("overall", "cold", "warm", "power"):
        if seg in agg:
            a = agg[seg]
            print(f"  {seg:8s} RMSE={a['rmse_mean']:.4f}+/-{a['rmse_std']:.4f}  "
                  f"MAE={a['mae_mean']:.4f}+/-{a['mae_std']:.4f}  (n_seeds={a['n_seeds']})")
    print(f"\n  alpha across seeds: {[f'{a:.2f}' for a in alphas]}")

    out = {"seeds": SEEDS, "alphas": alphas, "per_seed": per_seed, "aggregate": agg}
    path = config.METRICS_DIR / "model3_multiseed.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
