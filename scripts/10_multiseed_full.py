"""Multi-seed variance check across ALL 7 models: re-run the entire
pipeline (split -> experts -> every gate) under several seeds and report
mean +/- std segmented RMSE/MAE per model, instead of the single seed=42
point estimates the canonical results/ directory holds. A single seed can't
distinguish a real effect from noise in the stochastic simulated cold-start
split (atg.data.splits) and in gate training/GA search/bandit exploration.

This does NOT overwrite the canonical seed=42 results in results/ -- it's a
separate robustness check, saved to results/metrics/multiseed_full_comparison.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atg import config
from atg.data.splits import build_splits
from atg.experts.cf_svdpp import CFExpertSVDpp
from atg.experts.content_based import ContentBasedExpert
from atg.gates.static import StaticGate
from atg.gates.learned import LearnedGate
from atg.gates.bandit import ContextualBanditGate, run_stream
from atg.gates.ga import GAEvolvedGate
from atg.gates.sequential import SequentialGate, build_user_train_sequences
from atg.gates.features import build_item_popularity, build_gate_features, FEATURE_COLUMNS
from atg.models.hybrid import blend_scores
from atg.eval.metrics import segmented_rating_metrics
from atg.eval.multiseed import aggregate_segmented_metrics
from atg.utils.segments import build_user_segments, attach_segments

SEEDS = [42, 1, 2, 3, 4]


def run_pipeline_for_seed(seed: int, movies_df: pd.DataFrame, tags_df: pd.DataFrame) -> dict:
    train_df, val_df, test_df = build_splits(seed=seed)
    user_segments = build_user_segments(train_df)
    item_pop = build_item_popularity(train_df)

    cf = CFExpertSVDpp(random_state=seed).fit(train_df)
    cb = ContentBasedExpert().fit(train_df, movies_df, tags_df)

    def score(df):
        out = attach_segments(df.copy(), user_segments)
        out["cf_pred"] = cf.predict_batch(out)
        out["cb_pred"] = cb.predict_batch(out)
        return out

    val_s, test_s = score(val_df), score(test_df)
    results = {}

    def record(name, pred_col):
        results[name] = segmented_rating_metrics(test_s, true_col="rating", pred_col=pred_col)

    test_s["m1"] = test_s["cf_pred"]
    record("1_CF_SVDpp", "m1")
    test_s["m2"] = test_s["cb_pred"]
    record("2_ContentBased", "m2")

    static = StaticGate().fit(val_s, step=0.02)
    test_s["m3"] = blend_scores(test_s["cf_pred"], test_s["cb_pred"], static.g(len(test_s)))
    record("3_StaticHybrid", "m3")

    learned = LearnedGate(hidden_size=0, l2=1.0, es_frac=0.2, seed=seed).fit(val_s, item_pop, cf, cb)
    test_s["m4"] = blend_scores(test_s["cf_pred"], test_s["cb_pred"], learned.g(test_s, item_pop, cf, cb))
    record("4_LearnedGate", "m4")

    val_sorted = val_s.sort_values("timestamp").reset_index(drop=True)
    Xv = build_gate_features(val_sorted, item_pop, cf, cb).to_numpy(dtype=float)
    bandit = ContextualBanditGate(n_features=len(FEATURE_COLUMNS), strategy="ucb", seed=seed)
    run_stream(bandit, Xv, val_sorted["cf_pred"].to_numpy(dtype=float), val_sorted["cb_pred"].to_numpy(dtype=float),
               val_sorted["rating"].to_numpy(dtype=float), explore=True)
    Xt = build_gate_features(test_s, item_pop, cf, cb).to_numpy(dtype=float)
    preds5, _, _ = run_stream(bandit, Xt, test_s["cf_pred"].to_numpy(dtype=float), test_s["cb_pred"].to_numpy(dtype=float),
                               test_s["rating"].to_numpy(dtype=float), explore=False)
    test_s["m5"] = preds5
    record("5_BanditGate", "m5")

    ga = GAEvolvedGate(population_size=16, generations=12, seed=seed).fit(val_s, item_pop, cf, cb)
    test_s["m6"] = blend_scores(test_s["cf_pred"], test_s["cb_pred"], ga.g(test_s, item_pop, cf, cb))
    record("6_GAEvolvedGate", "m6")

    seqs = build_user_train_sequences(train_df)
    seq_gate = SequentialGate(fallback_alpha=static.alpha, seed=seed).fit(val_s, seqs, item_pop, cf, cb, epochs=60, patience=10)
    g7, _ = seq_gate.g(test_s, seqs, item_pop, cf, cb)
    test_s["m7"] = blend_scores(test_s["cf_pred"], test_s["cb_pred"], g7)
    record("7_SequentialGate", "m7")

    return results


def main():
    movies_df = pd.read_csv(config.MOVIES_CSV)
    tags_df = pd.read_csv(config.TAGS_CSV)

    print(f"Running full 7-model pipeline across {len(SEEDS)} seeds: {SEEDS}")
    per_seed = {}
    for seed in SEEDS:
        start = time.perf_counter()
        results = run_pipeline_for_seed(seed, movies_df, tags_df)
        elapsed = time.perf_counter() - start
        for name, m in results.items():
            per_seed.setdefault(name, []).append(m)
        print(f"  seed={seed} done in {elapsed:.1f}s  "
              f"overall_rmse: " + ", ".join(f"{n.split('_',1)[1]}={m['overall']['rmse']:.3f}" for n, m in results.items()))

    aggregate = {name: aggregate_segmented_metrics(metrics_list) for name, metrics_list in per_seed.items()}

    print("\n[Multi-seed summary, test set, mean +/- std over 5 seeds]")
    header = f"  {'model':20s} {'overall':>16s} {'cold':>16s} {'warm':>16s} {'power':>16s}"
    print(header)
    for name, agg in aggregate.items():
        cells = []
        for seg in ("overall", "cold", "warm", "power"):
            if seg in agg:
                a = agg[seg]
                cells.append(f"{a['rmse_mean']:.3f}+/-{a['rmse_std']:.3f}")
            else:
                cells.append("n/a")
        print(f"  {name:20s} " + " ".join(f"{c:>16s}" for c in cells))

    out = {"seeds": SEEDS, "per_seed": per_seed, "aggregate": aggregate}
    path = config.METRICS_DIR / "multiseed_full_comparison.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
