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
import os
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
from atg import baselines as atg_baselines
from atg.models.hybrid import blend_scores
from atg.eval.metrics import segmented_rating_metrics
from atg.eval.multiseed import aggregate_segmented_metrics
from atg.utils.segments import build_user_segments, attach_segments

# Seeds are configurable so a long run can be cut short (or extended) without
# editing code: ATG_SEEDS="42,1,2" runs three. The full pipeline is re-run per
# seed -- experts included -- so wall-clock scales linearly with this list.
SEEDS = [int(s) for s in os.environ.get("ATG_SEEDS", "42,1,2,3,4").split(",") if s.strip()]


def run_pipeline_for_seed(seed: int, items_df: pd.DataFrame) -> dict:
    train_df, val_df, test_df = build_splits(seed=seed)
    user_segments = build_user_segments(train_df)
    item_pop = build_item_popularity(train_df)

    cf = CFExpertSVDpp(random_state=seed).fit(train_df)
    cb = ContentBasedExpert().fit(train_df, items_df)

    def score(df):
        out = attach_segments(df.copy(), user_segments)
        out["cf_pred"] = cf.predict_batch(out)
        out["cb_pred"] = cb.predict_batch(out)
        return out

    val_s = score(val_df).reset_index(drop=True)
    test_s = score(test_df).reset_index(drop=True)
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

    # Gate features are the expensive part of a seed (one content-similarity
    # lookup per row), so build them ONCE per split in the split's own order and
    # permute for the bandit rather than rebuilding on the sorted frame.
    Xv = build_gate_features(val_s, item_pop, cf, cb).to_numpy(dtype=float)
    Xt = build_gate_features(test_s, item_pop, cf, cb).to_numpy(dtype=float)

    # Stable + tiebroken, matching scripts/05_bandit_gate.py -- see the note
    # there. Without this the bandit is the one model whose result does not
    # reproduce across runs, which would show up as inflated seed variance.
    order = val_s.sort_values(
        ["timestamp", "userId", "itemId"], kind="mergesort").index.to_numpy()
    val_sorted = val_s.loc[order].reset_index(drop=True)
    Xv_sorted = Xv[order]
    bandit = ContextualBanditGate(n_features=len(FEATURE_COLUMNS), strategy="ucb", seed=seed)
    run_stream(bandit, Xv_sorted, val_sorted["cf_pred"].to_numpy(dtype=float), val_sorted["cb_pred"].to_numpy(dtype=float),
               val_sorted["rating"].to_numpy(dtype=float), explore=True)
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

    # External baselines under the SAME seed, so the headline table reports
    # mean +/- std for the published alternatives too. Without this the
    # comparison is asymmetric: a gate with error bars against a baseline
    # point estimate, when the gaps between them are ~0.001-0.007.
    def arrs(df):
        return (df["cf_pred"].to_numpy(dtype=float), df["cb_pred"].to_numpy(dtype=float),
                df["rating"].to_numpy(dtype=float),
                df["train_rating_count"].to_numpy(dtype=float))

    cf_v, cb_v, y_v, cnt_v = arrs(val_s)
    cf_t, cb_t, _, cnt_t = arrs(test_s)
    fitted, _ = atg_baselines.fit_all(Xv, cf_v, cb_v, y_v, cnt_v, seed=seed)
    for name, pred in atg_baselines.predict_all(fitted, Xt, cf_t, cb_t, cnt_t).items():
        test_s[name] = pred
        record(name, name)

    return results


def main():
    items_df = pd.read_csv(config.NORMALIZED_ITEMS_CSV)

    path = config.METRICS_DIR / "multiseed_full_comparison.json"

    # Checkpoint after EVERY seed. This run is hours long and typically left
    # unattended overnight on a Colab runtime that disconnects on idle; without
    # an incremental write, a drop on the last seed discards every completed
    # one. Re-running resumes from whatever seeds are already on disk.
    per_seed, done_seeds = {}, []
    if path.exists():
        with open(path) as f:
            prev = json.load(f)
        per_seed = prev.get("per_seed", {})
        done_seeds = list(prev.get("seeds", []))
        if done_seeds:
            print(f"Resuming: seeds already on disk = {done_seeds}")

    todo = [s for s in SEEDS if s not in done_seeds]
    print(f"Running full 7-model pipeline across {len(todo)} seeds: {todo}")

    for seed in todo:
        start = time.perf_counter()
        results = run_pipeline_for_seed(seed, items_df)
        elapsed = time.perf_counter() - start
        for name, m in results.items():
            per_seed.setdefault(name, []).append(m)
        done_seeds.append(seed)
        print(f"  seed={seed} done in {elapsed:.1f}s  "
              f"overall_rmse: " + ", ".join(f"{n.split('_',1)[1]}={m['overall']['rmse']:.3f}" for n, m in results.items()))

        partial = {"seeds": done_seeds, "per_seed": per_seed,
                   "aggregate": {n: aggregate_segmented_metrics(ms) for n, ms in per_seed.items()}}
        with open(path, "w") as f:
            json.dump(partial, f, indent=2)
        print(f"    checkpointed {len(done_seeds)} seed(s) -> {path}")

    aggregate = {name: aggregate_segmented_metrics(metrics_list) for name, metrics_list in per_seed.items()}

    print(f"\n[Multi-seed summary, test set, mean +/- std over {len(done_seeds)} seeds: {done_seeds}]")
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

    out = {"seeds": done_seeds, "per_seed": per_seed, "aggregate": aggregate}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
