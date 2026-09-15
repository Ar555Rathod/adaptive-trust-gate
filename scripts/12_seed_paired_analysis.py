"""Paired, within-seed comparison of models from the multi-seed run.

Why this exists: 10_multiseed_full.py reports each model's RMSE as mean +/- std
across seeds. Those error bars are the WRONG thing to compare models with.
Changing the seed changes the train/val/test split, and a harder split raises
every model's RMSE together -- so most of each model's across-seed std is
shared split difficulty, not model noise. Two models can have fully
overlapping bars while one beats the other on every single seed.

The right comparison is paired: for each seed, take model A's RMSE minus model
B's RMSE on that SAME split, then look at the mean and spread of those
differences. The shared split effect cancels in the subtraction.

For each pair this reports the per-seed differences, their mean and std, a
paired t-test across seeds, and -- most usefully at small seed counts --
whether the sign agrees on every seed. With 3 seeds a t-test has 2 degrees of
freedom and little power, so "A beat B on 3/3 seeds" is the more honest
headline; the p-value is reported for completeness, not as the verdict.

Usage:  python scripts/12_seed_paired_analysis.py [path/to/multiseed_full_comparison.json]
        (defaults to the current dataset's results/<dataset>/metrics/ file)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy import stats

from atg import config

REFERENCE = "3_StaticHybrid"

# The comparisons the paper's argument actually rests on.
KEY_PAIRS = [
    ("4_LearnedGate", "3_StaticHybrid", "context under the convex constraint"),
    ("B3_RidgeStack", "3_StaticHybrid", "relaxing the constraint, no context"),
    ("B2_FWLS", "3_StaticHybrid", "context + relaxed constraint"),
    ("B4_GBMStack", "3_StaticHybrid", "nonlinear stacking"),
    ("B3_RidgeStack", "4_LearnedGate", "constraint relaxation vs context"),
    ("B2_FWLS", "4_LearnedGate", "FWLS vs its constrained special case"),
    ("B4_GBMStack", "4_LearnedGate", "best baseline vs best gate"),
    ("6_GAEvolvedGate", "4_LearnedGate", "GA search vs hand-picked features"),
    ("7_SequentialGate", "4_LearnedGate", "BiLSTM vs linear gate"),
    ("5_BanditGate", "3_StaticHybrid", "bandit vs the constant it replaces"),
    ("8_CalibratedGate", "4_LearnedGate", "adding the global correction to the gate"),
    ("8_CalibratedGate", "B2_FWLS", "calibrated gate vs unconstrained linear stacking"),
    ("8_CalibratedGate", "B4_GBMStack", "calibrated gate vs nonlinear stacking"),
    ("8_CalibratedGate", "8a_CalibratedNoCounts", "do the count features earn their place"),
]


def per_seed_rmse(data, name, segment="overall"):
    runs = data["per_seed"].get(name)
    if runs is None:
        return None
    return np.array([r[segment]["rmse"] for r in runs if segment in r], dtype=float)


def paired(a, b):
    d = a - b
    n = len(d)
    out = {"diffs": d, "mean": float(d.mean()), "n": n,
           "std": float(d.std(ddof=1)) if n > 1 else float("nan"),
           "wins": int((d < 0).sum()), "losses": int((d > 0).sum())}
    if n > 1 and np.any(d != d[0]):
        out["p"] = float(stats.ttest_rel(a, b).pvalue)
    else:
        out["p"] = float("nan")
    return out


def verdict(r):
    n = r["n"]
    if r["wins"] == n:
        return f"A better {n}/{n}"
    if r["losses"] == n:
        return f"B better {n}/{n}"
    return f"mixed {r['wins']}/{n}"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.METRICS_DIR / "multiseed_full_comparison.json"
    data = json.loads(path.read_text())
    seeds = data["seeds"]
    print(f"source: {path}")
    print(f"seeds:  {seeds}  (n={len(seeds)})\n")

    for name, runs in data["per_seed"].items():
        if len(runs) != len(seeds):
            raise SystemExit(f"{name} has {len(runs)} runs for {len(seeds)} seeds -- "
                             f"checkpoint is inconsistent, refusing to pair")
    if len(seeds) < 2:
        raise SystemExit("need at least 2 seeds for a paired comparison")

    # How much of the across-seed spread is shared split difficulty? Measured on
    # the pairs themselves: compare the spread of each model's own RMSE (what the
    # +/- bars show) with the spread of the paired difference. Centring across
    # ALL models instead is wrong -- ContentBased swings far more than the rest
    # and dominates the centre, which can push the estimate below zero.
    raw, diff = [], []
    for a_name, b_name, _ in KEY_PAIRS:
        a, b = per_seed_rmse(data, a_name), per_seed_rmse(data, b_name)
        if a is None or b is None:
            continue
        raw.append((a.std(ddof=1) + b.std(ddof=1)) / 2)
        diff.append((a - b).std(ddof=1))
    raw_std, resid_std = float(np.mean(raw)), float(np.mean(diff))
    shared = 1 - resid_std / raw_std if raw_std > 0 else float("nan")
    print("Why paired: across-seed spread, averaged over the compared pairs")
    print(f"  one model's RMSE (what +/- bars show)   {raw_std:.5f}")
    print(f"  paired difference A-B                   {resid_std:.5f}")
    if shared >= 0:
        print(f"  -> the paired comparison is {raw_std / resid_std:.1f}x tighter; "
              f"{shared:.0%} of the bar is split difficulty both models share\n")
    else:
        print("  -> paired differences are NOT tighter here; the models disagree about "
              "which seeds are hard, so the raw bars are not misleading\n")

    header = f"{'A':18s} {'B':16s} {'mean(A-B)':>10s} {'std':>8s} {'p':>7s}  {'verdict':14s} per-seed A-B"
    print(header)
    print("-" * (len(header) + 12))
    results = {}
    for a_name, b_name, label in KEY_PAIRS:
        a, b = per_seed_rmse(data, a_name), per_seed_rmse(data, b_name)
        if a is None or b is None:
            print(f"{a_name:18s} {b_name:16s}  (missing from this run)")
            continue
        r = paired(a, b)
        diffs = " ".join(f"{x:+.4f}" for x in r["diffs"])
        p = f"{r['p']:.3f}" if not np.isnan(r["p"]) else "  n/a"
        print(f"{a_name:18s} {b_name:16s} {r['mean']:+10.4f} {r['std']:8.4f} {p:>7s}  "
              f"{verdict(r):14s} {diffs}")
        results[f"{a_name}__vs__{b_name}"] = {
            "label": label, "mean_diff": r["mean"], "std_diff": r["std"], "p_paired_t": r["p"],
            "a_better_seeds": r["wins"], "b_better_seeds": r["losses"], "n_seeds": r["n"],
            "per_seed_diff": r["diffs"].tolist(),
        }
    print("\nNegative mean(A-B) = A has lower RMSE. With few seeds, trust the win count over p.")

    out = path.parent / "multiseed_paired_analysis.json"
    out.write_text(json.dumps({"seeds": seeds, "reference": REFERENCE,
                               "shared_split_fraction": shared,
                               "mean_raw_std": raw_std, "mean_paired_diff_std": resid_std,
                               "pairs": results}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
