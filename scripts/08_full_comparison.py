"""Consolidate all 7 models onto the identical TEST set: segmented
RMSE/MAE + ranking metrics (Spearman/Kendall/NDCG/ARHR/ROC-AUC), a compute-
cost table, and pairwise statistical significance (paired bootstrap +
Wilcoxon) between every model pair, overall and per segment. This is the
central results bundle the report's tables/figures are built from -- no
retraining, just reading back the predictions each model script already
saved (all row-aligned to experts_test.csv, verified by construction: every
gate script reads that same file and only appends columns).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atg import config
from atg.eval.metrics import segmented_rating_metrics
from atg.eval.ranking import segmented_ranking_metrics
from atg.eval.significance import compare_models

MODELS = {
    "1_CF_SVDpp": ("experts_test.csv", "cf_pred"),
    "2_ContentBased": ("experts_test.csv", "cb_pred"),
    "3_StaticHybrid": ("model3_static_hybrid_test.csv", "hybrid_pred"),
    "4_LearnedGate": ("model4_learned_gate_test.csv", "hybrid_pred"),
    "5_BanditGate": ("model5_bandit_gate_test.csv", "hybrid_pred"),
    "6_GAEvolvedGate": ("model6_ga_gate_test.csv", "hybrid_pred"),
    "7_SequentialGate": ("model7_sequential_gate_test.csv", "hybrid_pred"),
}

COMPUTE_COST_SOURCES = {
    "1_CF_SVDpp": ("experts_baseline.json", ["compute_cost", "cf_svdpp"]),
    "2_ContentBased": ("experts_baseline.json", ["compute_cost", "content_based"]),
    "3_StaticHybrid": ("model3_static_hybrid.json", ["compute_cost"]),
    "4_LearnedGate": ("model4_learned_gate.json", ["compute_cost"]),
    "5_BanditGate": ("model5_bandit_gate.json", ["compute_cost"]),
    "6_GAEvolvedGate": ("model6_ga_gate.json", ["compute_cost"]),
    "7_SequentialGate": ("model7_sequential_gate.json", ["compute_cost"]),
}


def dig(d, path):
    for k in path:
        d = d[k]
    return d


def main():
    base = pd.read_csv(config.PREDICTIONS_DIR / "experts_test.csv")
    master = base[["userId", "itemId", "rating", "segment", "train_rating_count"]].copy()

    accuracy = {}
    ranking = {}
    for name, (fname, col) in MODELS.items():
        df = pd.read_csv(config.PREDICTIONS_DIR / fname)
        assert (df["userId"].to_numpy() == base["userId"].to_numpy()).all(), f"{name} not row-aligned"
        assert (df["itemId"].to_numpy() == base["itemId"].to_numpy()).all(), f"{name} not row-aligned"
        master[f"pred__{name}"] = df[col].to_numpy()

        scored = base.copy()
        scored["pred"] = df[col].to_numpy()
        accuracy[name] = segmented_rating_metrics(scored, true_col="rating", pred_col="pred")
        ranking[name] = segmented_ranking_metrics(scored, true_col="rating", pred_col="pred")
        print(f"{name:20s} RMSE(overall)={accuracy[name]['overall']['rmse']:.4f}  "
              f"NDCG(overall)={ranking[name]['overall']['ndcg']:.4f}  "
              f"Spearman(overall)={ranking[name]['overall']['spearman']:.4f}")

    compute_cost = {}
    for name, (fname, path) in COMPUTE_COST_SOURCES.items():
        with open(config.METRICS_DIR / fname) as f:
            data = json.load(f)
        compute_cost[name] = dig(data, path)

    print("\nCompute cost:")
    for name, cc in compute_cost.items():
        print(f"  {name:20s} train_time_sec={cc.get('train_time_sec', cc.get('train_time_sec')):.3f}  n_params={cc.get('n_params', cc.get('n_params_repr_size'))}")

    print("\nRunning pairwise significance tests (paired bootstrap + Wilcoxon)...")
    pred_cols = {name: f"pred__{name}" for name in MODELS}
    sig_df = compare_models(master, true_col="rating", model_preds=pred_cols, segment_col="segment", n_boot=2000)
    sig_path = config.METRICS_DIR / "full_comparison_significance.csv"
    sig_df.to_csv(sig_path, index=False)
    print(f"Saved -> {sig_path}")

    key_pairs = sig_df[(sig_df["segment"] == "overall") &
                        (((sig_df["model_a"] == "3_StaticHybrid") & (sig_df["model_b"].str.startswith(("4_", "5_", "6_", "7_")))) |
                         ((sig_df["model_b"] == "3_StaticHybrid") & (sig_df["model_a"].str.startswith(("4_", "5_", "6_", "7_")))))]
    print("\nModel 3 (Static Hybrid) vs each adaptive gate, overall test set:")
    print(key_pairs[["model_a", "model_b", "rmse_a", "rmse_b", "mean_diff", "bootstrap_significant", "wilcoxon_p"]].to_string(index=False))

    accuracy_table = []
    for name in MODELS:
        for seg in ("overall", "cold", "warm", "power"):
            if seg in accuracy[name]:
                row = {"model": name, "segment": seg}
                row.update(accuracy[name][seg])
                row.update({f"rank_{k}": v for k, v in ranking[name][seg].items()})
                accuracy_table.append(row)
    accuracy_df = pd.DataFrame(accuracy_table)
    table_path = config.METRICS_DIR / "full_comparison_table.csv"
    accuracy_df.to_csv(table_path, index=False)
    print(f"\nSaved consolidated accuracy+ranking table -> {table_path}")

    master_path = config.PREDICTIONS_DIR / "full_comparison_predictions.csv"
    master.to_csv(master_path, index=False)
    print(f"Saved master prediction matrix -> {master_path}")

    results = {
        "accuracy": accuracy,
        "ranking": ranking,
        "compute_cost": compute_cost,
    }
    out_path = config.METRICS_DIR / "full_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
