"""Interpretability artifacts for the report:
  1. Model 4's gate coefficients (it's a linear/logistic-style model, so
     every weight is directly readable) as a bar chart.
  2. g(u,i) vs. user sparsity (log_user_count) for Model 4 -- does the gate
     actually behave the way the sparsity-aware story claims?
  3. An accuracy-vs-interpretability tradeoff plot across Models 3-7, using
     a qualitative 1-5 interpretability proxy (documented inline -- this is
     NOT a formal metric, just an ordinal ranking of "can you explain a
     single g(u,i) decision from its inputs").
  4. A case-study table: concrete (user, item) rows spanning segments where
     the gates visibly disagree with a fixed alpha, for qualitative
     discussion in the report.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atg import config
from atg.gates.features import FEATURE_COLUMNS

# Documented, qualitative 1-5 scale (5 = fully transparent, 1 = black box):
# can a reader explain why g(u,i) took a specific value from its inputs?
INTERPRETABILITY_SCORE = {
    "3_StaticHybrid": 5,     # single global constant
    "4_LearnedGate": 4,      # linear formula, every coefficient readable
    "6_GAEvolvedGate": 4,    # same linear formula, GA only chose which features
    "5_BanditGate": 3,       # linear per-arm reward model, but discrete arms + online state
    "7_SequentialGate": 1,   # BiLSTM hidden state, deliberately least interpretable
}


def plot_gate4_coefficients():
    with open(config.MODELS_DIR / "model4_learned_gate.pkl", "rb") as f:
        gate = pickle.load(f)
    weights = gate.net.W2.ravel()
    bias = float(gate.net.b2[0])

    order = np.argsort(weights)
    names = [FEATURE_COLUMNS[i] for i in order]
    vals = weights[order]

    plt.figure(figsize=(7.5, 4.5))
    colors = ["#c0392b" if v < 0 else "#2471a3" for v in vals]
    plt.barh(names, vals, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(f"Standardized coefficient (intercept g0={bias:.3f})")
    plt.title("Model 4 (Learned Gate): every g(u,i) coefficient")
    plt.tight_layout()
    path = config.METRICS_DIR / "model4_gate_coefficients.png"
    plt.savefig(path, dpi=150)
    print(f"Saved -> {path}")
    return dict(zip(FEATURE_COLUMNS, weights.tolist())), bias


def plot_g_vs_sparsity():
    df = pd.read_csv(config.PREDICTIONS_DIR / "model4_learned_gate_test.csv")
    df["log_user_count"] = np.log1p(df["train_rating_count"])
    bins = np.linspace(df["log_user_count"].min(), df["log_user_count"].max(), 16)
    df["bin"] = pd.cut(df["log_user_count"], bins)
    binned = df.groupby("bin", observed=True).agg(
        mean_g=("gate_g", "mean"), mean_log_count=("log_user_count", "mean"), n=("gate_g", "size")
    ).dropna()

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(binned["mean_log_count"], binned["mean_g"], marker="o")
    plt.xlabel("log(1 + user's TRAIN rating count)")
    plt.ylabel("Mean g(u,i)  [weight on CF]")
    plt.title("Model 4: gate weight vs. user sparsity (test set)")
    plt.axhline(0.84, color="gray", linestyle="--", linewidth=1, label="Model 3 fixed alpha=0.84")
    plt.legend()
    plt.tight_layout()
    path = config.METRICS_DIR / "model4_g_vs_sparsity.png"
    plt.savefig(path, dpi=150)
    print(f"Saved -> {path}")


def plot_interpretability_tradeoff():
    with open(config.METRICS_DIR / "full_comparison.json") as f:
        full = json.load(f)

    names = list(INTERPRETABILITY_SCORE.keys())
    rmse = [full["accuracy"][n]["overall"]["rmse"] for n in names]
    interp = [INTERPRETABILITY_SCORE[n] for n in names]

    plt.figure(figsize=(7, 5))
    plt.scatter(interp, rmse, s=80, color="#2471a3")
    for n, x, y in zip(names, interp, rmse):
        plt.annotate(n.split("_", 1)[1], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xlabel("Interpretability (qualitative, 5=transparent formula, 1=black box)")
    plt.ylabel("Overall test RMSE (lower is better)")
    plt.title("Interpretability vs. accuracy across the hybrid models")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    path = config.METRICS_DIR / "interpretability_vs_accuracy.png"
    plt.savefig(path, dpi=150)
    print(f"Saved -> {path}")


def build_case_studies():
    master = pd.read_csv(config.PREDICTIONS_DIR / "full_comparison_predictions.csv")
    seq = pd.read_csv(config.PREDICTIONS_DIR / "model7_sequential_gate_test.csv")
    m4 = pd.read_csv(config.PREDICTIONS_DIR / "model4_learned_gate_test.csv")
    m6 = pd.read_csv(config.PREDICTIONS_DIR / "model6_ga_gate_test.csv")

    master["g4"] = m4["gate_g"]
    master["g6"] = m6["gate_g"]
    master["g7"] = seq["gate_g"]
    master["used_fallback_7"] = seq["used_fallback"]
    master["disagreement"] = (master["pred__1_CF_SVDpp"] - master["pred__2_ContentBased"]).abs()

    rows = []
    rng = np.random.default_rng(config.RANDOM_SEED)
    for seg in ("cold", "warm", "power"):
        sub = master[master["segment"] == seg].sort_values("disagreement", ascending=False)
        picked = sub.head(20).sample(min(3, len(sub)), random_state=config.RANDOM_SEED)
        rows.append(picked)
    case_df = pd.concat(rows).reset_index(drop=True)

    cols = ["userId", "movieId", "segment", "train_rating_count", "rating",
            "pred__1_CF_SVDpp", "pred__2_ContentBased", "pred__3_StaticHybrid",
            "g4", "pred__4_LearnedGate", "g6", "pred__6_GAEvolvedGate",
            "g7", "used_fallback_7", "pred__7_SequentialGate"]
    case_df = case_df[cols].round(3)
    path = config.METRICS_DIR / "case_studies.csv"
    case_df.to_csv(path, index=False)
    print(f"Saved -> {path}")
    print("\nCase studies (high CF/CB disagreement, by segment):")
    print(case_df.to_string(index=False))
    return case_df


def main():
    coefs, bias = plot_gate4_coefficients()
    plot_g_vs_sparsity()
    plot_interpretability_tradeoff()
    case_df = build_case_studies()

    summary = {
        "model4_gate_coefficients": coefs,
        "model4_gate_bias": bias,
        "interpretability_scores": INTERPRETABILITY_SCORE,
    }
    with open(config.METRICS_DIR / "interpretability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {config.METRICS_DIR / 'interpretability_summary.json'}")


if __name__ == "__main__":
    main()
