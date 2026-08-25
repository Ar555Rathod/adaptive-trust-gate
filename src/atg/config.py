"""Project-wide paths and constants for the Adaptive Trust Gate pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-latest-small"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

RATINGS_CSV = RAW_DIR / "ratings.csv"
MOVIES_CSV = RAW_DIR / "movies.csv"
TAGS_CSV = RAW_DIR / "tags.csv"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
USER_SEGMENTS_CSV = PROCESSED_DIR / "user_segments.csv"

RANDOM_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Sparsity segmentation, defined on each user's TRAIN-set rating count
# (matches the proposal: cold < 5, warm 5-50, power 50+).
COLD_MAX = 5   # ratings < this -> cold
WARM_MAX = 50  # ratings < this (and >= COLD_MAX) -> warm; >= this -> power

# Simulated cold-start split: MovieLens guarantees every user has >=20
# ratings, so a plain random 70/15/15 split almost never leaves any user
# with <5 train ratings, leaving the "cold" segment structurally empty.
# To get genuine cold-start cases we designate a quota of real users whose
# TRAIN allocation is deliberately capped to a handful of their own ratings
# (see atg.data.splits.build_splits) -- everyone else still gets a standard
# per-user 70/15/15 split.
COLD_USER_FRAC = 0.15
COLD_TRAIN_MIN = 1
COLD_TRAIN_MAX = 4

# Model 7 (Sequential BiLSTM Gate): a query (u,i,t) needs at least this many
# of the user's own TRAIN ratings strictly before timestamp t to bother
# running the BiLSTM at all; below it, there just isn't a sequence worth
# encoding, so the gate falls back to Model 3's fixed alpha (proposal's
# documented Model 7 fallback behaviour). Matches COLD_MAX so the fallback
# population lines up with the cold segment definition.
SEQ_MIN_HISTORY = 5
SEQ_LONG_LEN = 50   # long-term window: up to this many most-recent prior ratings
SEQ_SHORT_LEN = 5   # short-term window: up to this many most-recent prior ratings

RATING_MIN = 0.5
RATING_MAX = 5.0

for _d in (PROCESSED_DIR, MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
