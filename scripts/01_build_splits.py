"""Build and cache the single 70/15/15 train/val/test split used by all 7 models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atg import config
from atg.data.splits import build_splits, save_splits
from atg.utils.segments import build_user_segments


def main():
    train_df, val_df, test_df = build_splits()
    save_splits(train_df, val_df, test_df)

    n = len(train_df) + len(val_df) + len(test_df)
    print(f"Total ratings: {n}")
    print(f"  train: {len(train_df):>6} ({len(train_df)/n:.1%})")
    print(f"  val:   {len(val_df):>6} ({len(val_df)/n:.1%})")
    print(f"  test:  {len(test_df):>6} ({len(test_df)/n:.1%})")

    user_segments = build_user_segments(train_df)
    user_segments.to_csv(config.USER_SEGMENTS_CSV, index=False)
    print("\nUser sparsity segments (by TRAIN rating count):")
    print(user_segments["segment"].value_counts())

    print(f"\nSaved splits to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
