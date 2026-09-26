"""Load drive data for the project.

Two sources are supported:
1. A real Backblaze CSV placed at data/raw/drive_stats.csv -> used automatically.
2. Otherwise, a realistic *synthetic* SMART dataset is generated so you can run
   the whole project immediately with no download.

The synthetic generator deliberately mimics the real world: failures are RARE
(class imbalance) and failing drives tend to show elevated error-related SMART
counts. That imbalance is the core ML challenge this project teaches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    RAW_CSV_PATH,
    TARGET_COLUMN,
)


def generate_synthetic_data(
    n_samples: int = 20_000,
    failure_rate: float = 0.02,
    silent_failure_rate: float = 0.35,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create a realistic, imbalanced SMART dataset.

    Realism choices that make this a genuine ML challenge (not a giveaway):
    * Failures are rare (`failure_rate`), so the model must handle imbalance.
    * Failing-drive SMART values are only *moderately* elevated and OVERLAP the
      healthy range — a drive with a few bad sectors might be fine or failing.
    * A fraction of failures are "silent" (`silent_failure_rate`): they look
      completely healthy, mirroring drives that die with no SMART warning. This
      caps achievable recall, which is exactly the real-world limitation.
    """
    rng = np.random.default_rng(random_state)
    n_fail = int(n_samples * failure_rate)
    n_ok = n_samples - n_fail

    def healthy(size):
        # Most healthy drives are clean, but a tail have a few error counts.
        return {
            "smart_5_raw": rng.poisson(0.6, size),          # reallocated sectors
            "smart_187_raw": rng.poisson(0.4, size),        # uncorrectable errors
            "smart_188_raw": rng.poisson(0.4, size),        # command timeouts
            "smart_197_raw": rng.poisson(0.3, size),        # pending sectors
            "smart_198_raw": rng.poisson(0.3, size),        # offline uncorrectable
            "smart_9_raw": rng.integers(1_000, 40_000, size),   # power-on hours
            "smart_194_raw": rng.normal(32, 4, size).clip(20, 60),  # temperature
            "smart_193_raw": rng.integers(10_000, 300_000, size),   # load cycles
        }

    def failing(size):
        # Elevated but overlapping with healthy — the signal is weak and noisy.
        return {
            "smart_5_raw": rng.poisson(6, size),
            "smart_187_raw": rng.poisson(3, size),
            "smart_188_raw": rng.poisson(2, size),
            "smart_197_raw": rng.poisson(4, size),
            "smart_198_raw": rng.poisson(3, size),
            "smart_9_raw": rng.integers(5_000, 60_000, size),
            "smart_194_raw": rng.normal(35, 5, size).clip(20, 65),
            "smart_193_raw": rng.integers(50_000, 500_000, size),
        }

    ok_df = pd.DataFrame(healthy(n_ok))
    ok_df[TARGET_COLUMN] = 0

    # Split failures into "silent" (look healthy) and "symptomatic" (elevated).
    n_silent = int(n_fail * silent_failure_rate)
    n_symptomatic = n_fail - n_silent
    silent_df = pd.DataFrame(healthy(n_silent))
    symptomatic_df = pd.DataFrame(failing(n_symptomatic))
    fail_df = pd.concat([silent_df, symptomatic_df], ignore_index=True)
    fail_df[TARGET_COLUMN] = 1

    data = pd.concat([ok_df, fail_df], ignore_index=True)
    # Shuffle so failures are not all at the bottom.
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Inject a few missing values to mirror messy real logs (~1% per feature).
    for col in FEATURE_COLUMNS:
        mask = rng.random(len(data)) < 0.01
        data.loc[mask, col] = np.nan

    return data


def load_data(sample_size: int | None = 50_000) -> pd.DataFrame:
    """Return the dataset as a DataFrame.

    Uses the real CSV at data/raw/drive_stats.csv when present, otherwise falls
    back to synthetic data. `sample_size` caps rows from the real CSV so EDA and
    training stay fast on a laptop.
    """
    if RAW_CSV_PATH.exists():
        print(f"Loading real data from {RAW_CSV_PATH}")
        df = pd.read_csv(RAW_CSV_PATH)
        keep = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
        df = df[keep]
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        return df

    print("No real CSV found -> generating synthetic SMART data.")
    return generate_synthetic_data()


if __name__ == "__main__":
    frame = load_data()
    print(frame.head())
    print(f"\nShape: {frame.shape}")
    print(f"Failure rate: {frame[TARGET_COLUMN].mean():.3%}")
