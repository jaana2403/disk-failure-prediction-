"""Feature engineering + preprocessing.

Two responsibilities:
1. `add_engineered_features` — create new, more informative columns from the raw
   SMART counters (this is the feature-engineering skill you're building on).
2. `build_preprocessor` — a scikit-learn transformer that imputes missing values
   and scales features, so the exact same steps run at train time and in the app.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import FEATURE_COLUMNS

# Columns produced by add_engineered_features(), modeled alongside the raw ones.
ENGINEERED_COLUMNS = ["total_error_count", "sectors_per_hour", "runs_hot"]
MODEL_COLUMNS = FEATURE_COLUMNS + ENGINEERED_COLUMNS


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive higher-signal features from raw SMART attributes.

    Returns a new DataFrame; the input is not modified.
    """
    out = df.copy()

    # Aggregate all error-type counters into one overall "health" signal.
    out["total_error_count"] = (
        out["smart_5_raw"].fillna(0)
        + out["smart_187_raw"].fillna(0)
        + out["smart_197_raw"].fillna(0)
        + out["smart_198_raw"].fillna(0)
    )

    # Reallocated sectors relative to age: young drives with many sectors are
    # far more suspicious than old drives with the same count.
    hours = out["smart_9_raw"].fillna(0) + 1  # +1 avoids divide-by-zero
    out["sectors_per_hour"] = out["smart_5_raw"].fillna(0) / hours

    # Simple binary flag for running hot (>= 40C).
    out["runs_hot"] = (out["smart_194_raw"] >= 40).astype(int)

    return out


def build_preprocessor() -> ColumnTransformer:
    """Impute missing values (median) then standardize every model column."""
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipe, MODEL_COLUMNS)],
        remainder="drop",
    )
