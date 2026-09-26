"""Central configuration: paths, feature definitions, and constants.

Keeping everything here means every other script imports the same source of
truth, so changing a path or a feature list only happens in one place.
"""
from pathlib import Path

# --- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# The trained model is saved here and loaded by evaluate.py and the app.
MODEL_PATH = MODELS_DIR / "model.joblib"

# If you download a real Backblaze CSV, drop it here and the loader uses it.
RAW_CSV_PATH = RAW_DIR / "drive_stats.csv"

# --- SMART features we model on ----------------------------------------------
# These are the SMART attributes Backblaze found most predictive of failure.
# Column name -> human-readable meaning (used for labels in the app/plots).
SMART_FEATURES = {
    "smart_5_raw": "Reallocated Sectors Count",
    "smart_187_raw": "Reported Uncorrectable Errors",
    "smart_188_raw": "Command Timeout Count",
    "smart_197_raw": "Current Pending Sector Count",
    "smart_198_raw": "Offline Uncorrectable Sector Count",
    "smart_9_raw": "Power-On Hours",
    "smart_194_raw": "Temperature (Celsius)",
    "smart_193_raw": "Load Cycle Count",
}

FEATURE_COLUMNS = list(SMART_FEATURES.keys())
TARGET_COLUMN = "failure"  # 1 = failed, 0 = healthy

# --- Reproducibility ---------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    """Create output folders if they do not exist yet."""
    for path in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
