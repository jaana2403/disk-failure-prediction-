"""Evaluate the saved model on the held-out test set.

Produces the metrics that actually matter for rare-failure prediction:
confusion matrix, precision/recall/F1, and a ROC curve. For predictive
maintenance, RECALL on the failure class is usually the key number — missing a
failing drive (a false negative) is costly.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    FIGURES_DIR,
    MODEL_PATH,
    PROCESSED_DIR,
    TARGET_COLUMN,
    ensure_dirs,
)

sns.set_theme(style="whitegrid")


def main():
    ensure_dirs()

    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found. Run: python src/train.py first.")

    test_path = PROCESSED_DIR / "test_set.csv"
    if not test_path.exists():
        raise SystemExit("No test set found. Run: python src/train.py first.")

    import pandas as pd

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    pipeline = joblib.load(MODEL_PATH)
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, preds, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=["Healthy", "Failed"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=120)

    print(f"\nSaved evaluation figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
