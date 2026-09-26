"""Exploratory Data Analysis using seaborn + matplotlib.

Run this to *understand* the data before modeling. It saves plots to
reports/figures/ so you can look at them and reason about the problem — this is
exactly the pandas/seaborn/matplotlib work you already know, applied to a real
predictive-maintenance dataset.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import FEATURE_COLUMNS, FIGURES_DIR, TARGET_COLUMN, ensure_dirs
from src.data_loader import load_data

sns.set_theme(style="whitegrid")


def plot_class_balance(df, ax):
    """Show how rare failures are — the central challenge of this project."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    sns.barplot(x=["Healthy (0)", "Failed (1)"], y=counts.values, ax=ax)
    ax.set_title("Class balance (failures are rare)")
    ax.set_ylabel("Number of drives")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")


def plot_feature_distributions(df):
    """Compare each SMART feature for healthy vs failed drives."""
    n = len(FEATURE_COLUMNS)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for ax, feature in zip(axes, FEATURE_COLUMNS):
        sns.kdeplot(
            data=df, x=feature, hue=TARGET_COLUMN,
            common_norm=False, fill=True, alpha=0.4, ax=ax,
        )
        ax.set_title(feature)
    for ax in axes[n:]:
        ax.remove()
    fig.tight_layout()
    return fig


def plot_correlation(df):
    """Heatmap of feature correlations (spot redundant SMART signals)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlations")
    fig.tight_layout()
    return fig


def main():
    ensure_dirs()
    df = load_data()

    print("Summary statistics:")
    print(df.describe(include="all").T)
    print(f"\nMissing values per column:\n{df.isna().sum()}")

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_class_balance(df, ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=120)

    plot_feature_distributions(df).savefig(
        FIGURES_DIR / "feature_distributions.png", dpi=120
    )
    plot_correlation(df).savefig(FIGURES_DIR / "correlation.png", dpi=120)

    print(f"\nSaved EDA figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
