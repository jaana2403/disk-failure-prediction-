"""Train and compare models, then save the best one.

Pipeline per model:  preprocess -> SMOTE (train only) -> classifier

SMOTE lives inside an imbalanced-learn pipeline, so it is applied ONLY while
fitting (never to the test data) — that prevents the classic mistake of leaking
resampled rows into evaluation. We compare a Logistic Regression baseline with a
Random Forest and keep whichever has the better ROC-AUC.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_PATH,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    ensure_dirs,
)
from src.data_loader import load_data
from src.features import add_engineered_features, build_preprocessor


def make_models() -> dict:
    """Return the candidate models to compare.

    `class_weight="balanced"` is a second defense against imbalance, on top of
    SMOTE, telling the model to care more about the rare failure class.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )


def main():
    ensure_dirs()

    df = add_engineered_features(load_data())
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Save the held-out test set so evaluate.py scores the exact same rows.
    test_set = X_test.copy()
    test_set[TARGET_COLUMN] = y_test.values
    test_set.to_csv(PROCESSED_DIR / "test_set.csv", index=False)

    best_name, best_pipeline, best_auc = None, None, -1.0
    for name, model in make_models().items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        preds = pipeline.predict(X_test)
        auc = roc_auc_score(y_test, proba)

        print(f"\n===== {name} =====")
        print(f"ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, preds, digits=3))

        if auc > best_auc:
            best_name, best_pipeline, best_auc = name, pipeline, auc

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f})")
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
