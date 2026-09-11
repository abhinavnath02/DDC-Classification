"""Train an improved classifier for the OCR pipeline.

This script is SEPARATE from the pipeline — the pipeline only loads a
frozen model, never trains. Run this script to produce the model artifact.

Model: Dual TF-IDF (word + char n-grams) + Calibrated Logistic Regression
- FeatureUnion of word (1,2)-grams and character (2,5)-grams
- GridSearchCV with GroupKFold for hyperparameter tuning
- CalibratedClassifierCV for calibrated probability outputs
- Splits by split_group_id using GroupShuffleSplit
- Serializes to models/baseline_classifier.joblib
- Reports per-class precision/recall/F1 and macro-F1

Usage:
    python -m src.ocr_pipeline.train_baseline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    GridSearchCV,
)
from sklearn.pipeline import FeatureUnion, Pipeline

# Import the existing loader
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ddc_data.load_data import load_data


def train_and_save(
    model_path: str | Path | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Train an improved model and save it.

    Parameters
    ----------
    model_path : where to save the model (default: models/baseline_classifier.joblib)
    test_size : fraction for test split
    val_size : fraction for validation split (from remaining after test)
    random_state : for reproducibility

    Returns
    -------
    dict with training metrics and model path
    """
    if model_path is None:
        model_path = _REPO_ROOT / "models" / "baseline_classifier.joblib"

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data using the existing loader — single source of truth
    data = load_data()
    X = data["X"]           # raw_text strings
    y = data["y"]           # label_level_1 strings (000–900)
    groups = data["groups"] # split_group_id

    X_arr = np.array(X)
    y_arr = np.array(y)
    groups_arr = np.array(groups)

    # Split by groups — no split_group_id may appear in more than one split
    # First: separate test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(X_arr, y_arr, groups_arr))

    # Then: separate validation from training
    adj_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adj_val_size, random_state=random_state)
    train_idx, val_idx = next(
        gss_val.split(X_arr[trainval_idx], y_arr[trainval_idx], groups_arr[trainval_idx])
    )
    # Map back to original indices
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    # Verify group integrity
    train_groups = set(groups_arr[train_idx])
    val_groups = set(groups_arr[val_idx])
    test_groups = set(groups_arr[test_idx])
    assert train_groups.isdisjoint(val_groups), "Train/val group leak!"
    assert train_groups.isdisjoint(test_groups), "Train/test group leak!"
    assert val_groups.isdisjoint(test_groups), "Val/test group leak!"

    X_train, y_train = X_arr[train_idx], y_arr[train_idx]
    X_val, y_val = X_arr[val_idx], y_arr[val_idx]
    X_test, y_test = X_arr[test_idx], y_arr[test_idx]

    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Group counts: train={len(train_groups)}, val={len(val_groups)}, test={len(test_groups)}")

    # Log class distribution per split
    for name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        dist = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))
        print(f"  {name} class distribution: {dist}")

    # Build the dual-feature pipeline:
    # - Word n-grams (1,2) capture topic terms and bigram patterns
    # - Character n-grams (2,5) capture morphological cues, author name
    #   fragments, and language-specific subword patterns
    feature_union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",
            min_df=2,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            max_features=50000,
            ngram_range=(2, 5),
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2,
        )),
    ])

    pipeline = Pipeline([
        ("features", feature_union),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        )),
    ])

    # Hyperparameter tuning with GroupKFold (respects split_group_id)
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "features__word_tfidf__max_features": [10000, 30000],
        "features__char_tfidf__max_features": [30000, 50000],
    }

    group_kfold = GroupKFold(n_splits=5)

    print("\nRunning GridSearchCV with GroupKFold (5 folds)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=group_kfold,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(
        X_train.tolist(),
        y_train.tolist(),
        groups=groups_arr[train_idx].tolist(),
    )

    print(f"\nBest params: {grid_search.best_params_}")
    print(f"Best CV macro-F1: {grid_search.best_score_:.4f}")

    best_pipeline = grid_search.best_estimator_

    # Calibrate probabilities using CalibratedClassifierCV
    # Uses the validation set for isotonic calibration
    print("\nCalibrating probabilities with isotonic regression on validation set...")
    calibrated_model = CalibratedClassifierCV(
        best_pipeline,
        cv="prefit",
        method="isotonic",
    )
    calibrated_model.fit(X_val.tolist(), y_val.tolist())

    # Evaluate on validation set
    print("\n=== Validation Set Results ===")
    y_val_pred = calibrated_model.predict(X_val.tolist())
    val_report = classification_report(y_val, y_val_pred, digits=3, zero_division=0)
    print(val_report)

    # Evaluate on test set (for reference — not used for model selection)
    print("\n=== Test Set Results (reference only) ===")
    y_test_pred = calibrated_model.predict(X_test.tolist())
    test_report = classification_report(y_test, y_test_pred, digits=3, zero_division=0)
    print(test_report)

    print("\n=== Confusion Matrix (test set) ===")
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    print(f"Labels: {labels}")
    print(cm)

    # Save model
    joblib.dump(calibrated_model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save split assignments and best hyperparameters for reproducibility
    split_record = {
        "random_state": random_state,
        "test_size": test_size,
        "val_size": val_size,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "test_group_count": len(test_groups),
        "best_params": grid_search.best_params_,
        "best_cv_macro_f1": round(grid_search.best_score_, 4),
        "model_type": "CalibratedClassifierCV(FeatureUnion[word_tfidf+char_tfidf]+LogisticRegression)",
        "calibration_method": "isotonic",
        "calibrated": True,
    }
    split_path = model_path.parent / "split_record.json"
    split_path.write_text(json.dumps(split_record, indent=2) + "\n")
    print(f"Split record saved to: {split_path}")

    return {
        "model_path": str(model_path),
        "val_report": val_report,
        "test_report": test_report,
        "split_record": split_record,
    }


if __name__ == "__main__":
    train_and_save()
