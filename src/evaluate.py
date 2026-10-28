"""Evaluation helpers: hold-out metrics, timing and repeated cross-validation."""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from src import config


def mine_probability(model, X) -> np.ndarray:
    """Probability of the positive class (Mine) for each row."""
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(config.POSITIVE_LABEL)]


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """Metrics on the untouched test set, with Mine as the positive class."""
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    latency_ms = (time.perf_counter() - t0) * 1000 / len(X_test)
    y_prob = mine_probability(model, X_test)
    y_true_bin = (np.asarray(y_test) == config.POSITIVE_LABEL).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=["R", "M"]).ravel()
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "overfit_gap": train_acc - test_acc,
        "precision": precision_score(y_test, y_pred, pos_label="M"),
        "recall": recall_score(y_test, y_pred, pos_label="M"),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_test, y_pred, pos_label="M"),
        "roc_auc": roc_auc_score(y_true_bin, y_prob),
        "missed_mines": int(fn),
        "false_alarms": int(fp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],  # rows: true R, M
        "latency_ms_per_sample": latency_ms,
    }


def repeated_cv(model, X, y) -> np.ndarray:
    """10 x 5-fold stratified CV accuracy on the training set.

    A 42-row test set moves 2.4 percentage points per misclassified sample,
    so CV over 50 folds gives a far more stable estimate for model selection.
    """
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS,
                                 n_repeats=config.CV_REPEATS,
                                 random_state=config.RANDOM_STATE)
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
