"""Data loading, quality audit, cleaning and train/test splitting.

Pipeline
--------
1. load_raw_data   - read the headerless CSV and attach meaningful column names
2. audit_data      - measure quality issues *before* touching anything
3. clean_data      - fix what the audit found (labels, missing, duplicates, range)
4. split_data      - stratified 80/20 hold-out split, saved to data/processed/

Scaling is deliberately NOT done here. It lives inside each model's sklearn
Pipeline so the scaler is fitted on training folds only (no data leakage).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)


def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw sonar CSV (no header row) and name its 61 columns."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] != config.N_FEATURES + 1:
        raise ValueError(
            f"Expected {config.N_FEATURES + 1} columns, found {df.shape[1]}"
        )
    df.columns = config.FEATURE_NAMES + [config.TARGET]
    return df


def _iqr_outlier_count(features: pd.DataFrame) -> int:
    q1, q3 = features.quantile(0.25), features.quantile(0.75)
    iqr = q3 - q1
    mask = (features < q1 - 1.5 * iqr) | (features > q3 + 1.5 * iqr)
    return int(mask.to_numpy().sum())


def audit_data(df: pd.DataFrame) -> dict:
    """Return a dictionary describing data quality. Nothing is modified."""
    features = df[config.FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    labels = df[config.TARGET].astype(str).str.strip().str.upper()
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_values": int(features.isna().sum().sum()
                              + df[config.TARGET].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "out_of_range_values": int(((features < config.FEATURE_MIN)
                                    | (features > config.FEATURE_MAX)).sum().sum()),
        "invalid_labels": int((~labels.isin(config.CLASSES)).sum()),
        "iqr_outlier_cells": _iqr_outlier_count(features),
        "class_counts": {k: int(v) for k, v in labels.value_counts().items()},
        "feature_min": float(np.nanmin(features.to_numpy())),
        "feature_max": float(np.nanmax(features.to_numpy())),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules. Safe to run even when the data is already clean.

    Rules
    -----
    * Labels: strip whitespace, upper-case, drop rows whose label is not R/M.
    * Features: coerce to float; missing values filled with the column median
      (median is robust to the skewed energy distributions in sonar bands).
    * Duplicates: exact duplicate rows are removed so they cannot appear in
      both train and test sets.
    * Range: energies are normalised to [0, 1] by the dataset authors, so any
      value outside that range is a recording error and is clipped.
    * IQR outliers are reported but NOT removed: with only 208 samples, and
      every value inside the valid physical range, these are genuine strong
      echoes rather than errors.
    """
    out = df.copy()
    out[config.TARGET] = out[config.TARGET].astype(str).str.strip().str.upper()
    before = len(out)
    out = out[out[config.TARGET].isin(config.CLASSES)]
    if len(out) < before:
        logger.info("Dropped %d rows with invalid labels", before - len(out))

    out[config.FEATURE_NAMES] = out[config.FEATURE_NAMES].apply(
        pd.to_numeric, errors="coerce"
    )
    n_missing = int(out[config.FEATURE_NAMES].isna().sum().sum())
    if n_missing:
        out[config.FEATURE_NAMES] = out[config.FEATURE_NAMES].fillna(
            out[config.FEATURE_NAMES].median()
        )
        logger.info("Imputed %d missing values with column medians", n_missing)

    before = len(out)
    out = out.drop_duplicates()
    if len(out) < before:
        logger.info("Removed %d duplicate rows", before - len(out))

    out[config.FEATURE_NAMES] = out[config.FEATURE_NAMES].clip(
        config.FEATURE_MIN, config.FEATURE_MAX
    )
    return out.reset_index(drop=True)


def split_data(df: pd.DataFrame, save: bool = True):
    """Stratified 80/20 split with the same seed as the original notebook."""
    X = df[config.FEATURE_NAMES]
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_train.assign(**{config.TARGET: y_train}).to_csv(config.TRAIN_PATH, index=False)
        X_test.assign(**{config.TARGET: y_test}).to_csv(config.TEST_PATH, index=False)
    return X_train, X_test, y_train, y_test


def run_preprocessing(save: bool = True):
    """Full step 1 of the project: load -> audit -> clean -> audit -> split."""
    raw = load_raw_data()
    audit_before = audit_data(raw)
    clean = clean_data(raw)
    audit_after = audit_data(clean)
    X_train, X_test, y_train, y_test = split_data(clean, save=save)

    summary = {
        "before_cleaning": audit_before,
        "after_cleaning": audit_after,
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_class_counts": {k: int(v) for k, v in y_train.value_counts().items()},
            "test_class_counts": {k: int(v) for k, v in y_test.value_counts().items()},
            "test_size": config.TEST_SIZE,
            "stratified": True,
            "random_state": config.RANDOM_STATE,
        },
    }
    if save:
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.METRICS_DIR / "data_audit.json", "w") as fh:
            json.dump(summary, fh, indent=2)
    return clean, (X_train, X_test, y_train, y_test), summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    _, _, s = run_preprocessing()
    print(json.dumps(s, indent=2))
