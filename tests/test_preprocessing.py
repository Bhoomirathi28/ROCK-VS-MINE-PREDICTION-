"""Step 1 checks: the data is clean and the split is correct."""
import numpy as np
import pandas as pd

from src import config
from src.data_preprocessing import (audit_data, clean_data, load_raw_data,
                                    split_data)


def test_raw_shape_and_columns():
    df = load_raw_data()
    assert df.shape == (208, 61)
    assert list(df.columns) == config.FEATURE_NAMES + [config.TARGET]


def test_clean_data_has_no_quality_issues():
    report = audit_data(clean_data(load_raw_data()))
    assert report["missing_values"] == 0
    assert report["duplicate_rows"] == 0
    assert report["out_of_range_values"] == 0
    assert report["invalid_labels"] == 0


def test_cleaning_repairs_dirty_rows():
    df = load_raw_data().head(10).copy()
    df.loc[0, "band_01"] = np.nan          # missing value
    df.loc[1, "band_02"] = 1.7             # out of range
    df.loc[2, config.TARGET] = " r "       # messy label
    df.loc[3, config.TARGET] = "X"         # invalid label
    df = pd.concat([df, df.iloc[[4]]])     # duplicate row
    out = clean_data(df)
    assert out[config.FEATURE_NAMES].isna().sum().sum() == 0
    assert out[config.FEATURE_NAMES].max().max() <= 1.0
    assert set(out[config.TARGET]) <= {"R", "M"}
    assert len(out) == 9                   # 11 rows - 1 invalid label - 1 duplicate


def test_split_is_stratified_and_disjoint():
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_data(df, save=False)
    assert len(X_train) == 166 and len(X_test) == 42
    assert set(X_train.index).isdisjoint(X_test.index)
    full_ratio = (df[config.TARGET] == "M").mean()
    assert abs((y_train == "M").mean() - full_ratio) < 0.02
    assert abs((y_test == "M").mean() - full_ratio) < 0.03


def test_split_is_reproducible():
    df = clean_data(load_raw_data())
    a = split_data(df, save=False)[1].index
    b = split_data(df, save=False)[1].index
    assert list(a) == list(b)
