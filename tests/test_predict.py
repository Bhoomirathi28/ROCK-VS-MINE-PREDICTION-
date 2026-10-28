"""Inference checks: validation, output shape and sanity of predictions."""
import pandas as pd
import pytest

from src import config
from src.predict import InvalidInputError, Predictor, validate_features


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


@pytest.fixture(scope="module")
def test_set():
    return pd.read_csv(config.TEST_PATH)


@pytest.mark.parametrize("bad", [[0.1] * 59, [0.1] * 61, [0.1] * 59 + [1.5],
                                 [0.1] * 59 + [-0.2], ["a"] * 60, [float("nan")] * 60])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(InvalidInputError):
        validate_features(bad)


def test_prediction_structure(predictor, test_set):
    row = test_set.iloc[0][config.FEATURE_NAMES].tolist()
    out = predictor.predict(row)
    assert out["label"] in ("M", "R")
    assert 0.5 <= out["confidence"] <= 1.0
    assert abs(out["probability_mine"] + out["probability_rock"] - 1) < 1e-9
    assert len(out["all_models"]) == 4
    assert out["consensus"]["mine_votes"] + out["consensus"]["rock_votes"] == 4


def test_every_model_can_be_selected(predictor, test_set):
    row = test_set.iloc[0][config.FEATURE_NAMES].tolist()
    for key in ("logistic_regression", "svm", "random_forest", "knn"):
        assert predictor.predict(row, key)["model"] == key


def test_unknown_model_is_rejected(predictor, test_set):
    with pytest.raises(InvalidInputError):
        predictor.predict(test_set.iloc[0][config.FEATURE_NAMES].tolist(), "xgboost")


def test_deployed_model_beats_80_percent_on_test_set(predictor, test_set):
    rows = test_set[config.FEATURE_NAMES].values.tolist()
    preds = [r["label"] for r in predictor.predict_many(rows)]
    accuracy = (pd.Series(preds).values == test_set[config.TARGET].values).mean()
    assert accuracy >= 0.80
