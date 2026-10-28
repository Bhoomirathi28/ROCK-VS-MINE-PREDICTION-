"""Inference: validate a 60-value sonar reading and classify it.

Library use:
    from src.predict import Predictor
    Predictor().predict([0.02, 0.037, ...])          # 60 floats

Command line:
    python -m src.predict --values "0.02,0.0371,...,0.0032"
    python -m src.predict --csv my_readings.csv        # one reading per row
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from src import config


class InvalidInputError(ValueError):
    """Raised when a reading is not 60 numbers between 0 and 1."""


def validate_features(values) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise InvalidInputError("Every value must be a number.") from None
    if arr.size != config.N_FEATURES:
        raise InvalidInputError(
            f"Expected {config.N_FEATURES} values, received {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise InvalidInputError("Values cannot be empty, NaN or infinite.")
    if arr.min() < config.FEATURE_MIN or arr.max() > config.FEATURE_MAX:
        raise InvalidInputError("Each value must be between 0 and 1 "
                                "(sonar energies are normalised).")
    return arr


class Predictor:
    """Loads every trained model once and serves predictions."""

    def __init__(self, models_dir=config.MODELS_DIR):
        with open(models_dir / "model_meta.json") as fh:
            self.meta = json.load(fh)
        self.best_key = self.meta["best_model"]
        self.models = {k: joblib.load(models_dir / v["file"])
                       for k, v in self.meta["models"].items()}

    def _frame(self, arr):
        return pd.DataFrame([arr], columns=config.FEATURE_NAMES)

    def _single(self, key, frame):
        model = self.models[key]
        classes = list(model.classes_)
        proba = model.predict_proba(frame)[0]
        p_mine = float(proba[classes.index("M")])
        label = "M" if p_mine >= 0.5 else "R"
        return {"model": key, "model_name": self.meta["models"][key]["name"],
                "label": label, "class_name": config.CLASS_NAMES[label],
                "probability_mine": p_mine, "probability_rock": 1 - p_mine,
                "confidence": max(p_mine, 1 - p_mine)}

    def predict(self, values, model: str | None = None) -> dict:
        arr = validate_features(values)
        frame = self._frame(arr)
        key = model or self.best_key
        if key not in self.models:
            raise InvalidInputError(f"Unknown model '{key}'. "
                                    f"Choose from {sorted(self.models)}.")
        primary = self._single(key, frame)
        votes = [self._single(k, frame) for k in self.models]
        tally = Counter(v["label"] for v in votes)
        primary["all_models"] = votes
        primary["consensus"] = {
            "mine_votes": tally.get("M", 0), "rock_votes": tally.get("R", 0),
            "agreement": max(tally.values()) / len(votes),
        }
        return primary

    def predict_many(self, rows, model: str | None = None) -> list[dict]:
        return [self.predict(r, model) for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Classify sonar readings as Rock or Mine")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--values", help="60 comma-separated numbers")
    src.add_argument("--csv", help="CSV file, one 60-value reading per row (no header)")
    parser.add_argument("--model", choices=["logistic_regression", "svm", "random_forest", "knn"])
    args = parser.parse_args()

    predictor = Predictor()
    if args.values:
        rows = [[float(v) for v in args.values.split(",")]]
    else:
        df = pd.read_csv(args.csv, header=None)
        rows = df.iloc[:, :config.N_FEATURES].values.tolist()
    for i, r in enumerate(predictor.predict_many(rows, args.model), 1):
        print(f"#{i:<3} {r['class_name']:<5} confidence {r['confidence']:.1%} "
              f"({r['model_name']}, {r['consensus']['mine_votes']}/4 models say mine)")


if __name__ == "__main__":
    main()
