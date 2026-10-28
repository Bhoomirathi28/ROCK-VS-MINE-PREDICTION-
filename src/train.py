"""End-to-end training: clean -> split -> baseline -> tune -> CV -> compare -> save.

Run from the repository root:
    python -m src.train
"""
from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config, visualize
from src.data_preprocessing import run_preprocessing
from src.evaluate import evaluate_model, repeated_cv
from src.models import MODEL_LABELS, baseline_models, tuned_search_spaces

logger = logging.getLogger("train")


def _row(key, variant, metrics, extra=None):
    row = {"model": key, "model_name": MODEL_LABELS[key], "variant": variant}
    row.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
    row.update(extra or {})
    return row


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    np.random.seed(config.RANDOM_STATE)

    # 1. Data cleaning & splitting ------------------------------------------
    clean_df, (X_train, X_test, y_train, y_test), audit = run_preprocessing()
    logger.info("Train %s | Test %s", X_train.shape, X_test.shape)

    visualize.plot_class_distribution(clean_df)
    visualize.plot_mean_signature(clean_df)
    visualize.plot_correlation(clean_df)
    visualize.plot_feature_boxplots(clean_df)

    rows, baseline_results, tuned_results = [], {}, {}
    tuned_models, cv_scores, best_params = {}, {}, {}

    # 2a. Baselines (reproduce the notebook) ----------------------------------
    for key, model in baseline_models().items():
        model.fit(X_train, y_train)
        m = evaluate_model(model, X_train, y_train, X_test, y_test)
        baseline_results[key] = m
        rows.append(_row(key, "baseline", m))
        logger.info("baseline %-20s test acc %.3f", key, m["test_accuracy"])

    # 2b. Tuned pipelines ------------------------------------------------------
    inner_cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                               random_state=config.RANDOM_STATE)
    for key, (estimator, grid) in tuned_search_spaces().items():
        t0 = time.perf_counter()
        search = GridSearchCV(estimator, grid, cv=inner_cv, scoring="accuracy",
                              n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        best = search.best_estimator_
        tuned_models[key] = best
        best_params[key] = {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                            for k, v in search.best_params_.items()}

        scores = repeated_cv(best, X_train, y_train)
        cv_scores[key] = scores
        m = evaluate_model(best, X_train, y_train, X_test, y_test)
        tuned_results[key] = m
        rows.append(_row(key, "tuned", m, {
            "cv_mean": float(scores.mean()), "cv_std": float(scores.std()),
            "grid_search_seconds": fit_s, "best_params": json.dumps(best_params[key]),
        }))
        joblib.dump(best, config.MODELS_DIR / f"{key}.joblib")
        logger.info("tuned    %-20s cv %.3f±%.3f  test acc %.3f  %s",
                    key, scores.mean(), scores.std(), m["test_accuracy"], best_params[key])

    # 3. Compare & choose ------------------------------------------------------
    # Selection uses CV mean (not test accuracy) so the test set stays an
    # honest, unseen estimate. Ties are broken by recall on mines.
    best_key = max(tuned_results, key=lambda k: (round(cv_scores[k].mean(), 3),
                                                 tuned_results[k]["recall"]))
    joblib.dump(tuned_models[best_key], config.BEST_MODEL_PATH)

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(cv_scores).to_csv(config.METRICS_DIR / "cv_scores.csv", index=False)

    visualize.plot_confusion_matrices(baseline_results, "baseline")
    visualize.plot_confusion_matrices(tuned_results, "tuned")
    visualize.plot_roc(tuned_models, X_test, y_test)
    visualize.plot_metric_comparison(df_metrics)
    visualize.plot_baseline_vs_tuned(df_metrics)
    visualize.plot_cv_boxplot(cv_scores)
    visualize.plot_overfitting(df_metrics)
    visualize.plot_feature_importance(tuned_models["random_forest"])

    meta = {
        "best_model": best_key,
        "best_model_name": MODEL_LABELS[best_key],
        "selection_rule": "highest 10x5 CV accuracy on training set; tie-break on mine recall",
        "feature_names": config.FEATURE_NAMES,
        "classes": {"M": "Mine", "R": "Rock"},
        "models": {
            k: {
                "name": MODEL_LABELS[k],
                "file": f"{k}.joblib",
                "best_params": best_params[k],
                "cv_mean": float(cv_scores[k].mean()),
                "cv_std": float(cv_scores[k].std()),
                "baseline_test_accuracy": baseline_results[k]["test_accuracy"],
                **{mk: (float(mv) if not isinstance(mv, list) else mv)
                   for mk, mv in tuned_results[k].items()},
            } for k in tuned_results
        },
        "data": audit,
        "versions": {"python": platform.python_version(),
                     "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(config.MODEL_META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n" + df_metrics[["model_name", "variant", "train_accuracy", "test_accuracy",
                              "precision", "recall", "f1", "roc_auc", "cv_mean"]]
          .round(3).to_string(index=False))
    print(f"\nSelected for deployment: {MODEL_LABELS[best_key]}")
    return meta


if __name__ == "__main__":
    main()
