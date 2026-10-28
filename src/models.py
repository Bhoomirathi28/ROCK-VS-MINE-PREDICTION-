"""Model definitions.

Each model has two versions:

* baseline - exactly the configuration used in the original Colab notebook,
             so results can be reproduced and compared.
* tuned    - a Pipeline (scaler + estimator) searched with GridSearchCV.

Distance- and margin-based models (LR, SVM, KNN) need standardised inputs;
Random Forest is scale-invariant, so it gets no scaler.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import RANDOM_STATE

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "svm": "Support Vector Machine",
    "random_forest": "Random Forest",
    "knn": "K-Nearest Neighbours",
}


def baseline_models() -> dict:
    """Notebook configurations (LR was trained on unscaled data there)."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "svm": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", probability=True,
                                     random_state=RANDOM_STATE))]),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "knn": Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=5, algorithm="ball_tree"))]),
    }


def tuned_search_spaces() -> dict:
    """(estimator, param_grid) pairs for GridSearchCV."""
    return {
        "logistic_regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000))]),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "svm": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
            {"clf__kernel": ["rbf"],
             "clf__C": [0.5, 1, 5, 10, 50, 100],
             "clf__gamma": ["scale", 0.05, 0.01, 0.005]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [200, 500],
             "max_depth": [None, 8],
             "max_features": ["sqrt", "log2"],
             "min_samples_leaf": [1, 2]},
        ),
        "knn": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", KNeighborsClassifier(algorithm="ball_tree"))]),
            {"clf__n_neighbors": [1, 3, 5, 7, 9],
             "clf__weights": ["uniform", "distance"],
             "clf__p": [1, 2]},
        ),
    }

# Note: KNN uses algorithm="ball_tree" (exact neighbours, identical results to
# "auto"). It avoids a scikit-learn code path that fails on string labels when
# predict_proba is called on a single sample, which is exactly what the web
# app does.
