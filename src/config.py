"""Central configuration: paths, constants and reproducibility settings.

Keeping every constant here means the notebook, training script, tests and
web app all agree on the same split, seed and file locations.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Data
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "sonar_data.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

# Artefacts
MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_META_PATH = MODELS_DIR / "model_meta.json"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Dataset schema
N_FEATURES = 60
FEATURE_NAMES = [f"band_{i:02d}" for i in range(1, N_FEATURES + 1)]
TARGET = "label"
CLASSES = ("M", "R")
CLASS_NAMES = {"M": "Mine", "R": "Rock"}
POSITIVE_LABEL = "M"          # a missed mine is the costly error
FEATURE_MIN, FEATURE_MAX = 0.0, 1.0

# Reproducibility (same values as the original Colab notebook)
RANDOM_STATE = 1
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 10
