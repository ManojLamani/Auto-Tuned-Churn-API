"""Paths, seeds and search spaces shared by the training scripts."""
from pathlib import Path

from scipy.stats import loguniform, randint

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODELS_DIR = ROOT / "models"

SEED = 42
TEST_SIZE = 0.2
N_ITER = 20
INNER_FOLDS = 3
OUTER_FOLDS = 5

SEARCH_SPACES = {
    "Logistic Regression": {
        "model__C": loguniform(1e-3, 1e2),
        "model__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        "model__n_estimators": randint(100, 501),
        "model__max_depth": [None, 4, 6, 8, 10, 12, 16, 20],
        "model__min_samples_split": randint(2, 21),
        "model__min_samples_leaf": randint(1, 21),
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    },
}
