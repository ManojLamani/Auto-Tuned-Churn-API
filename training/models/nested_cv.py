"""Random Search inside nested cross-validation.

Run: python -m training.models.nested_cv
"""
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

from training.config import INNER_FOLDS, N_ITER, OUTER_FOLDS, SEARCH_SPACES, SEED
from training.data import load_data, split
from training.evaluation import CV_SCORING
from training.features import build_pipeline

ESTIMATORS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
    "Random Forest": lambda: RandomForestClassifier(random_state=SEED, n_jobs=1),
}


def make_search(name, X, n_iter=N_ITER, inner_folds=INNER_FOLDS):
    """Inner loop: Random Search over the candidate's search space."""
    return RandomizedSearchCV(
        build_pipeline(X, ESTIMATORS[name]()),
        param_distributions=SEARCH_SPACES[name],
        n_iter=n_iter,
        scoring="f1",
        cv=StratifiedKFold(inner_folds, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=-1,
    )


def run_nested_cv(X, y, names=tuple(ESTIMATORS), n_iter=N_ITER,
                  inner_folds=INNER_FOLDS, outer_folds=OUTER_FOLDS):
    """Outer loop: score the whole tuning procedure on folds the search never saw."""
    outer_cv = StratifiedKFold(outer_folds, shuffle=True, random_state=SEED)
    results = {}
    for name in names:
        start = time.perf_counter()
        scores = cross_validate(
            make_search(name, X, n_iter, inner_folds), X, y,
            cv=outer_cv, scoring=CV_SCORING, return_estimator=True,
        )
        results[name] = {
            "outer_fold_f1": scores["test_f1"].round(4).tolist(),
            "means": {m: round(float(scores[f"test_{m}"].mean()), 4) for m in CV_SCORING},
            "f1_std": round(float(np.std(scores["test_f1"])), 4),
            "fold_params": [est.best_params_ for est in scores["estimator"]],
            "searches": scores["estimator"],
            "runtime_sec": round(time.perf_counter() - start, 1),
        }
    return results


def main():
    X, y = load_data()
    X_train, _, y_train, _ = split(X, y)
    for name, r in run_nested_cv(X_train, y_train).items():
        print(f"{name}: F1 {r['means']['f1']} +/- {r['f1_std']}  folds {r['outer_fold_f1']}")


if __name__ == "__main__":
    main()
