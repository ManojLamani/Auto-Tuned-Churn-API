"""Data science tests (PRD TEST-005, 007, 008, 009)."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from training.data import load_data, split
from training.evaluation import risk_tier, risk_tiers
from training.features import build_pipeline, feature_types
from training.models.nested_cv import make_search, run_nested_cv


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def small(data):
    X, y = data
    return X.sample(600, random_state=0), y.sample(600, random_state=0)


def test_005_threshold_boundary():
    tiers = risk_tiers(0.5)
    assert risk_tier(0.5, tiers) == "HIGH"
    assert risk_tier(0.4999, tiers) == "MEDIUM"
    assert risk_tier(0.25, tiers) == "MEDIUM"
    assert risk_tier(0.2499, tiers) == "LOW"


def test_007_reproducible_split_and_search(data, small):
    X, y = data
    assert split(X, y)[0].index.equals(split(X, y)[0].index)

    Xs, ys = small
    first = make_search("Logistic Regression", Xs, n_iter=3).fit(Xs, ys).best_params_
    second = make_search("Logistic Regression", Xs, n_iter=3).fit(Xs, ys).best_params_
    assert first == second


def test_008_preprocessing_fits_on_training_data_only(data):
    X, y = data
    X_train, _, y_train, _ = split(X, y)
    pipe = build_pipeline(X_train, LogisticRegression(max_iter=1000)).fit(X_train, y_train)

    numerical, _ = feature_types(X_train)
    scaler = pipe.named_steps["preprocessing"].named_transformers_["num"].named_steps["scaler"]
    assert scaler.n_samples_seen_ == len(X_train)
    assert np.allclose(scaler.mean_, X_train[numerical].mean())
    assert not np.allclose(scaler.mean_, X[numerical].mean())


def test_009_nested_cv_inner_tuning_isolated(small):
    X, y = small
    result = run_nested_cv(X, y, names=["Logistic Regression"], n_iter=2, inner_folds=3, outer_folds=3)
    r = result["Logistic Regression"]

    assert len(r["outer_fold_f1"]) == 3
    for search in r["searches"]:
        # Each inner search only ever saw its outer training fold (2/3 of the data), never the outer test fold.
        assert search.n_splits_ == 3
        assert search.best_estimator_.named_steps["model"].n_features_in_ > 0
        scaler = search.best_estimator_.named_steps["preprocessing"].named_transformers_["num"]["scaler"]
        assert scaler.n_samples_seen_ == pytest.approx(len(X) * 2 / 3, abs=1)
