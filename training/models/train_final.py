"""Full training run: baselines, nested CV, final fit, threshold, one test evaluation, saved artifact.

Run: python -m training.models.train_final
"""
import time
from collections import Counter
from datetime import UTC, datetime

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from training.artifacts import new_version, save_release
from training.config import (
    DATA_PATH,
    INNER_FOLDS,
    MODELS_DIR,
    N_ITER,
    OUTER_FOLDS,
    PREPROCESSING_VERSION,
    SEARCH_SPACES,
    SEED,
    TEST_SIZE,
)
from training.data import dataset_version, load_data, split
from training.evaluation import (
    CV_SCORING,
    best_f1_threshold,
    holdout_metrics,
    risk_tier,
    risk_tiers,
)
from training.features import build_pipeline, feature_types
from training.models.nested_cv import make_search, run_nested_cv
from training.schema import Customer


def describe_space(space):
    return {
        k.removeprefix("model__"): (f"{v.dist.name}{tuple(round(a, 4) for a in v.args)}"
                                    if hasattr(v, "dist") else [str(x) for x in v])
        for k, v in space.items()
    }


def feature_importance(model, top=15):
    """Top features by importance (trees) or absolute coefficient (linear models)."""
    names = model.named_steps["preprocessing"].get_feature_names_out()
    step = model.named_steps["model"]
    values = step.feature_importances_ if hasattr(step, "feature_importances_") else abs(step.coef_[0])
    ranked = sorted(zip(names, values, strict=True), key=lambda x: -x[1])[:top]
    return [{"feature": n.split("__", 1)[1].replace("_", ": ", 1) if n.startswith("cat__") else n.split("__", 1)[1],
             "importance": round(float(v), 4)} for n, v in ranked]


def clean_params(params):
    return {k.removeprefix("model__"): (round(float(v), 6) if isinstance(v, float) else v)
            for k, v in params.items()}


def main():
    run_time = datetime.now(UTC).isoformat(timespec="seconds")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    common = {"dataset_version": dataset_version(), "split_seed": SEED,
              "test_size": TEST_SIZE, "scoring": "f1", "timestamp": run_time,
              "preprocessing_version": PREPROCESSING_VERSION, "schema_version": 2}
    log = []

    # 1. Baselines (training set CV only)
    baselines = {
        "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=SEED),
        "Logistic Regression (default)": LogisticRegression(max_iter=1000, random_state=SEED),
    }
    for i, (name, model) in enumerate(baselines.items(), start=1):
        started = time.perf_counter()
        s = cross_validate(build_pipeline(X_train, model), X_train, y_train, cv=cv5, scoring=CV_SCORING)
        log.append({"run_id": f"baseline-{i:02d}", "model": name, "type": "baseline",
                    "cv": "StratifiedKFold(5)", "inner_cv": None, "outer_cv": "StratifiedKFold(5)",
                    "hyperparameters": model.get_params(), "threshold": 0.5,
                    "fold_metrics": {m: s[f"test_{m}"].round(6).tolist() for m in CV_SCORING},
                    "runtime_sec": round(time.perf_counter() - started, 3),
                    "cv_metrics": {m: round(float(s[f"test_{m}"].mean()), 4) for m in CV_SCORING},
                    "cv_f1_mean": round(float(s["test_f1"].mean()), 4),
                    "cv_f1_std": round(float(s["test_f1"].std()), 4), **common})
        print(f"{name}: CV F1 {log[-1]['cv_f1_mean']}")

    # 2. Random Search inside nested CV for each candidate
    nested = run_nested_cv(X_train, y_train)
    for name, r in nested.items():
        log.append({"run_id": "nested-" + name.lower().replace(" ", "-"), "model": name,
                    "type": "nested_cv_random_search",
                    "search_space": describe_space(SEARCH_SPACES[name]), "n_iter": N_ITER,
                    "inner_cv": f"StratifiedKFold({INNER_FOLDS})",
                    "outer_cv": f"StratifiedKFold({OUTER_FOLDS})",
                    "outer_fold_f1": r["outer_fold_f1"], "cv_metrics": r["means"],
                    "cv_f1_mean": r["means"]["f1"], "cv_f1_std": r["f1_std"],
                    "fold_params": [clean_params(p) for p in r["fold_params"]],
                    "hyperparameters": [clean_params(p) for p in r["fold_params"]],
                    "fold_metrics": r["fold_metrics"], "threshold": 0.5,
                    "runtime_sec": r["runtime_sec"], **common})
        print(f"{name}: nested CV F1 {r['means']['f1']} +/- {r['f1_std']}")

    # 3. Champion = best nested CV F1, then tune it on the full training set
    champion = max(nested, key=lambda n: nested[n]["means"]["f1"])
    search = make_search(champion, X_train)
    start = time.perf_counter()
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Champion: {champion} {clean_params(search.best_params_)}")

    # 4. Threshold from out-of-fold predictions on the training set (never the test set)
    oof = cross_val_predict(model, X_train, y_train, cv=cv5, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, oof)
    tiers = risk_tiers(threshold)

    # 5. The single evaluation on the untouched test set
    baseline = build_pipeline(X_train, LogisticRegression(max_iter=1000, random_state=SEED))
    baseline.fit(X_train, y_train)
    baseline_test = holdout_metrics(y_test, baseline.predict_proba(X_test)[:, 1], 0.5)
    proba = model.predict_proba(X_test)[:, 1]
    test = holdout_metrics(y_test, proba, threshold)
    tier_counts = Counter(risk_tier(p, tiers) for p in proba)
    print(f"Test F1: baseline {baseline_test['f1']:.4f} -> tuned {test['f1']:.4f}")

    def rounded(d):
        return {k: (round(float(v), 4) if not isinstance(v, list) else v) for k, v in d.items()}
    log.append({"run_id": "final-fit", "model": champion, "type": "final_random_search",
                "search_space": describe_space(SEARCH_SPACES[champion]), "n_iter": N_ITER,
                "inner_cv": f"StratifiedKFold({INNER_FOLDS})",
                "best_params": clean_params(search.best_params_),
                "hyperparameters": clean_params(search.best_params_), "outer_cv": None,
                "fold_metrics": {"f1": [float(search.cv_results_[f"split{i}_test_score"][search.best_index_])
                                         for i in range(INNER_FOLDS)]},
                "search_trials": [{"params": clean_params(p), "mean_f1": round(float(score), 6)}
                                  for p, score in zip(search.cv_results_["params"],
                                                      search.cv_results_["mean_test_score"], strict=True)],
                "cv_f1_mean": round(float(search.best_score_), 4), "threshold": threshold,
                "test_metrics": rounded(test),
                "runtime_sec": round(time.perf_counter() - start, 1), **common})

    # 6. Save the versioned artifact
    numerical, categorical = feature_types(X_train)
    version = new_version(champion)
    for run in log:
        run["artifact_version"] = version
    metadata = {
        "model_name": champion,
        "model_version": version,
        "threshold_version": f"{version}:threshold",
        "feature_types": {name: spec["type"] for name, spec in Customer.model_json_schema()["properties"].items()},
        "dataset_id": DATA_PATH.name,
        "dataset_version": dataset_version(),
        "training_timestamp": run_time,
        "feature_schema_version": "v1",
        "preprocessing_version": PREPROCESSING_VERSION,
        "sklearn_version": sklearn.__version__,
        "hyperparameters": clean_params(search.best_params_),
        "search_config": {"method": "RandomizedSearchCV", "n_iter": N_ITER, "scoring": "f1", "seed": SEED},
        "cv_config": {"inner": f"StratifiedKFold({INNER_FOLDS})", "outer": f"StratifiedKFold({OUTER_FOLDS})"},
        "threshold": threshold,
        "risk_tiers": tiers,
        "feature_importance": feature_importance(model),
        "metrics": {
            "baseline_cv_f1": log[1]["cv_f1_mean"],
            "nested_cv_f1_mean": nested[champion]["means"]["f1"],
            "nested_cv_f1_std": nested[champion]["f1_std"],
            "test": rounded(test),
            "test_baseline": rounded(baseline_test),
            "test_risk_tiers": {t: tier_counts.get(t, 0) for t in ("HIGH", "MEDIUM", "LOW")},
            "test_class_counts": {"no_churn": int((y_test == 0).sum()), "churn": int(y_test.sum())},
        },
        "feature_schema": {
            "numerical": numerical,
            "categorical": {c: sorted(X_train[c].unique().tolist()) for c in categorical},
        },
    }
    destination = save_release(MODELS_DIR, model, metadata, log)
    print(f"Saved {version} (threshold {threshold}) to {destination}")
    return metadata, log



if __name__ == "__main__":
    main()
