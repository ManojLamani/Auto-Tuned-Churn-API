"""CI quality gate: the saved artifact, metadata and experiment log are complete and consistent.

Run: python -m training.validate_artifact
"""
import json
import sys

import joblib
import pandas as pd

from training.config import MODELS_DIR

REQUIRED_METADATA = [
    "model_name", "model_version", "dataset_version", "training_timestamp", "feature_schema_version",
    "preprocessing_version", "hyperparameters", "search_config", "cv_config", "threshold", "risk_tiers",
    "metrics", "feature_schema", "feature_importance",
]
REQUIRED_METRICS = ["baseline_cv_f1", "nested_cv_f1_mean", "nested_cv_f1_std", "test", "test_baseline"]
REQUIRED_RUN_FIELDS = ["run_id", "model", "type", "dataset_version", "split_seed", "preprocessing_version",
                       "scoring", "cv_f1_mean", "artifact_version", "timestamp"]


def check():
    errors = []
    artifacts = sorted(MODELS_DIR.glob("*.joblib"))
    if len(artifacts) != 1:
        return [f"expected exactly one .joblib artifact, found {len(artifacts)}"]

    bundle = joblib.load(artifacts[0])
    meta = bundle["metadata"]
    saved = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    log = json.loads((MODELS_DIR / "experiment_log.json").read_text())

    errors += [f"metadata missing '{k}'" for k in REQUIRED_METADATA if k not in meta]
    errors += [f"metrics missing '{k}'" for k in REQUIRED_METRICS if k not in meta.get("metrics", {})]
    if meta != saved:
        errors.append("model_metadata.json does not match the metadata inside the artifact")
    if artifacts[0].stem != meta.get("model_version"):
        errors.append("artifact file name does not match model_version")
    if not 0 < meta.get("threshold", 0) < 1:
        errors.append("threshold must be between 0 and 1")
    if not {"inner", "outer"} <= set(meta.get("cv_config", {})):
        errors.append("nested CV configuration not recorded")

    types = {run.get("type") for run in log}
    if not {"baseline", "nested_cv_random_search", "final_random_search"} <= types:
        errors.append("experiment log is missing baseline, nested CV or final runs")
    for run in log:
        errors += [f"run {run.get('run_id')} missing '{k}'" for k in REQUIRED_RUN_FIELDS if k not in run]

    schema = meta["feature_schema"]
    row = {c: 0.0 for c in schema["numerical"]} | {c: v[0] for c, v in schema["categorical"].items()}
    proba = bundle["pipeline"].predict_proba(pd.DataFrame([row]))[0, 1]
    if not 0 <= proba <= 1:
        errors.append("artifact produced an invalid probability")
    return errors


if __name__ == "__main__":
    problems = check()
    for p in problems:
        print("FAIL:", p)
    print("Artifact validation", "failed" if problems else "passed")
    sys.exit(1 if problems else 0)
