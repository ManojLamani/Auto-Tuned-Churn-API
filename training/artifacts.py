"""Immutable bundles and atomic activation. Trust manifests like executable code.

Checksums detect corruption, not an attacker able to replace the manifest as well.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import joblib
import pandas as pd

from training.schema import Customer

VERSION = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,119}\Z")
REQUIRED_METADATA = {
    "model_name", "model_version", "dataset_version", "training_timestamp", "feature_schema_version",
    "preprocessing_version", "hyperparameters", "search_config", "cv_config", "threshold", "risk_tiers",
    "metrics", "feature_schema", "feature_importance",
}
RUN_FIELDS = {"run_id", "model", "type", "dataset_version", "split_seed", "preprocessing_version",
              "scoring", "cv_f1_mean", "artifact_version", "timestamp"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def version_name(value):
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise ValueError("Invalid model version")
    return value


def new_version(model_name):
    prefix = "lr" if model_name == "Logistic Regression" else "rf"
    return f"churn-{prefix}-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"


def validate_metadata(meta, runs):
    missing = REQUIRED_METADATA - meta.keys()
    if missing:
        raise ValueError(f"Incomplete artifact metadata: {sorted(missing)}")
    version_name(meta["model_version"])
    schema = meta["feature_schema"]
    contract = Customer.model_json_schema()["properties"]
    if set(schema["numerical"]) | set(schema["categorical"]) != set(contract):
        raise ValueError("Artifact feature names do not match customer schema")
    for name, values in schema["categorical"].items():
        if set(values) != set(get_args(Customer.model_fields[name].annotation)):
            raise ValueError(f"Artifact categories do not match customer schema: {name}")
    for name in schema["numerical"]:
        if contract[name]["type"] not in {"integer", "number"}:
            raise ValueError(f"Artifact numerical type mismatch: {name}")
    types = {name: spec["type"] for name, spec in contract.items()}
    if meta.get("feature_types", types) != types:
        raise ValueError("Artifact types do not match customer schema")
    if not math.isfinite(meta["threshold"]) or not 0 < meta["threshold"] < 1:
        raise ValueError("Invalid operating threshold")
    tiers = meta["risk_tiers"]
    if not 0 == tiers["LOW"] < tiers["MEDIUM"] < tiers["HIGH"] == meta["threshold"]:
        raise ValueError("Risk tiers do not match operating threshold")
    if not {"inner", "outer"} <= meta["cv_config"].keys():
        raise ValueError("Missing nested CV configuration")
    metrics = meta["metrics"]
    for key in ("baseline_cv_f1", "nested_cv_f1_mean", "nested_cv_f1_std", "test", "test_baseline",
                "test_risk_tiers", "test_class_counts"):
        if key not in metrics:
            raise ValueError(f"Missing metric {key}")
    for name in ("test", "test_baseline"):
        for key in ("f1", "precision", "recall", "roc_auc", "pr_auc"):
            value = metrics[name][key]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {name} {key}")
        cm = metrics[name]["confusion_matrix"]
        if len(cm) != 2 or any(len(row) != 2 or any(v < 0 for v in row) for row in cm):
            raise ValueError("Invalid confusion matrix")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Missing experiment log")
    if not {"baseline", "nested_cv_random_search", "final_random_search"} <= {r.get("type") for r in runs}:
        raise ValueError("Incomplete experiment types")
    if any(run.get("schema_version", 1) >= 2 for run in runs):
        if not meta.get("threshold_version") or not meta.get("feature_types"):
            raise ValueError("Version 2 release requires threshold version and feature types")
    for run in runs:
        if RUN_FIELDS - run.keys() or run["artifact_version"] != meta["model_version"]:
            raise ValueError("Incomplete or mismatched experiment record")
        if run.get("schema_version", 1) >= 2:
            if {"hyperparameters", "fold_metrics", "runtime_sec", "threshold", "inner_cv", "outer_cv"} - run.keys():
                raise ValueError("Incomplete version 2 experiment record")
            if not math.isfinite(run["runtime_sec"]) or run["runtime_sec"] < 0:
                raise ValueError("Invalid experiment runtime")
            if not run["fold_metrics"]:
                raise ValueError("Missing fold metrics")
            for values in run["fold_metrics"].values():
                if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                    raise ValueError("Invalid fold metrics")


def write_manifest(directory, artifact_name):
    directory = Path(directory)
    files = [artifact_name, "model_metadata.json", "experiment_log.json"]
    manifest = {"schema_version": 1, "artifact": artifact_name,
                "files": {name: digest(directory / name) for name in files}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_directory(directory):
    """Verify bytes before deserialization and reject unlisted files/path traversal."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    artifact = manifest["artifact"]
    if not isinstance(artifact, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.joblib", artifact):
        raise ValueError("Invalid manifest artifact path")
    expected = {artifact, "model_metadata.json", "experiment_log.json"}
    if set(manifest["files"]) != expected:
        raise ValueError("Incomplete artifact manifest")
    for name, checksum in manifest["files"].items():
        if digest(directory / name) != checksum:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    meta = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    runs = json.loads((directory / "experiment_log.json").read_text(encoding="utf-8"))
    validate_metadata(meta, runs)
    bundle = joblib.load(directory / artifact)
    if bundle["metadata"] != meta:
        raise ValueError("Bundle and JSON metadata do not match")
    schema = meta["feature_schema"]
    pipeline = bundle["pipeline"]
    encoded = pipeline.named_steps["preprocessing"].named_transformers_["cat"].named_steps["onehot"]
    columns = next(cols for name, _, cols in pipeline.named_steps["preprocessing"].transformers_ if name == "cat")
    if any(set(values) != set(schema["categorical"][name])
           for name, values in zip(columns, encoded.categories_, strict=True)):
        raise ValueError("Fitted pipeline categories do not match metadata")
    expected_features = set(schema["numerical"]) | set(schema["categorical"])
    if set(pipeline.feature_names_in_) != expected_features:
        raise ValueError("Pipeline feature names do not match metadata")
    row = {c: 0.0 for c in schema["numerical"]} | {c: v[0] for c, v in schema["categorical"].items()}
    probability = float(pipeline.predict_proba(pd.DataFrame([row]))[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Artifact produces invalid probabilities")
    return bundle, runs


def selected_directory(root, version=None):
    root = Path(root)
    if version is None and (root / "active.json").exists():
        version = version_name(json.loads((root / "active.json").read_text(encoding="utf-8"))["version"])
    if version is not None:
        version_name(version)
        directory = root / "versions" / version
        if directory.is_dir():
            return directory
        if (root / "manifest.json").exists():
            legacy = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
            if legacy["model_version"] == version:
                return root
        raise FileNotFoundError(f"Model version not found: {version}")
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"No verified model release in {root}")
    return root


def load_release(root, version=None):
    directory = selected_directory(root, version)
    bundle, runs = load_directory(directory)
    if version is not None and bundle["metadata"]["model_version"] != version:
        raise ValueError("Requested model version does not match bundle")
    if directory.parent.name == "versions" and bundle["metadata"]["model_version"] != directory.name:
        raise ValueError("Model directory does not match bundle version")
    return bundle, runs


def activate(root, version):
    """Validate first, then replace one pointer. Restart serving workers afterwards."""
    root = Path(root)
    load_release(root, version)
    temp = root / f".active-{uuid.uuid4().hex}.json"
    try:
        temp.write_text(json.dumps({"version": version}) + "\n", encoding="utf-8", newline="\n")
        temp.replace(root / "active.json")
    finally:
        temp.unlink(missing_ok=True)


def save_release(root, pipeline, metadata, runs):
    root = Path(root)
    version = version_name(metadata["model_version"])
    validate_metadata(metadata, runs)
    if (root / "manifest.json").exists():
        legacy = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
        if legacy["model_version"] == version:
            raise FileExistsError(f"Model version already exists: {version}")
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    target = versions / version
    if target.exists():
        raise FileExistsError(f"Model version already exists: {version}")
    staging = versions / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        joblib.dump({"pipeline": pipeline, "metadata": metadata}, staging / "artifact.joblib", compress=3)
        for name, value in (("model_metadata.json", metadata), ("experiment_log.json", runs)):
            (staging / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8", newline="\n")
        write_manifest(staging, "artifact.joblib")
        load_directory(staging)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    activate(root, version)
    return target


if __name__ == "__main__":
    from training.config import MODELS_DIR

    parser = argparse.ArgumentParser(description="Activate or roll back to a verified model version")
    parser.add_argument("version")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    activate(args.models_dir, args.version)
    print(f"Activated {args.version}; restart API workers to load it")
