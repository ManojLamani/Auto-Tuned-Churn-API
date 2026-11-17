"""Release contract, corruption, compatibility and rollback regressions."""
import copy
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.predictor import Predictor
from training.artifacts import activate, load_release, save_release, write_manifest
from training.config import MODELS_DIR


@pytest.fixture
def registry(tmp_path):
    for name in ("churn-rf-v1.joblib", "model_metadata.json", "experiment_log.json", "manifest.json"):
        shutil.copy2(MODELS_DIR / name, tmp_path / name)
    return tmp_path


def test_immutable_versions_and_rollback(registry, customer):
    bundle, runs = load_release(registry)
    with pytest.raises(FileExistsError):
        save_release(registry, bundle["pipeline"], bundle["metadata"], runs)
    meta = copy.deepcopy(bundle["metadata"])
    meta["model_version"] = "churn-rf-v2"
    for run in runs:
        run["artifact_version"] = meta["model_version"]
    save_release(registry, bundle["pipeline"], meta, runs)
    assert Predictor(registry).version == "churn-rf-v2"
    assert Predictor(registry, version="churn-rf-v1").version == "churn-rf-v1"
    with pytest.raises(FileExistsError):
        save_release(registry, bundle["pipeline"], meta, runs)
    activate(registry, "churn-rf-v1")
    assert Predictor(registry).version == "churn-rf-v1"
    assert (registry / "versions/churn-rf-v2/artifact.joblib").is_file()
    assert b"\r\n" not in (registry / "versions/churn-rf-v2/model_metadata.json").read_bytes()
    with pytest.raises(FileNotFoundError):
        activate(registry, "missing")
    assert Predictor(registry).version == "churn-rf-v1"


@pytest.mark.parametrize("name", ["churn-rf-v1.joblib", "model_metadata.json", "experiment_log.json"])
def test_corruption_rejected_before_deserialization(registry, name):
    with (registry / name).open("ab") as stream:
        stream.write(b"corrupt")
    with patch("training.artifacts.joblib.load") as deserialize:
        with pytest.raises(ValueError, match="checksum mismatch"):
            Predictor(registry)
        deserialize.assert_not_called()


@pytest.mark.parametrize("version", ["../outside", "/absolute", "..\\outside"])
def test_version_path_traversal_rejected(registry, version):
    with pytest.raises(ValueError, match="Invalid model version"):
        activate(registry, version)


def test_schema_categories_checked_at_startup(registry):
    bundle, runs = load_release(registry)
    bundle["metadata"]["model_version"] = "bad-categories"
    bundle["metadata"]["feature_schema"]["categorical"]["Contract"] = ["Weekly"]
    for run in runs:
        run["artifact_version"] = "bad-categories"
    with pytest.raises(ValueError, match="categories"):
        save_release(registry, bundle["pipeline"], bundle["metadata"], runs)
    assert Predictor(registry).version == "churn-rf-v1"


def test_missing_metrics_rejected(registry):
    meta = json.loads((registry / "model_metadata.json").read_text())
    del meta["metrics"]["test"]
    (registry / "model_metadata.json").write_text(json.dumps(meta))
    write_manifest(registry, "churn-rf-v1.joblib")
    with pytest.raises(ValueError, match="Missing metric"):
        load_release(registry)


def test_notebook_uses_canonical_training_export():
    notebook = json.loads(Path("notebooks/01_dataset_preparation.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "from training.models.train_final import main" in source
    assert "joblib.dump" not in source


def test_canonical_training_exports_complete_serving_contract(tmp_path, monkeypatch):
    from training.data import load_data
    from training.models import train_final
    from training.models.nested_cv import make_search, run_nested_cv

    features, labels = load_data()
    indices = features.sample(600, random_state=0).index
    monkeypatch.setattr(train_final, "load_data", lambda: (features.loc[indices], labels.loc[indices]))
    monkeypatch.setattr(train_final, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_final, "N_ITER", 2)
    monkeypatch.setattr(train_final, "OUTER_FOLDS", 3)
    monkeypatch.setattr(train_final, "make_search", lambda name, x: make_search(name, x, n_iter=2))
    monkeypatch.setattr(train_final, "run_nested_cv", lambda x, y: run_nested_cv(
        x, y, names=["Logistic Regression"], n_iter=2, outer_folds=3))
    metadata, runs = train_final.main()
    predictor = Predictor(tmp_path)
    assert predictor.version == metadata["model_version"]
    assert predictor.threshold_version == metadata["threshold_version"]
    assert all(run["schema_version"] == 2 and run["fold_metrics"] for run in runs)
    assert all(run["runtime_sec"] >= 0 for run in runs)
    assert predictor.metadata["metrics"]["test"]["f1"] >= 0
