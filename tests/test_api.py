"""API tests (PRD TEST-001 to 004, 006, 010 plus error contract)."""
import json

import pytest

from backend.app.config import MODELS_DIR
from backend.app.monitoring import RateLimiter
from backend.app.predictor import Predictor

PREDICT = "/api/v1/predict"


def test_001_valid_prediction(client, customer):
    r = client.post(PREDICT, json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churnProbability"] <= 1
    assert body["riskTier"] in {"HIGH", "MEDIUM", "LOW"}
    assert body["modelVersion"] == "churn-rf-v1"
    assert body["preprocessingVersion"] == "v1"
    assert body["thresholdVersion"] == "churn-rf-v1:threshold"
    assert r.headers["X-Request-ID"] == body["predictionId"]


def test_002_missing_feature(client, customer):
    del customer["Contract"]
    r = client.post(PREDICT, json=customer)
    assert r.status_code == 422
    assert r.json()["error"]["details"][0]["field"] == "Contract"


@pytest.mark.parametrize("field, value", [("tenure", "abc"), ("Contract", "Weekly"), ("MonthlyCharges", -5)])
def test_003_wrong_type_or_value(client, customer, field, value):
    r = client.post(PREDICT, json={**customer, field: value})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_004_model_unavailable(client, customer):
    saved = client.app.state.predictor
    client.app.state.predictor = None
    try:
        assert client.post(PREDICT, json=customer).status_code == 503
        health = client.get("/api/v1/health")
        assert health.status_code == 503 and health.json()["status"] == "degraded"
    finally:
        client.app.state.predictor = saved


def test_006_deterministic(client, customer):
    first = client.post(PREDICT, json=customer).json()
    second = client.post(PREDICT, json=customer).json()
    assert first["churnProbability"] == second["churnProbability"]
    assert first["riskTier"] == second["riskTier"]


def test_010_model_metadata(client):
    meta = client.get("/api/v1/model").json()
    saved = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    assert meta["model_version"] == saved["model_version"]
    assert meta["metrics"]["test"]["f1"] == saved["metrics"]["test"]["f1"]
    assert meta["threshold"] == saved["threshold"]


def test_experiments_and_model_card(client):
    runs = client.get("/api/v1/experiments").json()
    assert {r["type"] for r in runs} >= {"baseline", "nested_cv_random_search", "final_random_search"}
    card = client.get("/api/v1/model-card").json()
    assert card["modelDetails"]["version"] == "churn-rf-v1"
    assert card["limitations"]


def test_malformed_json_and_unknown_route(client):
    r = client.post(PREDICT, content="{bad json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "MALFORMED_REQUEST"
    assert client.get("/api/v1/unknown").status_code == 404


def test_startup_fails_without_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path)


def test_rate_limited(client, customer):
    saved = client.app.state.limiter
    client.app.state.limiter = RateLimiter(2)
    try:
        codes = [client.post(PREDICT, json=customer).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert client.post(PREDICT, json=customer).json()["error"]["code"] == "RATE_LIMITED"
    finally:
        client.app.state.limiter = saved


def test_metrics(client, customer):
    before = client.get("/api/v1/metrics").json()
    client.post(PREDICT, json=customer)
    client.get("/api/v1/unknown")
    after = client.get("/api/v1/metrics").json()
    assert after["predictions"] == before["predictions"] + 1
    assert after["errors"] >= before["errors"] + 1
    assert after["latencyMs"]["p95"] is not None
    assert after["modelVersion"] == "churn-rf-v1" and after["modelLoadMs"] > 0
    assert after["modelLoadFailures"] == 0
    assert after["requestsPerSecond"] >= 0
    assert after["modelVersionRequests"]["churn-rf-v1"] > 0


def test_unexpected_errors_are_structured_logged_and_counted(client, customer, monkeypatch):
    def fail(_customer):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(client.app.state.predictor, "predict", fail)
    before = client.get("/api/v1/metrics").json()["errors"]
    response = client.post(PREDICT, json=customer)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert response.headers["X-Request-ID"] == response.json()["error"]["requestId"]
    assert client.get("/api/v1/metrics").json()["errors"] == before + 1
