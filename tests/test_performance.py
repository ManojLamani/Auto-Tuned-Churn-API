"""Performance checks from the PRD: model load time, cold start, warm p50/p95 and a small concurrent test."""
import time
from concurrent.futures import ThreadPoolExecutor

from backend.app.config import MODELS_DIR
from backend.app.predictor import Predictor
from backend.app.schemas import Customer

PREDICT = "/api/v1/predict"


def test_model_load_and_cold_start(customer):
    start = time.perf_counter()
    predictor = Predictor(MODELS_DIR)
    load_s = time.perf_counter() - start

    start = time.perf_counter()
    predictor.predict(Customer(**customer))
    cold_s = time.perf_counter() - start

    print(f"\nmodel load {load_s * 1000:.0f} ms, cold prediction {cold_s * 1000:.0f} ms")
    assert load_s < 10
    assert cold_s < 2


def test_warm_latency(client, customer):
    client.post(PREDICT, json=customer)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        assert client.post(PREDICT, json=customer).status_code == 200
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    p50, p95 = times[24], times[47]
    print(f"\nwarm p50 {p50:.0f} ms, p95 {p95:.0f} ms")
    assert p95 < 1000


def test_concurrent_requests(client, customer):
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(lambda _: client.post(PREDICT, json=customer).status_code, range(40)))
    assert codes == [200] * 40
