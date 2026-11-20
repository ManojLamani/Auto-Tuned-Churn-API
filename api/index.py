"""Vercel serverless entrypoint for the FastAPI service."""
import time

from backend.app.config import MODELS_DIR, RATE_LIMIT_PER_MINUTE
from backend.app.main import app, log
from backend.app.monitoring import Metrics, RateLimiter
from backend.app.predictor import Predictor

# Serverless runtimes may skip ASGI lifespan events, so load the model at import time.
app.state.metrics = Metrics()
app.state.limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)
_start = time.perf_counter()
try:
    app.state.predictor = Predictor(MODELS_DIR)
    app.state.metrics.model_load_ms = round((time.perf_counter() - _start) * 1000, 1)
except Exception as exc:
    app.state.predictor = None
    app.state.metrics.model_load_failures += 1
    log.error("Model failed to load: %s", exc)
