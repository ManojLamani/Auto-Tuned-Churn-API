"""FastAPI inference service.

Run: uvicorn backend.app.main:app --reload --port 8000
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from backend.app.config import CORS_ORIGINS, MODELS_DIR
from backend.app.model_card import build_model_card
from backend.app.predictor import Predictor
from backend.app.schemas import Customer, Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("churn-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = Predictor(MODELS_DIR)
        log.info("Loaded model %s", app.state.predictor.version)
    except Exception as exc:  # the service still starts, but reports degraded health
        app.state.predictor = None
        log.error("Model failed to load: %s", exc)
    yield


app = FastAPI(title="Auto-Tuned Churn API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], expose_headers=["X-Request-ID"])


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Log request ID, route, status and latency. Customer inputs are never logged."""
    request.state.request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    log.info("id=%s %s %s status=%s latency_ms=%.1f", request.state.request_id, request.method,
             request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response


def error(request: Request, status: int, code: str, message: str, details=None):
    body = {"code": code, "message": message, "requestId": getattr(request.state, "request_id", None)}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status, content={"error": body})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if any(e["type"] == "json_invalid" for e in exc.errors()):
        return error(request, 400, "MALFORMED_REQUEST", "Request body is not valid JSON")
    details = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
    return error(request, 422, "VALIDATION_ERROR", "Invalid customer data", details)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 503: "MODEL_UNAVAILABLE"}
    return error(request, exc.status_code, codes.get(exc.status_code, "HTTP_ERROR"), str(exc.detail))


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    log.exception("Unexpected error id=%s", getattr(request.state, "request_id", None))
    return error(request, 500, "INTERNAL_ERROR", "Unexpected server error")


def get_predictor(request: Request) -> Predictor:
    if request.app.state.predictor is None:
        raise HTTPException(503, "Model is not loaded")
    return request.app.state.predictor


api = APIRouter(prefix="/api/v1")


@api.get("/health")
def health(request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        return JSONResponse(status_code=503, content={"status": "degraded", "modelLoaded": False})
    return {"status": "ok", "modelLoaded": True, "modelVersion": predictor.version}


@api.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request, predictor: Predictor = Depends(get_predictor)):
    probability, tier = predictor.predict(customer)
    return Prediction(
        predictionId=request.state.request_id,
        churnProbability=round(probability, 4),
        riskTier=tier,
        threshold=predictor.metadata["threshold"],
        modelVersion=predictor.version,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )


@api.get("/model")
def model(predictor: Predictor = Depends(get_predictor)):
    return predictor.metadata


@api.get("/experiments")
def experiments(predictor: Predictor = Depends(get_predictor)):
    return predictor.experiments


@api.get("/model-card")
def model_card(predictor: Predictor = Depends(get_predictor)):
    return build_model_card(predictor.metadata)


app.include_router(api)
