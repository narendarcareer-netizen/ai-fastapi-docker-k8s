"""Containerized AI inference service.

A FastAPI app serving a small sentiment model, built to demonstrate the
operational surface of an AI service: health/readiness probes, structured
JSON errors, request metrics in Prometheus format, and graceful behaviour
when the model is not yet loaded.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from app.model import MODEL_VERSION, SentimentModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("inference")

model = SentimentModel()

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds", "Latency of /predict in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load()
    yield


app = FastAPI(
    title="ai-inference-service",
    description="Containerized sentiment-classification inference service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id, count the request, and log one line per call."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()
    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%s",
        request_id, request.method, request.url.path,
        response.status_code, latency_ms,
    )
    return response


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to classify")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class ExplainResponse(BaseModel):
    label: str
    confidence: float
    top_tokens: list[dict]
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    ready: bool
    labels: list[str]
    train_samples: int
    train_seconds: float
    vocabulary_size: int
    classifier: str


@app.get("/")
def root():
    return {
        "service": "ai-inference-service",
        "model": MODEL_VERSION,
        "model_ready": model.ready,
        "endpoints": ["/health", "/ready", "/model/info", "/predict", "/predict/batch", "/predict/explain", "/metrics"],
    }


@app.get("/health")
def health():
    """Liveness probe: the process is up. Does not check the model."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe: the model is loaded and predictions will succeed."""
    if not model.ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not model.ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    started = time.perf_counter()
    label, confidence = model.predict(req.text)
    latency = time.perf_counter() - started
    PREDICT_LATENCY.observe(latency)
    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        model_version=MODEL_VERSION,
        latency_ms=round(latency * 1000, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    if not model.ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    results = []
    for text in req.texts:
        if not text.strip():
            raise HTTPException(status_code=422, detail="batch items must not be empty")
        started = time.perf_counter()
        label, confidence = model.predict(text)
        latency = time.perf_counter() - started
        PREDICT_LATENCY.observe(latency)
        results.append(
            PredictResponse(
                label=label,
                confidence=round(confidence, 4),
                model_version=MODEL_VERSION,
                latency_ms=round(latency * 1000, 2),
            )
        )
    return BatchPredictResponse(results=results)


@app.post("/predict/explain", response_model=ExplainResponse)
def predict_explain(req: PredictRequest):
    """Like /predict, plus the tokens that drove the decision."""
    if not model.ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    label, confidence, tokens = model.explain(req.text)
    return ExplainResponse(
        label=label,
        confidence=round(confidence, 4),
        top_tokens=tokens,
        model_version=MODEL_VERSION,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Model metadata: version, labels, training stats, vocabulary size.

    Useful for debugging which model a pod is actually serving and for
    smoke checks after a rollout — no prediction required.
    """
    return ModelInfoResponse(**model.info())


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
