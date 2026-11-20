"""Containerized AI inference service.

A FastAPI app serving a small sentiment model, built to demonstrate the
operational surface of an AI service: health/readiness probes, structured
JSON errors, request metrics in Prometheus format, and graceful behaviour
when the model is not yet loaded.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from app.model import MODEL_VERSION, SentimentModel

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
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()
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


@app.get("/")
def root():
    return {
        "service": "ai-inference-service",
        "model": MODEL_VERSION,
        "model_ready": model.ready,
        "endpoints": ["/health", "/ready", "/predict", "/predict/batch", "/metrics"],
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


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
