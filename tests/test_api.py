import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Entering the context manager runs the lifespan handler (model load).
    with TestClient(app) as c:
        yield c


def test_root_reports_model(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["model_ready"] is True


def test_version_reports_service_and_model(client):
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "ai-inference-service"
    assert body["service_version"]
    assert body["model_version"] == "sentiment-tfidf-lr-v1"
    assert body["model_ready"] is True
    assert body["uptime_seconds"] >= 0


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_predict_positive(client):
    r = client.post("/predict", json={"text": "I love this product, it works perfectly"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positive"
    assert body["confidence"] > 0.5
    assert body["model_version"] == "sentiment-tfidf-lr-v1"


def test_predict_negative(client):
    r = client.post("/predict", json={"text": "Terrible quality, broke after one day"})
    assert r.status_code == 200
    assert r.json()["label"] == "negative"


def test_predict_rejects_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_rejects_whitespace_only_text(client):
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 422


def test_explain_rejects_whitespace_only_text(client):
    r = client.post("/predict/explain", json={"text": " \t "})
    assert r.status_code == 422


def test_predict_batch(client):
    r = client.post(
        "/predict/batch",
        json={"texts": ["Great quality, very happy", "Worst purchase ever"]},
    )
    assert r.status_code == 200
    labels = [item["label"] for item in r.json()["results"]]
    assert labels == ["positive", "negative"]


def test_predict_batch_rejects_empty_list(client):
    r = client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422


def test_predict_batch_rejects_whitespace_item(client):
    r = client.post("/predict/batch", json={"texts": ["Great quality", "   "]})
    assert r.status_code == 422


def test_predict_batch_rejects_too_many_items(client):
    r = client.post("/predict/batch", json={"texts": ["Great quality"] * 101})
    assert r.status_code == 422


def test_predict_rejects_oversized_text(client):
    r = client.post("/predict", json={"text": "great " * 400})
    assert r.status_code == 422


def test_metrics_exposes_prometheus_counters(client):
    client.post("/predict", json={"text": "Works like a charm"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text
    assert "predict_latency_seconds" in r.text


def test_explain_returns_top_tokens(client):
    r = client.post(
        "/predict/explain",
        json={"text": "I love this product, it works perfectly"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positive"
    assert body["top_tokens"], "expected at least one contributing token"
    assert all(t["contribution"] > 0 for t in body["top_tokens"])


def test_explain_negative_tokens(client):
    r = client.post(
        "/predict/explain",
        json={"text": "Terrible quality, broke after one day"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "negative"
    assert body["top_tokens"]


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "sentiment-tfidf-lr-v1"
    assert body["ready"] is True
    assert body["labels"] == ["negative", "positive"]
    assert body["train_samples"] > 0
    assert body["vocabulary_size"] > 0


def test_request_id_header_present(client):
    r = client.get("/health")
    assert r.headers.get("X-Request-ID")


def test_request_id_echoed_when_provided(client):
    r = client.get("/health", headers={"X-Request-ID": "test-123"})
    assert r.headers["X-Request-ID"] == "test-123"
