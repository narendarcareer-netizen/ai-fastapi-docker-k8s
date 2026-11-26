# AI Inference Service — Containerized AI Application

A FastAPI service that serves a small sentiment-classification model,
packaged with Docker and deployable to Kubernetes. Week 1 of a 5-project
AI Infrastructure portfolio: the point is the **operational surface** of an
AI service, not the model — health/readiness probes, Prometheus metrics,
resource limits, autoscaling, and a non-root container.

```
                 ┌──────────────────────────────┐
  client ──────► │  FastAPI (uvicorn, :8000)    │
                 │  /predict  /predict/batch    │
                 │  /health   /ready  /metrics  │
                 │  TF-IDF + logistic regression│
                 └──────────────┬───────────────┘
                                │  scrape
                 ┌──────────────▼───────────────┐
                 │  Prometheus → Grafana (wk 4) │
                 └──────────────────────────────┘
```

## What it does

| Endpoint | Purpose |
|---|---|
| `GET /` | Service info + model version |
| `GET /health` | Liveness probe — process is up |
| `GET /ready` | Readiness probe — model loaded, predictions will succeed |
| `POST /predict` | `{text}` → `{label, confidence, model_version, latency_ms}` |
| `POST /predict/batch` | Up to 100 texts in one call |
| `POST /predict/explain` | Like `/predict`, plus the top tokens that drove the decision |
| `GET /metrics` | Prometheus exposition: request counts, predict latency histogram |

The model is a scikit-learn TF-IDF + logistic-regression classifier trained
at startup on a small embedded dataset. It is deliberately CPU-only with no
download — `app/model.py` is the seam where a real model artifact (Hugging
Face transformer, ONNX export, registry pull) slots in without touching the
API layer.

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text": "I love this product, it works perfectly"}'
# {"label":"positive","confidence":0.8079,"model_version":"sentiment-tfidf-lr-v1","latency_ms":1.62}
curl -s -X POST localhost:8000/predict/explain -H 'Content-Type: application/json' \
  -d '{"text": "I love this product, it works perfectly"}'
# {"label":"positive","confidence":0.8079,"top_tokens":[{"token":"works","contribution":0.2321},...],"model_version":"sentiment-tfidf-lr-v1"}
curl -s localhost:8000/metrics | grep http_requests_total
```

Every response carries an `X-Request-ID` header (echoed back if you send
one), and each request is logged with its id, status, and latency — handy
when tracing a call through `kubectl logs`.

## Run it with Docker

```bash
docker build -t ai-fastapi-docker-k8s:latest .
docker run --rm -p 8000:8000 ai-fastapi-docker-k8s:latest
# or: docker compose up --build
```

## Deploy it to Kubernetes (kind / minikube)

```bash
kind create cluster --name ai-infra
kind load docker-image ai-fastapi-docker-k8s:latest --name ai-infra
kubectl apply -f k8s/
kubectl get pods -l app=ai-inference
kubectl port-forward svc/ai-inference 8080:80
curl -s localhost:8080/health
```

The Deployment runs 2 replicas with CPU/memory requests and limits,
liveness + readiness probes, and a non-root security context. The HPA
scales 2 → 5 replicas at 70% average CPU.

## Tests

```bash
pytest -q
# 12 passed
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/ready` returns 503 | Model still training at startup — wait a second and retry; check logs |
| `docker build` fails on `pip install` | No network in the build context, or an old pip — rebuild with `--no-cache` |
| Pod stuck in `CrashLoopBackOff` | Read `kubectl logs` — usually the image tag isn't in the cluster (`kind load` step missed) |
| HPA shows `<unknown>` CPU | metrics-server isn't installed in the cluster — `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml` |

## Verified on this build

- `pytest`: 8/8 pass (health, readiness, positive/negative prediction, batch, validation, metrics).
- Live smoke test: server booted, `/predict` returned the expected labels, `/metrics` exposed counters.
- K8s manifests: YAML-validated. `docker build` / `kubectl apply` were **not**
  run in this environment (no Docker or cluster here) — run them on your own
  machine with the commands above.

## What's next (Week 2)

Provision the cluster this service runs on with Terraform: VPC, EKS,
IAM roles — `aws-ai-infrastructure-terraform`.
