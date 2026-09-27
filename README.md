# Support-ticket model service

A small, runnable classification service for synthetic support tickets. It has
an offline training/evaluation CLI, a FastAPI inference API, and two interchangeable
CPU-compatible backends:

- **sklearn (default):** normalized text -> TF-IDF unigram/bigram features ->
  logistic regression probabilities.
- **torch:** normalized text -> deterministic bag-of-words counts -> a small
  PyTorch MLP. Training and inference run on CPU; no GPU or network is needed.

The fixture in `data/tickets.csv` is synthetic and contains the labels
`account`, `billing`, `shipping`, and `technical`.

## Quickstart

From this directory:

```bash
uv sync
uv run ml-model-service train --backend sklearn
uv run ml-model-service evaluate --backend sklearn
uv run uvicorn ml_model_service.api:app --host 127.0.0.1 --port 8000
```

The API loads `artifacts/sklearn.joblib`. If that file is absent, the app
performs the same deterministic build once during startup so a fresh checkout is
still runnable; it never trains while handling a request. Prefer the explicit
`train` command for deployments and CI.

Use the alternate backend with:

```bash
MODEL_BACKEND=torch uv run ml-model-service train --backend torch
MODEL_BACKEND=torch uv run uvicorn ml_model_service.api:app --host 127.0.0.1 --port 8000
```

The Dockerfile trains the default sklearn artifact during the image build, and
`docker compose up --build` starts the service on port 8000.

## Architecture

`data.py` validates the CSV and `preprocessing.py` owns the shared Unicode,
case, and whitespace normalization. `protocol.py` defines the narrow
`ModelBackend` boundary and immutable `Prediction` value. Backend modules contain
only model-specific fit/load/save/forward logic. `training.py` owns splitting,
metrics, and artifact creation; `registry.py` loads one artifact during FastAPI
startup and reuses it for all requests. API schemas are Pydantic v2 models in
`schemas.py`.

Generated files in `artifacts/` are ignored by Git. Artifacts are intentionally
not shipped in the repository: rebuild them from the deterministic fixture with
one of the commands above.

## API

- `GET /v1/health/live` - process liveness (`{"status":"ok"}`).
- `GET /v1/health/ready` - model readiness; returns `503` with an error object
  if the startup artifact cannot be loaded.
- `GET /v1/model` - backend, model version, labels, algorithm, and artifact path.
- `POST /v1/predict` - validate and classify one ticket. Text is required and
  limited to 2,000 characters.

Example:

```bash
curl http://127.0.0.1:8000/v1/model
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"text":"My invoice has an unexpected charge"}'
```

A successful prediction looks like:

```json
{
  "label": "billing",
  "confidence": 0.81,
  "probabilities": {
    "account": 0.04,
    "billing": 0.81,
    "shipping": 0.06,
    "technical": 0.09
  },
  "backend": "sklearn",
  "model_version": "sklearn-tfidf-logreg-v1"
}
```

Invalid or blank input receives HTTP 422. Model startup failures receive HTTP
503. No endpoint accepts arbitrary model files or triggers request-time training.

## ML concepts and limitations

TF-IDF downweights common words and emphasizes terms that distinguish categories;
logistic regression maps those sparse features to calibrated-ish class scores.
The torch option demonstrates a neural forward path over a fixed count-vector
feature space, not a pretrained language model. Both models are trained on only
24 short synthetic examples. They are not production-quality language
understanding, are sensitive to vocabulary and spelling, are not guaranteed to
be calibrated, and should not be used for high-impact decisions. The fixture is
single-label, English-only, intentionally balanced, and does not represent real
support traffic. Retrain and evaluate on representative, consented data before
any serious use.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest --cov=ml_model_service --cov-report=term-missing
```

All tests are local and network-free. See `SECURITY.md` for the demo's security
posture and `CONTRIBUTING.md` for change guidelines.
