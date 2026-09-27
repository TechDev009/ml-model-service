from pathlib import Path

from fastapi.testclient import TestClient

from ml_model_service.api import create_app
from ml_model_service.config import Settings

DATA = Path(__file__).parents[1] / "data" / "tickets.csv"


def test_health_model_and_prediction_endpoints(tmp_path: Path) -> None:
    settings = Settings(
        model_backend="sklearn",
        data_path=DATA,
        artifact_dir=tmp_path,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/health/live").json() == {"status": "ok"}
        ready = client.get("/v1/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready"}

        model = client.get("/v1/model")
        assert model.status_code == 200
        assert model.json()["backend"] == "sklearn"
        assert "billing" in model.json()["labels"]

        response = client.post(
            "/v1/predict", json={"text": "my invoice has an unexpected charge"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["label"] == "billing"
        assert payload["backend"] == "sklearn"
        assert set(payload["probabilities"]) == set(model.json()["labels"])


def test_prediction_validation_returns_422(tmp_path: Path) -> None:
    settings = Settings(data_path=DATA, artifact_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/predict", json={"text": "   "}).status_code == 422
        assert client.post("/v1/predict", json={}).status_code == 422
        assert client.post("/v1/predict", json={"text": "x" * 2_001}).status_code == 422
