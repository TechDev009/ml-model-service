from pathlib import Path

import numpy as np
import pytest

from ml_model_service.backends.sklearn_backend import train_sklearn_model
from ml_model_service.backends.torch_backend import train_torch_model
from ml_model_service.data import load_tickets

DATA = Path(__file__).parents[1] / "data" / "tickets.csv"


def _training_data() -> tuple[list[str], list[str]]:
    frame = load_tickets(DATA)
    return frame["text"].tolist(), frame["label"].tolist()


def test_sklearn_backend_forward_and_protocol() -> None:
    texts, labels = _training_data()
    backend = train_sklearn_model(texts, labels)
    probabilities = backend.forward(["my invoice has a billing charge"])
    prediction = backend.predict(["my invoice has a billing charge"])[0]
    assert probabilities.shape == (1, 4)
    assert np.isclose(probabilities.sum(), 1.0)
    assert prediction.label in backend.labels
    assert set(prediction.probabilities) == set(backend.labels)


@pytest.mark.parametrize("seed", [7, 42])
def test_torch_backend_forward_path_is_cpu_and_probabilities_sum(seed: int) -> None:
    pytest.importorskip("torch")
    texts, labels = _training_data()
    backend = train_torch_model(texts, labels, seed=seed, epochs=20)
    logits = backend.forward(["where is my package shipment"])
    probabilities = backend.predict_proba(["where is my package shipment"])
    assert logits.shape == (1, 4)
    assert probabilities.shape == (1, 4)
    assert np.isclose(probabilities.sum(), 1.0)
    assert backend.predict(["where is my package shipment"])[0].label in backend.labels


def test_torch_artifact_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    texts, labels = _training_data()
    backend = train_torch_model(texts, labels, seed=42, epochs=20)
    path = backend.save(tmp_path / "model.pt")
    from ml_model_service.backends.torch_backend import load_torch_backend

    loaded = load_torch_backend(path)
    assert loaded.metadata() == backend.metadata()
    assert loaded.predict(["i forgot my account password"])[0].label == "account"
