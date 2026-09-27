"""The model protocol shared by all inference backends."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Prediction:
    """One model prediction with probabilities in label order."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class ModelBackend(Protocol):
    """Inference boundary implemented by sklearn and PyTorch backends."""

    backend_name: str
    model_version: str
    labels: tuple[str, ...]

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Predict one result for every input text."""
        ...

    def predict_proba(self, texts: Sequence[str]) -> object:
        """Return a probability matrix for a sequence of texts."""
        ...

    def metadata(self) -> dict[str, object]:
        """Return safe, JSON-serializable model metadata."""
        ...


def predict_one(backend: ModelBackend, text: str) -> Prediction:
    """Convenience helper that makes the single-ticket API explicit."""

    predictions = backend.predict([text])
    if len(predictions) != 1:  # pragma: no cover - protects a backend contract
        raise RuntimeError("model backend returned an unexpected number of results")
    return predictions[0]
