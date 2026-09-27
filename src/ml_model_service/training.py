"""Offline training, evaluation, and artifact persistence."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .backends.sklearn_backend import (
    SklearnBackend,
    load_sklearn_backend,
    train_sklearn_model,
)
from .backends.torch_backend import (
    TorchTextBackend,
    load_torch_backend,
    train_torch_model,
)
from .data import load_tickets
from .protocol import ModelBackend

BackendName = Literal["sklearn", "torch"]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Stable metrics emitted by the offline evaluation command."""

    accuracy: float
    macro_f1: float

    def as_dict(self) -> dict[str, float]:
        return {"accuracy": self.accuracy, "macro_f1": self.macro_f1}


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Result of training and saving one artifact."""

    backend: BackendName
    artifact_path: Path
    metrics: EvaluationMetrics
    labels: tuple[str, ...]


def evaluate_backend(
    backend: ModelBackend, texts: Sequence[str], labels: Sequence[str]
) -> EvaluationMetrics:
    """Evaluate an already-fitted backend without changing its state."""

    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty and the same length")
    predictions = backend.predict(texts)
    predicted_labels = [prediction.label for prediction in predictions]
    return EvaluationMetrics(
        accuracy=float(accuracy_score(labels, predicted_labels)),
        macro_f1=float(
            f1_score(
                labels,
                predicted_labels,
                average="macro",
                zero_division=cast(Any, 0),
            )
        ),
    )


def train_backend(
    backend: BackendName, texts: Sequence[str], labels: Sequence[str], seed: int = 42
) -> ModelBackend:
    """Fit the selected backend behind the shared model protocol."""

    if backend == "sklearn":
        return train_sklearn_model(texts, labels, seed=seed)
    if backend == "torch":
        return train_torch_model(texts, labels, seed=seed)
    raise ValueError(f"unsupported backend: {backend}")


def load_backend(backend: BackendName, artifact_path: str | Path) -> ModelBackend:
    """Load an artifact using an explicit backend choice."""

    if backend == "sklearn":
        return load_sklearn_backend(artifact_path)
    if backend == "torch":
        return load_torch_backend(artifact_path)
    raise ValueError(f"unsupported backend: {backend}")


def train_and_save(
    backend: BackendName,
    data_path: str | Path,
    artifact_path: str | Path,
    seed: int = 42,
) -> TrainingResult:
    """Evaluate a split, fit on all data, and save a serving artifact.

    This function is intentionally separate from the request path.  The API may
    call it once during a missing-artifact startup bootstrap, never per request.
    """

    frame = load_tickets(data_path)
    texts = [str(value) for value in frame["text"].tolist()]
    labels = [str(value) for value in frame["label"].tolist()]
    train_texts_raw, test_texts_raw, train_labels_raw, test_labels_raw = (
        train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=seed,
            stratify=labels,
        )
    )
    train_texts = [str(value) for value in train_texts_raw]
    test_texts = [str(value) for value in test_texts_raw]
    train_labels = [str(value) for value in train_labels_raw]
    test_labels = [str(value) for value in test_labels_raw]
    evaluated_model = train_backend(backend, train_texts, train_labels, seed=seed)
    metrics = evaluate_backend(evaluated_model, test_texts, test_labels)
    serving_model = train_backend(backend, texts, labels, seed=seed)
    output_path = Path(artifact_path)
    if isinstance(serving_model, (SklearnBackend, TorchTextBackend)):
        serving_model.save(output_path)
    else:
        raise TypeError("backend does not support artifact saving")
    return TrainingResult(
        backend=backend,
        artifact_path=output_path,
        metrics=metrics,
        labels=serving_model.labels,
    )


def evaluate_artifact(
    backend: BackendName, data_path: str | Path, artifact_path: str | Path
) -> EvaluationMetrics:
    """Evaluate a saved artifact against the complete fixture."""

    model = load_backend(backend, artifact_path)
    frame = load_tickets(data_path)
    texts = [str(value) for value in frame["text"].tolist()]
    labels = [str(value) for value in frame["label"].tolist()]
    return evaluate_backend(model, texts, labels)


__all__ = [
    "BackendName",
    "EvaluationMetrics",
    "TrainingResult",
    "evaluate_artifact",
    "evaluate_backend",
    "load_backend",
    "train_and_save",
    "train_backend",
    "SklearnBackend",
    "TorchTextBackend",
]
