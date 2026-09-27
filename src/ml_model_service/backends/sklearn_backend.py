"""TF-IDF plus logistic-regression inference backend."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..preprocessing import preprocess_texts
from ..protocol import Prediction

MODEL_VERSION = "sklearn-tfidf-logreg-v1"


def build_sklearn_pipeline(seed: int = 42) -> Pipeline:
    """Build the deterministic sklearn pipeline used by training and serving."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    max_iter=1_000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


class SklearnBackend:
    """Serve a fitted sklearn text pipeline through the model protocol."""

    backend_name = "sklearn"

    def __init__(
        self,
        pipeline: Pipeline,
        labels: Sequence[str],
        model_version: str = MODEL_VERSION,
    ) -> None:
        self.pipeline = pipeline
        self.labels = tuple(labels)
        self.model_version = model_version
        if not self.labels:
            raise ValueError("a model must contain at least one label")

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "SklearnBackend":
        pipeline = artifact.get("pipeline")
        labels = artifact.get("labels")
        if not isinstance(pipeline, Pipeline) or not isinstance(labels, (list, tuple)):
            raise ValueError("invalid sklearn artifact")
        version = artifact.get("model_version", MODEL_VERSION)
        return cls(pipeline, [str(label) for label in labels], str(version))

    def _prepare(self, texts: Sequence[str]) -> list[str]:
        prepared = preprocess_texts(texts)
        if not prepared:
            raise ValueError("at least one text is required")
        if any(not text for text in prepared):
            raise ValueError("text must not be blank")
        return prepared

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return class probabilities in the fitted classifier's class order."""

        prepared = self._prepare(texts)
        probabilities = self.pipeline.predict_proba(prepared)
        return np.asarray(probabilities, dtype=np.float64)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        prepared = self._prepare(texts)
        probabilities = np.asarray(self.pipeline.predict_proba(prepared))
        classes = [str(value) for value in self.pipeline.classes_]
        results: list[Prediction] = []
        for row in probabilities:
            index = int(np.argmax(row))
            results.append(
                Prediction(
                    label=classes[index],
                    confidence=float(row[index]),
                    probabilities={
                        label: float(row[class_index])
                        for class_index, label in enumerate(classes)
                    },
                )
            )
        return results

    def forward(self, texts: Sequence[str]) -> np.ndarray:
        """Expose the backend's probability forward path for integrations."""

        return self.predict_proba(texts)

    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "model_version": self.model_version,
            "labels": list(self.labels),
            "algorithm": "tfidf-logistic-regression",
        }

    def save(self, path: str | Path) -> Path:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "backend": self.backend_name,
                "model_version": self.model_version,
                "labels": list(self.labels),
                "pipeline": self.pipeline,
            },
            artifact_path,
        )
        return artifact_path


def train_sklearn_model(
    texts: Sequence[str], labels: Sequence[str], seed: int = 42
) -> SklearnBackend:
    """Fit the sklearn pipeline on normalized ticket text."""

    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty and the same length")
    prepared = preprocess_texts(texts)
    if any(not text for text in prepared):
        raise ValueError("training text must not be blank")
    label_names = tuple(sorted({str(label) for label in labels}))
    if len(label_names) < 2:
        raise ValueError("training requires at least two labels")
    pipeline = build_sklearn_pipeline(seed)
    pipeline.fit(prepared, [str(label) for label in labels])
    return SklearnBackend(pipeline, label_names)


def load_sklearn_backend(path: str | Path) -> SklearnBackend:
    """Load and validate a sklearn artifact from disk."""

    loaded = joblib.load(Path(path))
    if isinstance(loaded, Pipeline):
        classes = getattr(loaded, "classes_", None)
        if classes is None:
            raise ValueError("sklearn pipeline artifact is not fitted")
        return SklearnBackend(loaded, [str(value) for value in classes])
    if not isinstance(loaded, dict):
        raise ValueError("invalid sklearn artifact")
    return SklearnBackend.from_artifact(cast(dict[str, Any], loaded))


# Short aliases make the backend boundary convenient for callers.
SklearnTextBackend = SklearnBackend
train_sklearn = train_sklearn_model
