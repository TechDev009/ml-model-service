"""Small CPU-friendly PyTorch feature classifier."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from ..preprocessing import preprocess_texts
from ..protocol import Prediction

MODEL_VERSION = "torch-bow-mlp-v1"


@dataclass(frozen=True, slots=True)
class FeatureEncoder:
    """Deterministic count-vector encoder shared by torch training/inference."""

    vocabulary: tuple[str, ...]

    @classmethod
    def fit(cls, texts: Sequence[str]) -> "FeatureEncoder":
        tokens = {
            token for text in preprocess_texts(texts) for token in text.split() if token
        }
        if not tokens:
            tokens = {"<empty>"}
        return cls(tuple(sorted(tokens)))

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        indices = {token: index for index, token in enumerate(self.vocabulary)}
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row, text in enumerate(preprocess_texts(texts)):
            for token in text.split():
                index = indices.get(token)
                if index is not None:
                    matrix[row, index] += 1.0
        return matrix


class TorchFeatureClassifier(nn.Module):
    """A compact deterministic MLP over bag-of-words counts."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class TorchTextBackend:
    """Serve a CPU PyTorch classifier through the common model protocol."""

    backend_name = "torch"

    def __init__(
        self,
        model: TorchFeatureClassifier,
        encoder: FeatureEncoder,
        labels: Sequence[str],
        model_version: str = MODEL_VERSION,
    ) -> None:
        self.model = model.to("cpu")
        self.model.eval()
        self.encoder = encoder
        self.labels = tuple(labels)
        self.model_version = model_version
        if not self.labels:
            raise ValueError("a model must contain at least one label")

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "TorchTextBackend":
        vocabulary = artifact.get("vocabulary")
        labels = artifact.get("labels")
        state_dict = artifact.get("state_dict")
        input_dim = artifact.get("input_dim")
        hidden_dim = artifact.get("hidden_dim", 32)
        if not isinstance(vocabulary, (list, tuple)):
            raise ValueError("invalid torch vocabulary")
        if not isinstance(labels, (list, tuple)):
            raise ValueError("invalid torch labels")
        if not isinstance(state_dict, dict) or not isinstance(input_dim, int):
            raise ValueError("invalid torch state")
        model = TorchFeatureClassifier(input_dim, len(labels), int(hidden_dim))
        model.load_state_dict(state_dict)
        return cls(
            model,
            FeatureEncoder(tuple(str(token) for token in vocabulary)),
            [str(label) for label in labels],
            str(artifact.get("model_version", MODEL_VERSION)),
        )

    def _prepare(self, texts: Sequence[str]) -> list[str]:
        prepared = preprocess_texts(texts)
        if not prepared:
            raise ValueError("at least one text is required")
        if any(not text for text in prepared):
            raise ValueError("text must not be blank")
        return prepared

    def forward(self, texts: Sequence[str]) -> np.ndarray:
        """Run the CPU network and return logits as a NumPy matrix."""

        prepared = self._prepare(texts)
        features = torch.from_numpy(self.encoder.transform(prepared))
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(features)
        return logits.detach().cpu().numpy()

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        logits = torch.from_numpy(self.forward(texts))
        return torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        probabilities = self.predict_proba(texts)
        results: list[Prediction] = []
        for row in probabilities:
            index = int(np.argmax(row))
            results.append(
                Prediction(
                    label=self.labels[index],
                    confidence=float(row[index]),
                    probabilities={
                        label: float(row[class_index])
                        for class_index, label in enumerate(self.labels)
                    },
                )
            )
        return results

    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "model_version": self.model_version,
            "labels": list(self.labels),
            "algorithm": "bag-of-words-mlp",
            "device": "cpu",
        }

    def save(self, path: str | Path) -> Path:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": self.backend_name,
                "model_version": self.model_version,
                "labels": list(self.labels),
                "vocabulary": list(self.encoder.vocabulary),
                "input_dim": len(self.encoder.vocabulary),
                "hidden_dim": self.model.hidden_dim,
                "state_dict": self.model.state_dict(),
            },
            artifact_path,
        )
        return artifact_path


def train_torch_model(
    texts: Sequence[str],
    labels: Sequence[str],
    seed: int = 42,
    epochs: int = 120,
    learning_rate: float = 0.03,
) -> TorchTextBackend:
    """Train the small classifier on CPU with deterministic settings."""

    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty and the same length")
    prepared = preprocess_texts(texts)
    if any(not text for text in prepared):
        raise ValueError("training text must not be blank")
    label_names = tuple(sorted({str(label) for label in labels}))
    if len(label_names) < 2:
        raise ValueError("training requires at least two labels")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    encoder = FeatureEncoder.fit(prepared)
    features = torch.from_numpy(encoder.transform(prepared))
    target = torch.tensor(
        [label_names.index(str(label)) for label in labels], dtype=torch.long
    )
    model = TorchFeatureClassifier(len(encoder.vocabulary), len(label_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), target)
        loss.backward()
        optimizer.step()
    return TorchTextBackend(model, encoder, label_names)


def load_torch_backend(path: str | Path) -> TorchTextBackend:
    """Load a CPU-mapped PyTorch artifact."""

    loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError("invalid torch artifact")
    return TorchTextBackend.from_artifact(cast(dict[str, Any], loaded))


TorchBackend = TorchTextBackend
train_torch = train_torch_model
