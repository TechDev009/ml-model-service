"""One-process model loading and inference registry."""

from collections.abc import Sequence
from pathlib import Path

from .config import Settings
from .protocol import ModelBackend, Prediction
from .training import BackendName, load_backend, train_and_save


class ModelRegistry:
    """Load one immutable backend at startup and reuse it for every request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._backend: ModelBackend | None = None
        self.error: str | None = None
        self.artifact_path: Path = settings.resolved_artifact_path()

    @property
    def backend(self) -> ModelBackend | None:
        return self._backend

    @property
    def ready(self) -> bool:
        return self._backend is not None

    def load(self) -> None:
        """Load an artifact, bootstrapping the deterministic default if absent."""

        backend_name: BackendName = self.settings.model_backend
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.artifact_path.is_file():
            train_and_save(
                backend_name,
                self.settings.data_path,
                self.artifact_path,
            )
        self._backend = load_backend(backend_name, self.artifact_path)
        self.error = None

    def fail(self, error: Exception) -> None:
        self._backend = None
        self.error = str(error)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if self._backend is None:
            raise RuntimeError(self.error or "model is not ready")
        return self._backend.predict(texts)

    def metadata(self) -> dict[str, object]:
        if self._backend is None:
            raise RuntimeError(self.error or "model is not ready")
        return {
            **self._backend.metadata(),
            "artifact": str(self.artifact_path),
        }
