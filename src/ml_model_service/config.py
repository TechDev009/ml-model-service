"""Application settings loaded from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for training and serving.

    Relative defaults point at the repository so the application can be started
    either from this project directory or by using ``uv --project`` elsewhere.
    Environment variables use the uppercase field names, for example
    ``MODEL_BACKEND=torch``.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    model_backend: Literal["sklearn", "torch"] = "sklearn"
    data_path: Path = Field(default=PROJECT_ROOT / "data" / "tickets.csv")
    artifact_dir: Path = Field(default=PROJECT_ROOT / "artifacts")
    artifact_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"

    def resolved_artifact_path(self) -> Path:
        """Return the configured artifact path for the selected backend."""

        if self.artifact_path is not None:
            return self.artifact_path
        suffix = "joblib" if self.model_backend == "sklearn" else "pt"
        return self.artifact_dir / f"{self.model_backend}.{suffix}"
