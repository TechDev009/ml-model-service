"""Support-ticket classification service."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ml-model-service")
except PackageNotFoundError:  # pragma: no cover - useful when running from a checkout
    __version__ = "0.1.0"

__all__ = ["__version__"]
