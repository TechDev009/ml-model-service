"""Shared, deterministic text preprocessing."""

import re
import unicodedata
from collections.abc import Iterable

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize user text without discarding meaningful ticket words."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    return _WHITESPACE.sub(" ", normalized)


def preprocess_texts(texts: Iterable[str]) -> list[str]:
    """Normalize a sequence while preserving its order."""

    return [normalize_text(text) for text in texts]
