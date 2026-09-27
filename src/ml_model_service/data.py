"""Loading and validating the small deterministic ticket fixture."""

from pathlib import Path

import pandas as pd

from .preprocessing import normalize_text

REQUIRED_COLUMNS = ("text", "label")


def load_tickets(path: str | Path) -> pd.DataFrame:
    """Load a validated ticket data frame from CSV."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"ticket data not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"ticket data is missing required columns: {missing}")
    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    if frame.empty:
        raise ValueError("ticket data must contain at least one row")
    if frame["text"].isna().any() or frame["label"].isna().any():
        raise ValueError("ticket text and label must not be empty")

    frame["text"] = frame["text"].map(normalize_text)
    frame["label"] = frame["label"].astype(str).str.strip()
    if (frame["text"] == "").any() or (frame["label"] == "").any():
        raise ValueError("ticket text and label must not be blank")
    if frame["label"].nunique() < 2:
        raise ValueError("ticket data must contain at least two labels")
    return frame.reset_index(drop=True)
