from pathlib import Path

from ml_model_service.data import load_tickets
from ml_model_service.training import (
    evaluate_artifact,
    evaluate_backend,
    train_and_save,
    train_sklearn_model,
)

DATA = Path(__file__).parents[1] / "data" / "tickets.csv"


def test_fixture_loads_with_expected_labels() -> None:
    frame = load_tickets(DATA)
    assert len(frame) == 24
    assert tuple(sorted(frame["label"].unique())) == (
        "account",
        "billing",
        "shipping",
        "technical",
    )


def test_sklearn_training_and_evaluation_are_deterministic(tmp_path: Path) -> None:
    frame = load_tickets(DATA)
    texts = frame["text"].tolist()
    labels = frame["label"].tolist()
    first = train_sklearn_model(texts, labels, seed=42)
    second = train_sklearn_model(texts, labels, seed=42)
    first_predictions = first.predict(texts)
    second_predictions = second.predict(texts)
    assert [item.label for item in first_predictions] == [
        item.label for item in second_predictions
    ]
    assert evaluate_backend(first, texts, labels) == evaluate_backend(
        second, texts, labels
    )

    first_result = train_and_save("sklearn", DATA, tmp_path / "one.joblib")
    second_result = train_and_save("sklearn", DATA, tmp_path / "two.joblib")
    assert first_result.metrics == second_result.metrics
    assert evaluate_artifact("sklearn", DATA, first_result.artifact_path).accuracy > 0.8
