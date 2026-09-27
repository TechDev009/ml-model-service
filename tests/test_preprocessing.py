from ml_model_service.preprocessing import normalize_text, preprocess_texts


def test_normalize_text_is_stable_and_unicode_aware() -> None:
    assert normalize_text("  PAY\u00a0MENT\nFailed ") == "pay ment failed"
    assert preprocess_texts([" A ", "B\tC"]) == ["a", "b c"]
