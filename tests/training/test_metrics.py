from __future__ import annotations

import pytest

from src.training.metrics import compute_cer_wer_metrics, compute_detection_metrics


def test_compute_cer_wer_metrics_returns_overall_and_per_language():
    samples = [
        {"language": "pt", "reference_text": "abc", "predicted_text": "abc"},
        {"language": "pt", "reference_text": "abcd", "predicted_text": "abxd"},
        {"language": "en", "reference_text": "cat dog", "predicted_text": "cat"},
    ]

    metrics = compute_cer_wer_metrics(samples)

    assert metrics["overall"]["cer"] == pytest.approx(5 / 14)
    assert metrics["overall"]["wer"] == pytest.approx(0.5)
    assert metrics["per_language"]["pt"]["cer"] == pytest.approx(1 / 7)
    assert metrics["per_language"]["pt"]["wer"] == pytest.approx(0.5)
    assert metrics["per_language"]["en"]["cer"] == pytest.approx(4 / 7)
    assert metrics["per_language"]["en"]["wer"] == pytest.approx(0.5)


def test_compute_detection_metrics_is_deterministic_from_counts():
    metrics = compute_detection_metrics(true_positives=8, false_positives=2, false_negatives=1)

    assert metrics == {
        "precision": pytest.approx(0.8),
        "recall": pytest.approx(8 / 9),
        "f1": pytest.approx(16 / 19),
    }
