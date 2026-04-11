from types import SimpleNamespace

import pandas as pd
import pytest

from src.training.pseudo_labeling import (
    SOURCE_KIND_PSEUDO_LABEL,
    compute_pseudo_ratio_stats_by_language,
    enforce_pseudo_ratio_cap,
    filter_pseudo_labels,
    validate_pseudo_ratio_stats_by_language,
)


def test_filter_pseudo_labels_requires_confidence_column():
    df = pd.DataFrame([{"prediction_text": "foo"}])
    cfg = SimpleNamespace(confidence_threshold=0.8)

    with pytest.raises(ValueError, match="missing required columns: confidence"):
        filter_pseudo_labels(df, cfg)


def test_filter_pseudo_labels_requires_prediction_text_column():
    df = pd.DataFrame([{"confidence": 0.9}])
    cfg = SimpleNamespace(confidence_threshold=0.8)

    with pytest.raises(ValueError, match="missing required columns: prediction_text"):
        filter_pseudo_labels(df, cfg)


def test_filter_pseudo_labels_rejects_blank_or_empty_predictions():
    df = pd.DataFrame(
        [
            {"prediction_text": "", "confidence": 0.95},
            {"prediction_text": "   ", "confidence": 0.95},
            {"prediction_text": "ok", "confidence": 0.95},
        ]
    )
    cfg = SimpleNamespace(confidence_threshold=0.8)

    filtered = filter_pseudo_labels(df, cfg)

    assert list(filtered["label_text"]) == ["ok"]


def test_filter_pseudo_labels_rejects_inconsistent_predictions():
    df = pd.DataFrame(
        [
            {"prediction_text": 123, "confidence": 0.95},
            {"prediction_text": "linha\nquebrada", "confidence": 0.95},
            {"prediction_text": "valido", "confidence": 0.95},
        ]
    )
    cfg = SimpleNamespace(confidence_threshold=0.8)

    filtered = filter_pseudo_labels(df, cfg)

    assert list(filtered["label_text"]) == ["valido"]


def test_filter_pseudo_labels_filters_by_threshold_and_sets_required_columns():
    df = pd.DataFrame(
        [
            {"prediction_text": "alto", "confidence": 0.95, "language": "pt"},
            {"prediction_text": "baixo", "confidence": 0.79, "language": "pt"},
            {"prediction_text": "limiar", "confidence": 0.80, "language": "en"},
        ]
    )
    cfg = SimpleNamespace(confidence_threshold=0.8)

    filtered = filter_pseudo_labels(df, cfg)

    assert list(filtered["label_text"]) == ["alto", "limiar"]
    assert "prediction_text" not in filtered.columns
    assert set(filtered["source_kind"]) == {"pseudo_label"}
    assert list(filtered["confidence"]) == [0.95, 0.80]
    assert list(filtered["language"]) == ["pt", "en"]


def test_enforce_pseudo_ratio_cap_raises_when_ratio_exceeds_limit():
    stats = {"pseudo": 8, "human": 2}
    cfg = SimpleNamespace(max_pseudo_ratio_per_language=0.7)

    with pytest.raises(ValueError, match="pseudo-label ratio exceeds max_pseudo_ratio_per_language"):
        enforce_pseudo_ratio_cap(stats, cfg)


def test_enforce_pseudo_ratio_cap_allows_ratio_equal_to_limit():
    stats = {"pseudo": 7, "human": 3}
    cfg = SimpleNamespace(max_pseudo_ratio_per_language=0.7)

    enforce_pseudo_ratio_cap(stats, cfg)


def test_compute_pseudo_ratio_stats_by_language_counts_human_and_pseudo_rows():
    df = pd.DataFrame(
        [
            {"language": "pt", "source_kind": SOURCE_KIND_PSEUDO_LABEL},
            {"language": "pt", "source_kind": "human_label"},
            {"language": "en", "source_kind": SOURCE_KIND_PSEUDO_LABEL},
            {"language": "en", "source_kind": SOURCE_KIND_PSEUDO_LABEL},
        ]
    )

    stats = compute_pseudo_ratio_stats_by_language(df)

    assert stats == {
        "pt": {"pseudo": 1, "human": 1},
        "en": {"pseudo": 2, "human": 0},
    }


def test_validate_pseudo_ratio_stats_by_language_raises_for_exceeding_language():
    cfg = SimpleNamespace(max_pseudo_ratio_per_language=0.7)
    stats_by_language = {
        "pt": {"pseudo": 7, "human": 3},
        "en": {"pseudo": 8, "human": 2},
    }

    with pytest.raises(
        ValueError,
        match="pseudo-label ratio exceeds max_pseudo_ratio_per_language for language: en",
    ):
        validate_pseudo_ratio_stats_by_language(stats_by_language, cfg)
