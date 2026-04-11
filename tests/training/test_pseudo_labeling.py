from types import SimpleNamespace

import pandas as pd
import pytest

from src.training.pseudo_labeling import enforce_pseudo_ratio_cap, filter_pseudo_labels


def test_filter_pseudo_labels_requires_confidence_column():
    df = pd.DataFrame([{"prediction_text": "foo"}])
    cfg = SimpleNamespace(confidence_threshold=0.8)

    with pytest.raises(ValueError, match="missing required columns: confidence"):
        filter_pseudo_labels(df, cfg)


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
