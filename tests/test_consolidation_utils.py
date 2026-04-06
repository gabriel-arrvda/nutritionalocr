import json

import pandas as pd
import pytest

from src.utils.consolidation import (
    assert_quality_gates,
    count_unmatched_mappings,
    get_mapping_warning_reason,
    match_original_row,
    serialize_translated_nutrients,
)


def test_match_original_row_returns_single_deterministic_match():
    original_df = pd.DataFrame(
        {
            "image_url": ["match-a", "match-b"],
            "calories": [10.0, 20.0],
        }
    )

    row, warning = match_original_row(original_df, "match-b")

    assert warning is None
    assert row is not None
    assert float(row["calories"]) == 20.0


def test_match_original_row_returns_warning_when_no_match():
    original_df = pd.DataFrame({"image_url": ["match-a"], "calories": [10.0]})

    row, warning = match_original_row(original_df, "missing")

    assert row is None
    assert warning == "No deterministic match for image_url"


def test_count_unmatched_mappings_counts_all_reasons_including_other():
    warning_reason_counts = {
        "missing image_url": 2,
        "ambiguous": 1,
        "no match": 3,
        "other": 4,
    }

    assert count_unmatched_mappings(warning_reason_counts) == 10


def test_get_mapping_warning_reason_handles_supported_categories():
    assert get_mapping_warning_reason("Ambiguous image_url match (2 rows)") == "ambiguous"
    assert get_mapping_warning_reason("No deterministic match for image_url") == "no match"
    assert get_mapping_warning_reason("'image_url' column not available") == "missing image_url"
    assert get_mapping_warning_reason("unexpected warning") == "other"


def test_assert_quality_gates_passes_inside_thresholds():
    thresholds = {
        "min_nutrients_completion_pct": 0.80,
        "max_unmatched_mapping_rate": 0.20,
    }

    assert assert_quality_gates(0.80, 0.20, thresholds) is True


def test_assert_quality_gates_fails_outside_thresholds():
    thresholds = {
        "min_nutrients_completion_pct": 0.80,
        "max_unmatched_mapping_rate": 0.20,
    }

    with pytest.raises(AssertionError, match="Quality gate\\(s\\) failed"):
        assert_quality_gates(0.79, 0.21, thresholds)


def test_serialize_translated_nutrients_returns_valid_json():
    payload = serialize_translated_nutrients(
        {"calories": 100, "protein": 12.5},
        target_lang="pt",
        translation_map={"calories": "calorias", "protein": "proteína"},
    )

    parsed = json.loads(payload)
    assert parsed == {"calorias": 100, "proteína": 12.5}


def test_serialize_translated_nutrients_fallbacks_when_translation_fails():
    def broken_translate(_: dict, __: str) -> dict:
        raise RuntimeError("translation failed")

    payload = serialize_translated_nutrients(
        {"calories": 100},
        target_lang="pt",
        translate_fn=broken_translate,
    )

    assert json.loads(payload) == {"calories": 100}
