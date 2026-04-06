"""Consolidation helpers reused by notebook Section 6."""

from __future__ import annotations

import json
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

import pandas as pd

Number = Union[int, float]
Nutrients = Dict[str, Number]
TranslateFn = Callable[[Nutrients, str], Nutrients]


def match_original_row(
    original_df: pd.DataFrame,
    image_url: Optional[str],
    image_url_column: str = "image_url",
) -> Tuple[Optional[pd.Series], Optional[str]]:
    if image_url_column not in original_df.columns:
        return None, f"'{image_url_column}' column not available"

    if pd.isna(image_url):
        return None, f"{image_url_column} is empty"

    original_match = original_df[original_df[image_url_column] == image_url]
    if len(original_match) == 1:
        return original_match.iloc[0], None

    if len(original_match) > 1:
        return None, f"Ambiguous {image_url_column} match ({len(original_match)} rows)"

    return None, f"No deterministic match for {image_url_column}"


def get_mapping_warning_reason(mapping_warning: Optional[str]) -> Optional[str]:
    if not mapping_warning:
        return None

    warning_lower = mapping_warning.lower()
    if "ambiguous" in warning_lower:
        return "ambiguous"
    if "no deterministic match" in warning_lower:
        return "no match"
    if "image_url" in warning_lower:
        return "missing image_url"
    return "other"


def count_unmatched_mappings(warning_reason_counts: Optional[Mapping[str, int]]) -> int:
    if not warning_reason_counts:
        return 0
    return int(sum(int(count) for count in warning_reason_counts.values()))


def assert_quality_gates(
    nutrients_json_completion_pct: float,
    unmatched_mapping_rate: float,
    thresholds: Mapping[str, float],
) -> bool:
    failure_messages = []
    min_completion = thresholds["min_nutrients_completion_pct"]
    max_unmatched = thresholds["max_unmatched_mapping_rate"]

    if nutrients_json_completion_pct < min_completion:
        failure_messages.append(
            f"nutrients_json completion {nutrients_json_completion_pct:.2%} below threshold {min_completion:.2%}"
        )

    if unmatched_mapping_rate > max_unmatched:
        failure_messages.append(
            f"unmatched mapping rate {unmatched_mapping_rate:.2%} above threshold {max_unmatched:.2%}"
        )

    if failure_messages:
        raise AssertionError("Quality gate(s) failed: " + " | ".join(failure_messages))

    return True


def _translate_with_map(nutrients: Nutrients, translation_map: Mapping[str, str]) -> Nutrients:
    translated: Nutrients = {}
    for nutrient_name, value in nutrients.items():
        lookup_key = str(nutrient_name).strip().lower()
        translated_key = translation_map.get(lookup_key, str(nutrient_name))
        translated[translated_key] = value
    return translated


def serialize_translated_nutrients(
    nutrients: Mapping[str, Number],
    target_lang: str = "pt",
    translation_map: Optional[Mapping[str, str]] = None,
    translate_fn: Optional[TranslateFn] = None,
) -> str:
    normalized_nutrients: Nutrients = {
        str(nutrient_name): value for nutrient_name, value in nutrients.items()
    }

    translated: Nutrients = normalized_nutrients.copy()

    if target_lang == "pt":
        try:
            if translate_fn is not None:
                translated = {
                    str(nutrient_name): value
                    for nutrient_name, value in translate_fn(
                        normalized_nutrients, target_lang
                    ).items()
                }
            elif translation_map is not None:
                translated = _translate_with_map(normalized_nutrients, translation_map)
        except Exception:
            translated = normalized_nutrients.copy()

    return json.dumps(translated, ensure_ascii=False, sort_keys=True)
