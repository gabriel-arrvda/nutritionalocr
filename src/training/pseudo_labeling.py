from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import pandas as pd

SOURCE_KIND_COLUMN = "source_kind"
SOURCE_KIND_PSEUDO_LABEL = "pseudo_label"
PREDICTION_TEXT_COLUMN = "prediction_text"
LABEL_TEXT_COLUMN = "label_text"


class _PseudoLabelFilterConfig(Protocol):
    confidence_threshold: float


class _PseudoRatioConfig(Protocol):
    max_pseudo_ratio_per_language: float


def _is_consistent_prediction_text(value: object) -> bool:
    if not isinstance(value, str):
        return False

    stripped_value = value.strip()
    if not stripped_value:
        return False

    return all(ch.isprintable() and ch not in {"\n", "\r"} for ch in value)


def filter_pseudo_labels(df: pd.DataFrame, cfg: _PseudoLabelFilterConfig) -> pd.DataFrame:
    missing_columns = [
        column
        for column in (PREDICTION_TEXT_COLUMN, "confidence")
        if column not in df.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"missing required columns: {missing}")

    filtered = df.loc[df["confidence"] >= cfg.confidence_threshold].copy()
    filtered = filtered.loc[filtered[PREDICTION_TEXT_COLUMN].map(_is_consistent_prediction_text)]
    filtered[SOURCE_KIND_COLUMN] = SOURCE_KIND_PSEUDO_LABEL
    filtered = filtered.rename(columns={PREDICTION_TEXT_COLUMN: LABEL_TEXT_COLUMN})
    return filtered


def enforce_pseudo_ratio_cap(stats: Mapping[str, int | float], cfg: _PseudoRatioConfig) -> None:
    pseudo = float(stats.get("pseudo", 0))
    human = float(stats.get("human", 0))
    total = pseudo + human
    if total <= 0:
        return

    pseudo_ratio = pseudo / total
    if pseudo_ratio > cfg.max_pseudo_ratio_per_language:
        raise ValueError("pseudo-label ratio exceeds max_pseudo_ratio_per_language")
