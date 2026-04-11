from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import pandas as pd

SOURCE_KIND_COLUMN = "source_kind"
SOURCE_KIND_PSEUDO_LABEL = "pseudo_label"
LANGUAGE_COLUMN = "language"
PREDICTION_TEXT_COLUMN = "prediction_text"
LABEL_TEXT_COLUMN = "label_text"


class _PseudoLabelFilterConfig(Protocol):
    confidence_threshold: float


class _PseudoRatioConfig(Protocol):
    max_pseudo_ratio_per_language: float


class _PseudoMergeConfig(_PseudoLabelFilterConfig, _PseudoRatioConfig, Protocol):
    pass


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


def compute_pseudo_ratio_stats_by_language(
    df: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    required_columns = (LANGUAGE_COLUMN, SOURCE_KIND_COLUMN)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"missing required columns: {missing}")

    stats_by_language: dict[str, dict[str, int]] = {}
    for language, group in df.groupby(LANGUAGE_COLUMN, sort=False):
        pseudo_count = int((group[SOURCE_KIND_COLUMN] == SOURCE_KIND_PSEUDO_LABEL).sum())
        human_count = int(len(group) - pseudo_count)
        stats_by_language[str(language)] = {"pseudo": pseudo_count, "human": human_count}
    return stats_by_language


def validate_pseudo_ratio_stats_by_language(
    stats_by_language: Mapping[str, Mapping[str, int | float]],
    cfg: _PseudoRatioConfig,
) -> None:
    for language, stats in stats_by_language.items():
        try:
            enforce_pseudo_ratio_cap(stats, cfg)
        except ValueError as exc:
            raise ValueError(
                "pseudo-label ratio exceeds max_pseudo_ratio_per_language "
                f"for language: {language}"
            ) from exc


def merge_filtered_pseudo_labels(
    *,
    human_labels_df: pd.DataFrame,
    pseudo_candidates_df: pd.DataFrame,
    cfg: _PseudoMergeConfig,
) -> pd.DataFrame:
    if pseudo_candidates_df.empty:
        return human_labels_df.copy()

    filtered_pseudo_df = filter_pseudo_labels(pseudo_candidates_df, cfg)
    if filtered_pseudo_df.empty:
        return human_labels_df.copy()

    selected_frames: list[pd.DataFrame] = [human_labels_df.copy()]
    for language, pseudo_group in filtered_pseudo_df.groupby(LANGUAGE_COLUMN, sort=False):
        human_count = int((human_labels_df[LANGUAGE_COLUMN] == language).sum())
        ratio_limit = cfg.max_pseudo_ratio_per_language
        if ratio_limit >= 1.0:
            max_pseudo_count = len(pseudo_group)
        elif human_count <= 0:
            max_pseudo_count = 0
        else:
            max_pseudo_count = int((ratio_limit * human_count) / (1.0 - ratio_limit))

        if max_pseudo_count <= 0:
            continue

        ranked_group = pseudo_group.sort_values(
            by=["confidence", "image_path"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected_frames.append(ranked_group.head(max_pseudo_count))

    merged_df = pd.concat(selected_frames, ignore_index=True)
    stats = compute_pseudo_ratio_stats_by_language(merged_df)
    validate_pseudo_ratio_stats_by_language(stats, cfg)
    return merged_df
