from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import pandas as pd


class _PseudoLabelFilterConfig(Protocol):
    confidence_threshold: float


class _PseudoRatioConfig(Protocol):
    max_pseudo_ratio_per_language: float


def filter_pseudo_labels(df: pd.DataFrame, cfg: _PseudoLabelFilterConfig) -> pd.DataFrame:
    if "confidence" not in df.columns:
        raise ValueError("missing required columns: confidence")

    filtered = df.loc[df["confidence"] >= cfg.confidence_threshold].copy()
    filtered["source_kind"] = "pseudo_label"
    filtered = filtered.rename(columns={"prediction_text": "label_text"})
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
