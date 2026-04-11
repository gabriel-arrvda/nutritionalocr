from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd


def normalize_label_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


def stratified_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if df.empty:
        raise ValueError("input dataframe cannot be empty")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in df.groupby(["language", "source_kind"], sort=False):
        indices = group.index.to_numpy()
        shuffled = rng.permutation(indices)

        test_size = int(len(shuffled) * test_ratio)
        val_size = int(len(shuffled) * val_ratio)

        test_indices.extend(shuffled[:test_size].tolist())
        val_indices.extend(shuffled[test_size : test_size + val_size].tolist())
        train_indices.extend(shuffled[test_size + val_size :].tolist())

    train_df = df.loc[train_indices]
    val_df = df.loc[val_indices]
    test_df = df.loc[test_indices]

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
