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
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")

    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must satisfy 0 <= test_ratio < 1")

    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    required_columns = ("language", "source_kind")
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {', '.join(missing_columns)}")

    if df.empty:
        raise ValueError("input dataframe cannot be empty")

    strat_columns = list(required_columns)
    null_mask = df[strat_columns].isna().any(axis=1)
    blank_mask = (
        df[strat_columns]
        .astype(str)
        .apply(lambda column: column.str.strip().eq(""))
        .any(axis=1)
    )
    if (null_mask | blank_mask).any():
        raise ValueError("language and source_kind must contain only non-blank values")

    train_ratio = 1 - val_ratio - test_ratio
    split_ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    split_order = ("train", "val", "test")
    active_splits = [split for split in split_order if split_ratios[split] > 0]

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for (language, source_kind), group in df.groupby(["language", "source_kind"], sort=False):
        indices = group.index.to_numpy()
        shuffled = rng.permutation(indices)
        group_size = len(shuffled)

        minimum_required = len(active_splits)
        if group_size < minimum_required:
            raise ValueError(
                "cannot preserve stratified representation for stratum "
                f"({language}, {source_kind}) with {group_size} rows; "
                f"need at least {minimum_required} rows for ratios "
                f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
            )

        split_sizes = {"train": 0, "val": 0, "test": 0}
        for split in active_splits:
            split_sizes[split] = 1

        remaining = group_size - minimum_required
        targets = {
            split: group_size * split_ratios[split]
            for split in active_splits
        }

        while remaining > 0:
            selected_split = max(
                active_splits,
                key=lambda split: (
                    targets[split] - split_sizes[split],
                    -split_order.index(split),
                ),
            )
            split_sizes[selected_split] += 1
            remaining -= 1

        test_size = split_sizes["test"]
        val_size = split_sizes["val"]

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
