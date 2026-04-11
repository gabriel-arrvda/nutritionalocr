from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.training.config import TrainingConfig


def normalize_label_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


def build_training_manifest(validated_rows: pd.DataFrame) -> pd.DataFrame:
    manifest_columns = ["image_path", "label_text", "language", "source_kind"]
    missing_columns = [column for column in manifest_columns if column not in validated_rows.columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {', '.join(missing_columns)}")

    manifest = validated_rows.loc[:, manifest_columns].copy()
    manifest = manifest.sort_values(
        by=["language", "source_kind", "image_path", "label_text"],
        kind="mergesort",
    )
    return manifest.reset_index(drop=True)


def write_training_manifests(
    *,
    validated_rows: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Path]:
    splits = stratified_split(validated_rows, val_ratio=0.2, test_ratio=0.2, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {
        "train": output_dir / "manifest_train.csv",
        "val": output_dir / "manifest_val.csv",
        "test": output_dir / "manifest_test.csv",
    }
    for split_name, split_df in splits.items():
        manifest_df = build_training_manifest(split_df)
        manifest_df.to_csv(manifest_paths[split_name], index=False)
    return manifest_paths


def validate_training_dataset(
    *,
    config: TrainingConfig,
    processed_csv: Path,
    image_root: Path,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    errors: list[str] = []
    warnings: list[str] = []
    validated_rows: pd.DataFrame | None = None
    required_columns = ("image_path", "label_text", "language", "source_kind")

    if not processed_csv.is_file():
        errors.append(f"processed csv not found: {processed_csv}")
    else:
        validated_rows = pd.read_csv(processed_csv)
        missing_columns = [column for column in required_columns if column not in validated_rows.columns]
        if missing_columns:
            errors.append(f"missing required columns: {', '.join(missing_columns)}")
        elif validated_rows.empty:
            errors.append("processed dataset is empty")
        else:
            for idx, row in validated_rows.iterrows():
                language = str(row["language"]).strip()
                if language not in config.languages:
                    errors.append(f"row {idx}: invalid language '{language}'")

                image_path_value = str(row["image_path"]).strip()
                resolved_path = Path(image_path_value)
                if not resolved_path.is_absolute():
                    resolved_path = image_root / resolved_path

                if not resolved_path.is_file():
                    errors.append(f"row {idx}: image not found '{resolved_path}'")
                    continue

                try:
                    with Image.open(resolved_path) as img:
                        width, height = img.size
                        if width < config.min_image_width or height < config.min_image_height:
                            errors.append(
                                "row "
                                f"{idx}: image too small '{resolved_path}' "
                                f"({width}x{height}, min={config.min_image_width}x{config.min_image_height})"
                            )
                except (UnidentifiedImageError, OSError):
                    errors.append(f"row {idx}: unreadable image '{resolved_path}'")

    if not image_root.exists():
        warnings.append(f"image root not found: {image_root}")

    report = {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "row_count": 0 if validated_rows is None else int(len(validated_rows)),
        "processed_csv": str(processed_csv),
        "image_root": str(image_root),
    }
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.logs_dir / "dataset_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, validated_rows


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
