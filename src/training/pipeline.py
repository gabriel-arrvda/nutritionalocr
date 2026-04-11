from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import TypedDict

import pandas as pd

from src.training.config import TrainingConfig
from src.training.io import ensure_training_dirs
from src.training.pseudo_labeling import (
    compute_pseudo_ratio_stats_by_language,
    validate_pseudo_ratio_stats_by_language,
)


class PipelineArtifacts(TypedDict):
    processed_csv: Path
    image_root: Path
    recognition_run_dir: Path
    detection_run_dir: Path


class DryRunReport(TypedDict):
    status: str
    stages: list[str]
    artifacts: PipelineArtifacts


class DatasetValidationReport(TypedDict):
    status: str
    errors: list[str]
    warnings: list[str]
    row_count: int


def _validate_dataset(
    *,
    processed_csv: Path,
    image_root: Path,
    fail_fast: bool,
) -> tuple[DatasetValidationReport, pd.DataFrame | None]:
    errors: list[str] = []
    warnings: list[str] = []
    dataset_df: pd.DataFrame | None = None

    if not processed_csv.is_file():
        message = f"processed csv not found: {processed_csv}"
        if fail_fast:
            errors.append(message)
        else:
            warnings.append(message)
    else:
        dataset_df = pd.read_csv(processed_csv)
        required_columns = ("language", "source_kind")
        missing_columns = [column for column in required_columns if column not in dataset_df.columns]
        if missing_columns:
            errors.append(f"missing required columns: {', '.join(missing_columns)}")
        if dataset_df.empty:
            errors.append("processed dataset is empty")

    if not image_root.exists():
        message = f"image root not found: {image_root}"
        if fail_fast:
            errors.append(message)
        else:
            warnings.append(message)

    report: DatasetValidationReport = {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "row_count": 0 if dataset_df is None else int(len(dataset_df)),
    }
    return report, dataset_df


def _write_dataset_validation_report(
    *,
    logs_dir: Path,
    processed_csv: Path,
    image_root: Path,
    report: DatasetValidationReport,
) -> Path:
    report_path = logs_dir / "dataset_validation_report.json"
    payload = {
        "status": report["status"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "row_count": report["row_count"],
        "processed_csv": str(processed_csv),
        "image_root": str(image_root),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def run_training_pipeline(
    config: TrainingConfig,
    processed_csv: Path,
    image_root: Path,
    dry_run: bool = False,
    pseudo_ratio_stats_by_language: Mapping[str, Mapping[str, int | float]] | None = None,
) -> DryRunReport:
    artifacts = ensure_training_dirs(logs_dir=config.logs_dir, data_dir=config.data_dir)
    validation_report, dataset_df = _validate_dataset(
        processed_csv=processed_csv,
        image_root=image_root,
        fail_fast=not dry_run,
    )
    _write_dataset_validation_report(
        logs_dir=config.logs_dir,
        processed_csv=processed_csv,
        image_root=image_root,
        report=validation_report,
    )

    if validation_report["errors"]:
        raise ValueError("; ".join(validation_report["errors"]))

    stats_by_language = pseudo_ratio_stats_by_language
    if stats_by_language is None:
        stats_by_language = (
            compute_pseudo_ratio_stats_by_language(dataset_df)
            if dataset_df is not None
            else {}
        )

    validate_pseudo_ratio_stats_by_language(stats_by_language, config.pseudo_label)

    return {
        "status": "validation_only_dry_run" if dry_run else "validation_only_execute",
        "stages": ["recognition", "detection"],
        "artifacts": {
            "processed_csv": processed_csv,
            "image_root": image_root,
            "recognition_run_dir": artifacts["recognition_run_dir"],
            "detection_run_dir": artifacts["detection_run_dir"],
        },
    }
