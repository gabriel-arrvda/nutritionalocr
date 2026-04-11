from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import TypedDict

from src.training.config import TrainingConfig
from src.training.io import ensure_training_dirs
from src.training.pseudo_labeling import validate_pseudo_ratio_stats_by_language


class PipelineArtifacts(TypedDict):
    processed_csv: Path
    image_root: Path
    recognition_run_dir: Path
    detection_run_dir: Path


class DryRunReport(TypedDict):
    status: str
    stages: list[str]
    artifacts: PipelineArtifacts


def run_training_pipeline(
    config: TrainingConfig,
    processed_csv: Path,
    image_root: Path,
    dry_run: bool = False,
    pseudo_ratio_stats_by_language: Mapping[str, Mapping[str, int | float]] | None = None,
) -> DryRunReport:
    if not dry_run:
        raise NotImplementedError("Non-dry pipeline execution is not implemented yet")

    artifacts = ensure_training_dirs(logs_dir=config.logs_dir, data_dir=config.data_dir)
    validate_pseudo_ratio_stats_by_language(
        pseudo_ratio_stats_by_language or {},
        config.pseudo_label,
    )

    return {
        "status": "dry_run_ok",
        "stages": ["recognition", "detection"],
        "artifacts": {
            "processed_csv": processed_csv,
            "image_root": image_root,
            "recognition_run_dir": artifacts["recognition_run_dir"],
            "detection_run_dir": artifacts["detection_run_dir"],
        },
    }
