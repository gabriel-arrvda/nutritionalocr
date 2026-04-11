from pathlib import Path
from typing import TypedDict

from src.training.config import TrainingConfig
from src.training.io import ensure_training_dirs


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
) -> DryRunReport:
    if not dry_run:
        raise NotImplementedError("Non-dry pipeline execution is not implemented yet")

    ensure_training_dirs(logs_dir=config.logs_dir, data_dir=config.data_dir)

    return {
        "status": "dry_run_ok",
        "stages": ["recognition", "detection"],
        "artifacts": {
            "processed_csv": processed_csv,
            "image_root": image_root,
            "recognition_run_dir": config.logs_dir / "recognition",
            "detection_run_dir": config.logs_dir / "detection",
        },
    }
