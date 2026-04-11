from pathlib import Path

import pytest

from src.training.config import TrainingConfig
from src.training.io import ensure_training_dirs
from src.training.pipeline import run_training_pipeline


def test_ensure_training_dirs_creates_expected_directories(tmp_path: Path):
    ensure_training_dirs(tmp_path)

    assert (tmp_path / "logs" / "training").is_dir()
    assert (tmp_path / "data" / "processed" / "training").is_dir()


def test_run_training_pipeline_returns_dry_run_report(tmp_path: Path):
    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"

    report = run_training_pipeline(
        config=config,
        processed_csv=processed_csv,
        image_root=image_root,
        dry_run=True,
    )

    assert config.logs_dir.is_dir()
    assert config.data_dir.is_dir()
    assert report["status"] == "dry_run_ok"
    assert report["stages"] == ["recognition", "detection"]
    assert report["artifacts"] == {
        "processed_csv": processed_csv,
        "image_root": image_root,
        "recognition_run_dir": config.logs_dir / "recognition",
        "detection_run_dir": config.logs_dir / "detection",
    }


def test_run_training_pipeline_raises_for_non_dry_run(tmp_path: Path):
    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"

    with pytest.raises(NotImplementedError):
        run_training_pipeline(
            config=config,
            processed_csv=processed_csv,
            image_root=image_root,
            dry_run=False,
        )
