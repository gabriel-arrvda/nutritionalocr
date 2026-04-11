from pathlib import Path
import json
import subprocess
import sys

import pytest

from src.training.config import TrainingConfig
from src.training.io import ensure_training_dirs
from src.training.pipeline import run_training_pipeline


def test_ensure_training_dirs_creates_expected_directories(tmp_path: Path):
    logs_dir = tmp_path / "logs" / "training"
    data_dir = tmp_path / "data" / "processed" / "training"
    ensure_training_dirs(logs_dir=logs_dir, data_dir=data_dir)

    assert logs_dir.is_dir()
    assert data_dir.is_dir()


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
    assert (config.logs_dir / "recognition").is_dir()
    assert (config.logs_dir / "detection").is_dir()
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


def test_run_training_pipeline_respects_non_default_config_directories(tmp_path: Path):
    logs_dir = tmp_path / "custom" / "outputs" / "logs-dir"
    data_dir = tmp_path / "custom" / "outputs" / "dataset-dir"
    config = TrainingConfig(data_dir=data_dir, logs_dir=logs_dir)

    report = run_training_pipeline(
        config=config,
        processed_csv=data_dir / "merged.csv",
        image_root=tmp_path / "images",
        dry_run=True,
    )

    assert logs_dir.is_dir()
    assert data_dir.is_dir()
    assert (logs_dir / "recognition").is_dir()
    assert (logs_dir / "detection").is_dir()
    assert report["artifacts"]["recognition_run_dir"] == logs_dir / "recognition"
    assert report["artifacts"]["detection_run_dir"] == logs_dir / "detection"


def test_run_training_pipeline_enforces_pseudo_ratio_caps(tmp_path: Path):
    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
    )

    with pytest.raises(
        ValueError,
        match="pseudo-label ratio exceeds max_pseudo_ratio_per_language for language: en",
    ):
        run_training_pipeline(
            config=config,
            processed_csv=config.data_dir / "merged.csv",
            image_root=tmp_path / "data" / "images",
            dry_run=True,
            pseudo_ratio_stats_by_language={
                "pt": {"pseudo": 7, "human": 3},
                "en": {"pseudo": 8, "human": 2},
            },
        )


def test_train_ocr_cli_dry_run_prints_json_report(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    processed_csv = tmp_path / "data" / "processed" / "training" / "merged.csv"
    image_root = tmp_path / "data" / "images"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_ocr.py"),
            "--dry-run",
            "--processed-csv",
            str(processed_csv),
            "--image-root",
            str(image_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run_ok"
    assert payload["stages"] == ["recognition", "detection"]
    assert payload["artifacts"]["processed_csv"] == str(processed_csv)
    assert payload["artifacts"]["image_root"] == str(image_root)
