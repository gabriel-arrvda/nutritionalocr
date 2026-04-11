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


def test_run_training_pipeline_returns_validation_only_dry_run_report(tmp_path: Path):
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
    assert report["status"] == "validation_only_dry_run"
    assert report["stages"] == ["recognition", "detection"]
    assert (config.logs_dir / "dataset_validation_report.json").is_file()
    assert report["artifacts"] == {
        "processed_csv": processed_csv,
        "image_root": image_root,
        "recognition_run_dir": config.logs_dir / "recognition",
        "detection_run_dir": config.logs_dir / "detection",
    }


def test_run_training_pipeline_execute_mode_still_returns_validation_status(tmp_path: Path):
    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    processed_csv.write_text(
        "language,source_kind\npt,human\nen,human\nen,human\nen,pseudo_label\n",
        encoding="utf-8",
    )

    report = run_training_pipeline(
        config=config,
        processed_csv=processed_csv,
        image_root=image_root,
        dry_run=False,
    )
    assert report["status"] == "validation_only_execute"
    assert (config.logs_dir / "dataset_validation_report.json").is_file()


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
        match="pseudo-label ratio exceeds max_pseudo_ratio_per_language for language: pt",
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
    assert payload["status"] == "validation_only_dry_run"
    assert payload["stages"] == ["recognition", "detection"]
    assert payload["artifacts"]["processed_csv"] == str(processed_csv)
    assert payload["artifacts"]["image_root"] == str(image_root)


def test_train_ocr_cli_defaults_to_dry_run_and_succeeds(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "train_ocr.py")],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "validation_only_dry_run"


def test_train_ocr_cli_execute_mode_still_reports_validation_only_status(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    processed_csv = tmp_path / "data" / "processed" / "training" / "merged.csv"
    image_root = tmp_path / "data" / "images"
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    processed_csv.write_text(
        "language,source_kind\npt,human\nen,human\nen,human\nen,pseudo_label\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_ocr.py"),
            "--execute",
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
    assert payload["status"] == "validation_only_execute"
