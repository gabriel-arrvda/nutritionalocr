from pathlib import Path
import json
import subprocess
import sys

import pytest
from PIL import Image

from src.training.config import TrainingConfig
from src.training.io import ensure_training_dirs
from src.training.pipeline import run_training_pipeline


def test_ensure_training_dirs_creates_expected_directories(tmp_path: Path):
    logs_dir = tmp_path / "logs" / "training"
    data_dir = tmp_path / "data" / "processed" / "training"
    ensure_training_dirs(logs_dir=logs_dir, data_dir=data_dir)

    assert logs_dir.is_dir()
    assert data_dir.is_dir()


def _create_dataset_row(
    *,
    data_dir: Path,
    image_root: Path,
    image_name: str,
    label_text: str = "texto",
    language: str = "pt",
    source_kind: str = "human",
) -> str:
    image_root.mkdir(parents=True, exist_ok=True)
    image_path = image_root / image_name
    Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_path)
    return f"{image_name},{label_text},{language},{source_kind}\n"


def _create_valid_dataset(processed_csv: Path, image_root: Path) -> None:
    rows = ["image_path,label_text,language,source_kind\n"]
    for language in ("pt", "en"):
        for idx in range(3):
            rows.append(
                _create_dataset_row(
                    data_dir=processed_csv.parent,
                    image_root=image_root,
                    image_name=f"{language}_{idx}.png",
                    language=language,
                )
            )
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_csv.write_text("".join(rows), encoding="utf-8")


def test_run_training_pipeline_dry_run_fails_on_invalid_dataset(tmp_path: Path):
    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"

    with pytest.raises(ValueError, match="processed csv not found"):
        run_training_pipeline(
            config=config,
            processed_csv=processed_csv,
            image_root=image_root,
            dry_run=True,
        )

    assert (config.logs_dir / "dataset_validation_report.json").is_file()


def test_run_training_pipeline_dry_run_success_returns_ready_and_creates_manifests(
    tmp_path: Path,
):
    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    _create_valid_dataset(processed_csv, image_root)

    report = run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=True)

    assert config.logs_dir.is_dir()
    assert config.data_dir.is_dir()
    assert (config.logs_dir / "recognition").is_dir()
    assert (config.logs_dir / "detection").is_dir()
    assert report["status"] == "dry_run_ready"
    assert [step["name"] for step in report["steps"]] == [
        "validate_dataset",
        "build_manifests",
        "stage_a_train_recognizer",
        "stage_b_train_detector",
        "evaluate",
        "export",
    ]
    assert (config.logs_dir / "dataset_validation_report.json").is_file()
    assert report["artifacts"]["manifest_train_csv"].is_file()
    assert report["artifacts"]["manifest_val_csv"].is_file()
    assert report["artifacts"]["manifest_test_csv"].is_file()
    assert report["artifacts"] == {
        "processed_csv": processed_csv,
        "image_root": image_root,
        "recognition_run_dir": config.logs_dir / "recognition",
        "detection_run_dir": config.logs_dir / "detection",
        "manifest_train_csv": config.data_dir / "manifest_train.csv",
        "manifest_val_csv": config.data_dir / "manifest_val.csv",
        "manifest_test_csv": config.data_dir / "manifest_test.csv",
        "evaluation_report_path": config.logs_dir / "evaluation_report.json",
        "export_bundle_path": config.logs_dir / "model_export.tar.gz",
    }


def test_run_training_pipeline_execute_mode_calls_stages_and_returns_completed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import src.training.pipeline as pipeline_module

    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    _create_valid_dataset(processed_csv, image_root)

    call_order: list[str] = []

    def _fake_stage_a(*args, **kwargs):
        call_order.append("stage_a_train_recognizer")
        return {"status": "completed", "artifact_path": str(config.logs_dir / "recognition" / "checkpoint")}

    def _fake_stage_b(*args, **kwargs):
        call_order.append("stage_b_train_detector")
        return {"status": "completed", "artifact_path": str(config.logs_dir / "detection" / "checkpoint")}

    def _fake_evaluate(*args, **kwargs):
        call_order.append("evaluate")
        return {"status": "completed", "artifact_path": str(config.logs_dir / "evaluation_report.json")}

    def _fake_export(*args, **kwargs):
        call_order.append("export")
        return {"status": "completed", "artifact_path": str(config.logs_dir / "model_export.tar.gz")}

    monkeypatch.setattr(pipeline_module.stages, "stage_a_train_recognizer", _fake_stage_a)
    monkeypatch.setattr(pipeline_module.stages, "stage_b_train_detector", _fake_stage_b)
    monkeypatch.setattr(pipeline_module.stages, "evaluate", _fake_evaluate)
    monkeypatch.setattr(pipeline_module.stages, "export", _fake_export)

    report = run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)
    assert report["status"] == "completed"
    assert call_order == [
        "stage_a_train_recognizer",
        "stage_b_train_detector",
        "evaluate",
        "export",
    ]
    assert report["artifacts"]["evaluation_report_path"] == config.logs_dir / "evaluation_report.json"
    assert report["artifacts"]["export_bundle_path"] == config.logs_dir / "model_export.tar.gz"
    assert (config.logs_dir / "dataset_validation_report.json").is_file()


def test_run_training_pipeline_respects_non_default_config_directories(tmp_path: Path):
    logs_dir = tmp_path / "custom" / "outputs" / "logs-dir"
    data_dir = tmp_path / "custom" / "outputs" / "dataset-dir"
    config = TrainingConfig(data_dir=data_dir, logs_dir=logs_dir)

    processed_csv = data_dir / "merged.csv"
    image_root = tmp_path / "images"
    _create_valid_dataset(processed_csv, image_root)

    report = run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=True)

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
        _create_valid_dataset(config.data_dir / "merged.csv", tmp_path / "data" / "images")
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

    assert result.returncode != 0


def test_train_ocr_cli_execute_mode_reports_completed_status(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    processed_csv = tmp_path / "data" / "processed" / "training" / "merged.csv"
    image_root = tmp_path / "data" / "images"
    _create_valid_dataset(processed_csv, image_root)

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
    assert payload["status"] == "completed"
