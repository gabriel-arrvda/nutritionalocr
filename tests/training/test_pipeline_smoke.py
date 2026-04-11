from pathlib import Path
import json
import subprocess
import sys
import tarfile

import pytest
from PIL import Image

from src.training.config import PromotionGateConfig, PseudoLabelConfig, TrainingConfig
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
    source_kind: str = "human_label",
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


def _write_execute_metrics_artifacts(config: TrainingConfig) -> None:
    recognition_predictions = config.logs_dir / "recognition" / "val_predictions.csv"
    recognition_predictions.parent.mkdir(parents=True, exist_ok=True)
    recognition_predictions.write_text(
        "image_path,reference_text,predicted_text,language\n"
        "pt_0.png,texto 0,texto 0,pt\n"
        "en_0.png,texto 0,texta 0,en\n",
        encoding="utf-8",
    )
    detection_metrics = config.logs_dir / "detection" / "metrics.json"
    detection_metrics.parent.mkdir(parents=True, exist_ok=True)
    detection_metrics.write_text(
        json.dumps({"true_positives": 8, "false_positives": 2, "false_negatives": 1}),
        encoding="utf-8",
    )
    (config.logs_dir / "baseline_metrics.json").write_text(
        json.dumps({"cer": 0.5, "wer": 0.6}),
        encoding="utf-8",
    )


def _write_execute_metrics_artifacts_with_sources(config: TrainingConfig) -> None:
    recognition_predictions = config.logs_dir / "recognition" / "val_predictions.csv"
    recognition_predictions.parent.mkdir(parents=True, exist_ok=True)
    recognition_predictions.write_text(
        "image_path,reference_text,predicted_text,language,source_kind\n"
        "pt_human.png,texto humano,texto humano,pt,human_label\n"
        "en_pseudo.png,pseudo text,pseodo text,en,pseudo_label\n",
        encoding="utf-8",
    )
    detection_metrics = config.logs_dir / "detection" / "metrics.json"
    detection_metrics.parent.mkdir(parents=True, exist_ok=True)
    detection_metrics.write_text(
        json.dumps({"true_positives": 8, "false_positives": 2, "false_negatives": 1}),
        encoding="utf-8",
    )
    (config.logs_dir / "baseline_metrics.json").write_text(
        json.dumps({"cer": 0.5, "wer": 0.6}),
        encoding="utf-8",
    )


def test_run_training_pipeline_dry_run_fails_on_invalid_dataset(tmp_path: Path):
    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
        pseudo_label=PseudoLabelConfig(confidence_threshold=0.8, max_pseudo_ratio_per_language=0.8),
        promotion_gate=PromotionGateConfig(
            max_source_cer_degradation=1.0,
            max_source_wer_degradation=1.0,
        ),
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


def test_run_training_pipeline_dry_run_fails_on_empty_label_text(tmp_path: Path):
    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
        pseudo_label=PseudoLabelConfig(confidence_threshold=0.8, max_pseudo_ratio_per_language=0.8),
        promotion_gate=PromotionGateConfig(
            max_source_cer_degradation=1.0,
            max_source_wer_degradation=1.0,
        ),
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_root / "pt_0.png")
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_csv.write_text(
        "image_path,label_text,language,source_kind\n"
        "pt_0.png,,pt,human_label\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required field 'label_text' must be non-empty"):
        run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=True)


def test_run_training_pipeline_allows_unlabeled_rows_before_teacher_pass_and_enforces_labels_after_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import src.training.pipeline as pipeline_module

    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
        pseudo_label=PseudoLabelConfig(confidence_threshold=0.8, max_pseudo_ratio_per_language=0.8),
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    _create_valid_dataset(processed_csv, image_root)
    Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_root / "pt_3.png")
    Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_root / "pt_4.png")
    Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_root / "pt_5.png")
    processed_csv.write_text(
        "image_path,label_text,language,source_kind\n"
        "pt_0.png,texto 0,pt,human_label\n"
        "pt_1.png,texto 1,pt,human_label\n"
        "pt_3.png,texto 3,pt,human_label\n"
        "pt_2.png,,pt,human_label\n"
        "pt_4.png,,pt,human_label\n"
        "pt_5.png,,pt,human_label\n"
        "en_0.png,texto 0,en,human_label\n"
        "en_1.png,texto 1,en,human_label\n"
        "en_2.png,texto 2,en,human_label\n",
        encoding="utf-8",
    )

    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text(
            "image_path,prediction_text,confidence,language\n"
            "pt_2.png,texto pseudo,0.99,pt\n"
            "pt_4.png,texto pseudo 4,0.98,pt\n"
            "pt_5.png,texto pseudo 5,0.97,pt\n",
            encoding="utf-8",
        )
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)

    report = run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=True)

    assert report["status"] == "dry_run_ready"
    assert "pt_2.png" in (report["artifacts"]["pseudo_candidates_csv"]).read_text(encoding="utf-8")
    assert ",," not in report["artifacts"]["manifest_train_csv"].read_text(encoding="utf-8")


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
        "gpu_profile_check",
        "teacher_pass_generate_pseudo_labels",
        "merge_filtered_pseudo_labels",
        "validate_pseudo_ratio",
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
    assert report["artifacts"]["evaluation_report_path"].is_file()
    assert report["artifacts"]["baseline_comparison_path"].is_file()
    assert report["artifacts"]["metadata_bundle_path"].is_file()
    assert report["artifacts"]["inference_config_path"].is_file()

    evaluation_payload = json.loads(report["artifacts"]["evaluation_report_path"].read_text(encoding="utf-8"))
    assert "overall" in evaluation_payload["recognition"]
    assert "per_language" in evaluation_payload["recognition"]
    assert set(evaluation_payload["detection"]) == {"precision", "recall", "f1"}
    assert "baseline_vs_best" in evaluation_payload
    assert report["artifacts"] == {
        "processed_csv": processed_csv,
        "image_root": image_root,
        "recognition_run_dir": config.logs_dir / "recognition",
        "detection_run_dir": config.logs_dir / "detection",
        "manifest_train_csv": config.data_dir / "manifest_train.csv",
        "manifest_val_csv": config.data_dir / "manifest_val.csv",
        "manifest_test_csv": config.data_dir / "manifest_test.csv",
        "pseudo_candidates_csv": config.data_dir / "pseudo_candidates.csv",
        "evaluation_report_path": config.logs_dir / "evaluation_report.json",
        "baseline_comparison_path": config.logs_dir / "baseline_vs_best.json",
        "export_bundle_path": config.logs_dir / "model_export.tar.gz",
        "metadata_bundle_path": config.logs_dir / "metadata_bundle.json",
        "inference_config_path": config.logs_dir / "inference_config.json",
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
        _write_execute_metrics_artifacts(config)
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

    def _fake_teacher(*args, **kwargs):
        call_order.append("teacher_pass_generate_pseudo_labels")
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)
    monkeypatch.setattr(pipeline_module.stages, "stage_a_train_recognizer", _fake_stage_a)
    monkeypatch.setattr(pipeline_module.stages, "stage_b_train_detector", _fake_stage_b)
    monkeypatch.setattr(pipeline_module.stages, "evaluate", _fake_evaluate)
    monkeypatch.setattr(pipeline_module.stages, "export", _fake_export)

    report = run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)
    assert report["status"] == "completed"
    assert call_order == [
        "teacher_pass_generate_pseudo_labels",
        "stage_a_train_recognizer",
        "stage_b_train_detector",
        "evaluate",
        "export",
    ]
    assert report["artifacts"]["evaluation_report_path"] == config.logs_dir / "evaluation_report.json"
    assert report["artifacts"]["export_bundle_path"] == config.logs_dir / "model_export.tar.gz"
    assert (config.logs_dir / "dataset_validation_report.json").is_file()


def test_run_training_pipeline_uses_stage_a_hard_examples_for_stage_b(
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

    stage_a_artifact = config.logs_dir / "recognition" / "recognizer.ckpt"
    captured_stage_b_manifest: Path | None = None

    def _fake_stage_a(*args, **kwargs):
        stage_a_artifact.parent.mkdir(parents=True, exist_ok=True)
        stage_a_artifact.write_text("stage-a-output", encoding="utf-8")
        _write_execute_metrics_artifacts(config)
        return {"status": "completed", "artifact_path": str(stage_a_artifact)}

    def _fake_stage_b(*, manifest_train_csv: Path, detection_run_dir: Path, execute: bool):
        nonlocal captured_stage_b_manifest
        captured_stage_b_manifest = manifest_train_csv
        return {"status": "completed", "artifact_path": str(detection_run_dir / "detector.ckpt")}

    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    monkeypatch.setattr(pipeline_module.stages, "stage_a_train_recognizer", _fake_stage_a)
    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)
    monkeypatch.setattr(pipeline_module.stages, "stage_b_train_detector", _fake_stage_b)
    monkeypatch.setattr(
        pipeline_module.stages,
        "evaluate",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "evaluation_report.json")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "export",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "model_export.tar.gz")},
    )

    report = run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)

    assert report["status"] == "completed"
    assert captured_stage_b_manifest is not None
    assert captured_stage_b_manifest != report["artifacts"]["manifest_train_csv"]
    assert captured_stage_b_manifest.name == "manifest_stage_b_hard_examples.csv"
    assert captured_stage_b_manifest.is_file()


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


def test_train_ocr_cli_execute_mode_fails_fast_without_required_execute_artifacts(tmp_path: Path):
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

    assert result.returncode != 0
    stderr = result.stderr.lower()
    assert "paddleocr" in stderr or "required metrics artifact" in stderr or "pseudo_candidates.csv" in stderr


def test_run_training_pipeline_execute_mode_fails_when_teacher_pass_does_not_create_pseudo_candidates(
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

    monkeypatch.setattr(
        pipeline_module.stages,
        "teacher_pass_generate_pseudo_labels",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.data_dir / "pseudo_candidates.csv")},
    )

    with pytest.raises(FileNotFoundError, match="pseudo_candidates.csv"):
        run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)


def test_run_training_pipeline_persists_divergence_diagnostics_when_stage_a_fails(
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

    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    def _failing_stage_a(*args, **kwargs):
        raise RuntimeError("loss diverged with NaN")

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)
    monkeypatch.setattr(pipeline_module.stages, "stage_a_train_recognizer", _failing_stage_a)

    with pytest.raises(RuntimeError, match="loss diverged with NaN"):
        run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)

    diagnostics_path = config.logs_dir / "divergence_diagnostics.json"
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "stage_a_train_recognizer"
    assert "loss diverged with NaN" in payload["error"]
    assert payload["timestamp"]
    assert "command" in payload
    assert payload["context"]["processed_csv"] == str(processed_csv)


def test_run_training_pipeline_execute_mode_fails_when_required_metrics_artifacts_are_missing(
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

    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)
    monkeypatch.setattr(
        pipeline_module.stages,
        "stage_a_train_recognizer",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "recognition" / "recognizer.ckpt")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "stage_b_train_detector",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "detection" / "detector.ckpt")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "evaluate",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "evaluation_report.json")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "export",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "model_export.tar.gz")},
    )

    with pytest.raises(FileNotFoundError, match="val_predictions.csv|metrics.json|baseline_metrics.json"):
        run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)


def test_run_training_pipeline_execute_mode_uses_real_baseline_metrics_artifact(
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

    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    def _fake_stage_a(*args, **kwargs):
        _write_execute_metrics_artifacts(config)
        return {"status": "completed", "artifact_path": str(config.logs_dir / "recognition" / "recognizer.ckpt")}

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)
    monkeypatch.setattr(pipeline_module.stages, "stage_a_train_recognizer", _fake_stage_a)
    monkeypatch.setattr(
        pipeline_module.stages,
        "stage_b_train_detector",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "detection" / "detector.ckpt")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "evaluate",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "evaluation_report.json")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "export",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "model_export.tar.gz")},
    )

    report = run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)
    baseline_vs_best = json.loads(report["artifacts"]["baseline_comparison_path"].read_text(encoding="utf-8"))
    assert baseline_vs_best["baseline"] == {"cer": 0.5, "wer": 0.6}
    assert baseline_vs_best["best"]["cer"] != baseline_vs_best["baseline"]["cer"]
    assert baseline_vs_best["delta"]["cer"] == pytest.approx(
        baseline_vs_best["baseline"]["cer"] - baseline_vs_best["best"]["cer"]
    )


def test_run_training_pipeline_execute_mode_accepts_user_provided_baseline_metrics_file(
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
    baseline_metrics_path = tmp_path / "custom" / "baseline_metrics.json"
    baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_metrics_path.write_text(json.dumps({"cer": 0.8, "wer": 0.9}), encoding="utf-8")

    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    def _fake_stage_a(*args, **kwargs):
        _write_execute_metrics_artifacts(config)
        return {"status": "completed", "artifact_path": str(config.logs_dir / "recognition" / "recognizer.ckpt")}

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)
    monkeypatch.setattr(pipeline_module.stages, "stage_a_train_recognizer", _fake_stage_a)
    monkeypatch.setattr(
        pipeline_module.stages,
        "stage_b_train_detector",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "detection" / "detector.ckpt")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "evaluate",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "evaluation_report.json")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "export",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "model_export.tar.gz")},
    )

    report = run_training_pipeline(
        config=config,
        processed_csv=processed_csv,
        image_root=image_root,
        dry_run=False,
        baseline_metrics_path=baseline_metrics_path,
    )
    baseline_vs_best = json.loads(report["artifacts"]["baseline_comparison_path"].read_text(encoding="utf-8"))
    assert baseline_vs_best["baseline"] == {"cer": 0.8, "wer": 0.9}


def test_run_training_pipeline_execute_mode_errors_when_baseline_metrics_artifact_is_missing(
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

    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    def _fake_stage_a(*args, **kwargs):
        _write_execute_metrics_artifacts(config)
        (config.logs_dir / "baseline_metrics.json").unlink(missing_ok=True)
        return {"status": "completed", "artifact_path": str(config.logs_dir / "recognition" / "recognizer.ckpt")}

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)
    monkeypatch.setattr(pipeline_module.stages, "stage_a_train_recognizer", _fake_stage_a)
    monkeypatch.setattr(
        pipeline_module.stages,
        "stage_b_train_detector",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "detection" / "detector.ckpt")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "evaluate",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "evaluation_report.json")},
    )
    monkeypatch.setattr(
        pipeline_module.stages,
        "export",
        lambda **kwargs: {"status": "completed", "artifact_path": str(config.logs_dir / "model_export.tar.gz")},
    )

    with pytest.raises(FileNotFoundError, match="baseline_metrics.json"):
        run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)


def test_execute_evaluation_report_includes_per_language_and_source_segmented_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import src.training.pipeline as pipeline_module

    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
        pseudo_label=PseudoLabelConfig(confidence_threshold=0.8, max_pseudo_ratio_per_language=0.8),
        promotion_gate=PromotionGateConfig(
            max_source_cer_degradation=1.0,
            max_source_wer_degradation=1.0,
        ),
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    for image_name in ("pt_human.png", "pt_1.png", "pt_2.png", "en_human.png", "en_1.png", "en_2.png"):
        Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_root / image_name)
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_csv.write_text(
        "image_path,label_text,language,source_kind\n"
        "pt_human.png,texto humano,pt,human_label\n"
        "pt_1.png,texto 1,pt,human_label\n"
        "pt_2.png,texto 2,pt,human_label\n"
        "en_human.png,human text,en,human_label\n"
        "en_1.png,human text 1,en,human_label\n"
        "en_2.png,human text 2,en,human_label\n",
        encoding="utf-8",
    )

    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text(
            "image_path,prediction_text,confidence,language\n"
            "en_pseudo.png,pseudo text,0.95,en\n"
            "en_pseudo_1.png,pseudo text 1,0.96,en\n"
            "en_pseudo_2.png,pseudo text 2,0.97,en\n",
            encoding="utf-8",
        )
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    def _fake_stage_a(*args, **kwargs):
        _write_execute_metrics_artifacts_with_sources(config)
        recognizer_ckpt = config.logs_dir / "recognition" / "recognizer.ckpt"
        recognizer_ckpt.parent.mkdir(parents=True, exist_ok=True)
        recognizer_ckpt.write_text("recognizer", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(recognizer_ckpt)}

    def _fake_stage_b(*, manifest_train_csv: Path, detection_run_dir: Path, execute: bool):
        detector_ckpt = detection_run_dir / "detector.ckpt"
        detector_ckpt.parent.mkdir(parents=True, exist_ok=True)
        detector_ckpt.write_text("detector", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(detector_ckpt)}

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)
    monkeypatch.setattr(pipeline_module.stages, "stage_a_train_recognizer", _fake_stage_a)
    monkeypatch.setattr(pipeline_module.stages, "stage_b_train_detector", _fake_stage_b)

    report = run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)
    evaluation_payload = json.loads(report["artifacts"]["evaluation_report_path"].read_text(encoding="utf-8"))

    assert set(evaluation_payload["recognition"]["per_language"]) == {"pt", "en"}
    assert set(evaluation_payload["recognition"]["per_source_kind"]) == {"human_label", "pseudo_label"}
    assert "per_language" in evaluation_payload["recognition"]["per_source_kind"]["human_label"]
    assert "per_language" in evaluation_payload["recognition"]["per_source_kind"]["pseudo_label"]
    assert evaluation_payload["evaluation_mode"] == "execute"
    assert "dry_run_status" not in evaluation_payload
    assert set(evaluation_payload["baseline_vs_best"]) == {"baseline", "best", "delta"}
    assert set(evaluation_payload["baseline_vs_best"]["baseline"]) == {"cer", "wer"}
    assert set(evaluation_payload["baseline_vs_best"]["best"]) == {"cer", "wer"}


def test_execute_export_bundle_is_tar_gz_with_expected_artifacts(
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

    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    def _fake_stage_a(*args, **kwargs):
        _write_execute_metrics_artifacts(config)
        recognizer_ckpt = config.logs_dir / "recognition" / "recognizer.ckpt"
        recognizer_ckpt.parent.mkdir(parents=True, exist_ok=True)
        recognizer_ckpt.write_text("recognizer", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(recognizer_ckpt)}

    def _fake_stage_b(*, manifest_train_csv: Path, detection_run_dir: Path, execute: bool):
        detector_ckpt = detection_run_dir / "detector.ckpt"
        detector_ckpt.parent.mkdir(parents=True, exist_ok=True)
        detector_ckpt.write_text("detector", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(detector_ckpt)}

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _fake_teacher)
    monkeypatch.setattr(pipeline_module.stages, "stage_a_train_recognizer", _fake_stage_a)
    monkeypatch.setattr(pipeline_module.stages, "stage_b_train_detector", _fake_stage_b)

    report = run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)
    export_bundle = report["artifacts"]["export_bundle_path"]
    assert export_bundle.is_file()

    with tarfile.open(export_bundle, "r:gz") as archive:
        names = set(archive.getnames())

    assert "recognition/recognizer.ckpt" in names
    assert "detection/detector.ckpt" in names
    assert "metadata/metadata_bundle.json" in names
    assert "metrics/evaluation_report.json" in names
    assert "metrics/baseline_vs_best.json" in names
    assert "config/inference_config.json" in names
