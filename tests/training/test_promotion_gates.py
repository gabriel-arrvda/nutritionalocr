from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.training.config import TrainingConfig
from src.training.pipeline import run_training_pipeline, select_stage_b_hard_examples


def _create_valid_dataset(processed_csv: Path, image_root: Path) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    rows = ["image_path,label_text,language,source_kind\n"]
    for language in ("pt", "en"):
        for idx in range(3):
            image_name = f"{language}_{idx}.png"
            Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_root / image_name)
            rows.append(f"{image_name},texto {idx},{language},human_label\n")
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_csv.write_text("".join(rows), encoding="utf-8")


def _stub_teacher(config: TrainingConfig):
    def _fake_teacher(*args, **kwargs):
        pseudo_candidates = config.data_dir / "pseudo_candidates.csv"
        pseudo_candidates.parent.mkdir(parents=True, exist_ok=True)
        pseudo_candidates.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(pseudo_candidates)}

    return _fake_teacher


def _stub_stage_b(config: TrainingConfig):
    def _fake_stage_b(*, manifest_train_csv: Path, detection_run_dir: Path, execute: bool):
        detector_ckpt = detection_run_dir / "detector.ckpt"
        detector_ckpt.parent.mkdir(parents=True, exist_ok=True)
        detector_ckpt.write_text("detector", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(detector_ckpt)}

    return _fake_stage_b


def _stub_stage_a_with_metrics(config: TrainingConfig, predictions_csv: str, baseline: dict[str, float]):
    def _fake_stage_a(*args, **kwargs):
        recognition_predictions = config.logs_dir / "recognition" / "val_predictions.csv"
        recognition_predictions.parent.mkdir(parents=True, exist_ok=True)
        recognition_predictions.write_text(predictions_csv, encoding="utf-8")
        detection_metrics = config.logs_dir / "detection" / "metrics.json"
        detection_metrics.parent.mkdir(parents=True, exist_ok=True)
        detection_metrics.write_text(
            json.dumps({"true_positives": 8, "false_positives": 2, "false_negatives": 1}),
            encoding="utf-8",
        )
        (config.logs_dir / "baseline_metrics.json").write_text(json.dumps(baseline), encoding="utf-8")
        recognizer_ckpt = config.logs_dir / "recognition" / "recognizer.ckpt"
        recognizer_ckpt.parent.mkdir(parents=True, exist_ok=True)
        recognizer_ckpt.write_text("recognizer", encoding="utf-8")
        return {"status": "completed", "artifact_path": str(recognizer_ckpt)}

    return _fake_stage_a


def test_run_training_pipeline_blocks_promotion_on_baseline_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import src.training.pipeline as pipeline_module

    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
        languages=("pt", "en"),
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    _create_valid_dataset(processed_csv, image_root)

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _stub_teacher(config))
    monkeypatch.setattr(
        pipeline_module.stages,
        "stage_a_train_recognizer",
        _stub_stage_a_with_metrics(
            config,
            predictions_csv=(
                "image_path,reference_text,predicted_text,language,source_kind\n"
                "pt_0.png,texto 0,errado total,pt,human_label\n"
                "en_0.png,texto 0,errado total,en,human_label\n"
            ),
            baseline={"cer": 0.05, "wer": 0.05},
        ),
    )
    monkeypatch.setattr(pipeline_module.stages, "stage_b_train_detector", _stub_stage_b(config))

    with pytest.raises(ValueError, match="Baseline regression gate failed"):
        run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)

    evaluation_payload = json.loads((config.logs_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert evaluation_payload["promotion_gates"]["baseline_regression"]["status"] == "failed"
    assert "Current best multilingual CER/WER" in evaluation_payload["promotion_gates"]["baseline_regression"]["message"]


def test_run_training_pipeline_blocks_promotion_on_source_segmentation_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import src.training.pipeline as pipeline_module

    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
        languages=("pt", "en"),
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    _create_valid_dataset(processed_csv, image_root)

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _stub_teacher(config))
    monkeypatch.setattr(
        pipeline_module.stages,
        "stage_a_train_recognizer",
        _stub_stage_a_with_metrics(
            config,
            predictions_csv=(
                "image_path,reference_text,predicted_text,language,source_kind\n"
                "pt_0.png,texto 0,texto 0,pt,human_label\n"
                "en_0.png,texto 0,texto 0,en,human_label\n"
                "pt_1.png,texto 1,xxxx xxxxx,pt,pseudo_label\n"
                "en_1.png,texto 1,xxxx xxxxx,en,pseudo_label\n"
            ),
            baseline={"cer": 0.9, "wer": 0.9},
        ),
    )
    monkeypatch.setattr(pipeline_module.stages, "stage_b_train_detector", _stub_stage_b(config))

    with pytest.raises(ValueError, match="Source-segmentation degradation gate failed"):
        run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)

    evaluation_payload = json.loads((config.logs_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert evaluation_payload["promotion_gates"]["source_segmentation_degradation"]["status"] == "failed"
    assert "pseudo_label validation metrics degraded" in evaluation_payload["promotion_gates"]["source_segmentation_degradation"]["message"]


def test_run_training_pipeline_uses_multilingual_macro_metrics_for_baseline_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import src.training.pipeline as pipeline_module

    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
        languages=("pt", "en"),
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    _create_valid_dataset(processed_csv, image_root)

    monkeypatch.setattr(pipeline_module.stages, "teacher_pass_generate_pseudo_labels", _stub_teacher(config))
    monkeypatch.setattr(
        pipeline_module.stages,
        "stage_a_train_recognizer",
        _stub_stage_a_with_metrics(
            config,
            predictions_csv=(
                "image_path,reference_text,predicted_text,language,source_kind\n"
                "pt_0.png,aaaaaaaaaaaaaaaaaaaa,aaaaaaaaaaaaaaaaaaaa,pt,human_label\n"
                "en_0.png,ok,xx,en,human_label\n"
            ),
            baseline={"cer": 0.2, "wer": 0.2},
        ),
    )
    monkeypatch.setattr(pipeline_module.stages, "stage_b_train_detector", _stub_stage_b(config))

    with pytest.raises(ValueError, match="Baseline regression gate failed"):
        run_training_pipeline(config=config, processed_csv=processed_csv, image_root=image_root, dry_run=False)

    evaluation_payload = json.loads((config.logs_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert evaluation_payload["promotion_gates"]["baseline_regression"]["status"] == "failed"
    assert "multilingual CER/WER regressed beyond baseline threshold" in evaluation_payload["promotion_gates"]["baseline_regression"]["message"]

def test_select_stage_b_hard_examples_prioritizes_stage_a_failures_and_low_confidence(tmp_path: Path):
    manifest_train_csv = tmp_path / "manifest_train.csv"
    manifest_train_csv.write_text(
        "image_path,label_text,language,source_kind\n"
        "a.png,alpha,pt,human_label\n"
        "b.png,beta,pt,human_label\n"
        "c.png,gamma,en,human_label\n"
        "d.png,delta,en,human_label\n",
        encoding="utf-8",
    )
    stage_a_predictions_csv = tmp_path / "stage_a_predictions.csv"
    stage_a_predictions_csv.write_text(
        "image_path,reference_text,predicted_text,language,confidence\n"
        "a.png,alpha,alpha,pt,0.95\n"
        "b.png,beta,erro,pt,0.98\n"
        "c.png,gamma,gamma,en,0.20\n"
        "d.png,delta,delta,en,0.99\n",
        encoding="utf-8",
    )
    output_manifest_csv = tmp_path / "manifest_stage_b_hard_examples.csv"

    written_path = select_stage_b_hard_examples(
        manifest_train_csv=manifest_train_csv,
        stage_a_artifact_path=stage_a_predictions_csv,
        output_manifest_csv=output_manifest_csv,
        hard_example_ratio=0.5,
    )

    assert written_path == output_manifest_csv
    hard_examples_df = pd.read_csv(output_manifest_csv)
    assert set(hard_examples_df["image_path"]) == {"b.png", "c.png"}
