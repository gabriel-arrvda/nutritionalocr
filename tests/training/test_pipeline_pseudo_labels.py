from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.training.config import PseudoLabelConfig, TrainingConfig
from src.training.pipeline import run_training_pipeline


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


def test_pipeline_runs_teacher_pass_and_merges_filtered_pseudo_labels_before_stage_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import src.training.pipeline as pipeline_module

    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
        pseudo_label=PseudoLabelConfig(confidence_threshold=0.9, max_pseudo_ratio_per_language=0.5),
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    _create_valid_dataset(processed_csv, image_root)

    pseudo_candidates_csv = config.data_dir / "pseudo_candidates.csv"
    pseudo_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"image_path": "pseudo_high_1.png", "prediction_text": "texto alto 1", "confidence": 0.95, "language": "pt"},
            {"image_path": "pseudo_high_2.png", "prediction_text": "texto alto 2", "confidence": 0.93, "language": "pt"},
            {"image_path": "pseudo_high_3.png", "prediction_text": "texto alto 3", "confidence": 0.92, "language": "pt"},
            {"image_path": "pseudo_low.png", "prediction_text": "texto baixo", "confidence": 0.5, "language": "pt"},
        ]
    ).to_csv(pseudo_candidates_csv, index=False)

    captured_manifest_df: pd.DataFrame | None = None

    monkeypatch.setattr(
        pipeline_module.stages,
        "teacher_pass_generate_pseudo_labels",
        lambda **kwargs: {"status": "completed", "artifact_path": str(pseudo_candidates_csv)},
    )

    def _fake_stage_a(*, manifest_train_csv: Path, recognition_run_dir: Path, execute: bool, stage_a_cfg):
        nonlocal captured_manifest_df
        captured_manifest_df = pd.read_csv(manifest_train_csv)
        return {"status": "completed", "artifact_path": str(recognition_run_dir / "recognizer.ckpt")}

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

    assert report["status"] == "completed"
    step_names = [step["name"] for step in report["steps"]]
    assert step_names.index("teacher_pass_generate_pseudo_labels") < step_names.index("stage_a_train_recognizer")
    assert step_names.index("merge_filtered_pseudo_labels") < step_names.index("stage_a_train_recognizer")
    assert captured_manifest_df is not None
    pseudo_rows = captured_manifest_df[captured_manifest_df["source_kind"] == "pseudo_label"]
    assert len(pseudo_rows) >= 1
    assert "texto baixo" not in set(pseudo_rows["label_text"])
