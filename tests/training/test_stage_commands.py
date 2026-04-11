from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.training.config import GPUProfileConfig, StageAConfig, TrainingConfig
from src.training.pipeline import run_training_pipeline
from src.training.stages import build_stage_a_command


def _create_valid_dataset(processed_csv: Path, image_root: Path) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    rows = ["image_path,label_text,language,source_kind\n"]
    for language in ("pt", "en"):
        for idx in range(2):
            image_name = f"{language}_{idx}.png"
            Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_root / image_name)
            rows.append(f"{image_name},texto {idx},{language},human_label\n")
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_csv.write_text("".join(rows), encoding="utf-8")


def test_build_stage_a_command_contains_weighted_sampling_and_early_stopping(tmp_path: Path):
    command = build_stage_a_command(
        manifest_train_csv=tmp_path / "manifest_train.csv",
        recognition_run_dir=tmp_path / "logs" / "recognition",
        stage_a_cfg=StageAConfig(
            weighted_sampling_human_weight=3.0,
            weighted_sampling_pseudo_weight=0.7,
            early_stopping_patience=9,
            early_stopping_metric="wer",
        ),
    )

    assert "--weighted-sampling-human-weight" in command
    assert "3.0" in command
    assert "--weighted-sampling-pseudo-weight" in command
    assert "0.7" in command
    assert "--early-stopping-patience" in command
    assert "9" in command
    assert "--early-stopping-metric" in command
    assert "wer" in command


def test_execute_mode_fails_fast_when_required_gpu_profile_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = TrainingConfig(
        data_dir=tmp_path / "data" / "processed" / "training",
        logs_dir=tmp_path / "logs" / "training",
        gpu_profile=GPUProfileConfig(required=True, backend="cuda", min_devices=1),
    )
    processed_csv = config.data_dir / "merged.csv"
    image_root = tmp_path / "data" / "images"
    _create_valid_dataset(processed_csv, image_root)

    monkeypatch.setattr("src.training.stages.detect_available_gpus", lambda backend: [])

    with pytest.raises(RuntimeError, match="required GPU profile unavailable"):
        run_training_pipeline(
            config=config,
            processed_csv=processed_csv,
            image_root=image_root,
            dry_run=False,
        )
