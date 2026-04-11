from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

from src.training.config import GPUProfileConfig, StageAConfig


def _run_stage_command(*, command: list[str], execute: bool) -> dict[str, str]:
    if execute:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return {"status": "completed", "command": " ".join(command)}
    return {"status": "skipped_dry_run", "command": " ".join(command)}


def detect_available_gpus(backend: str) -> list[str]:
    if backend == "cuda":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if backend == "mps":
        mps_visible = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        return ["mps"] if mps_visible is not None else []

    return []


def ensure_gpu_profile_available(gpu_profile: GPUProfileConfig, *, execute: bool) -> None:
    if not execute or not gpu_profile.required:
        return
    available_devices = detect_available_gpus(gpu_profile.backend)
    if len(available_devices) < gpu_profile.min_devices:
        raise RuntimeError(
            "required GPU profile unavailable: "
            f"backend={gpu_profile.backend}, "
            f"required_devices={gpu_profile.min_devices}, "
            f"available_devices={len(available_devices)}"
        )


def build_teacher_pass_command(*, processed_csv: Path, output_csv: Path) -> list[str]:
    return [
        "python3",
        "-m",
        "paddleocr",
        "ocr",
        "--input-csv",
        str(processed_csv),
        "--output-csv",
        str(output_csv),
    ]


def teacher_pass_generate_pseudo_labels(
    *,
    processed_csv: Path,
    output_csv: Path,
    execute: bool,
) -> dict[str, str]:
    command = build_teacher_pass_command(processed_csv=processed_csv, output_csv=output_csv)
    result = _run_stage_command(command=command, execute=execute)
    if not execute:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_csv.write_text("image_path,prediction_text,confidence,language\n", encoding="utf-8")
    result["artifact_path"] = str(output_csv)
    return result


def build_stage_a_command(*, manifest_train_csv: Path, recognition_run_dir: Path, stage_a_cfg: StageAConfig) -> list[str]:
    return [
        "python3",
        "-m",
        "paddleocr.tools.train",
        "--task",
        "recognition",
        "--manifest",
        str(manifest_train_csv),
        "--output-dir",
        str(recognition_run_dir),
        "--weighted-sampling-human-weight",
        str(stage_a_cfg.weighted_sampling_human_weight),
        "--weighted-sampling-pseudo-weight",
        str(stage_a_cfg.weighted_sampling_pseudo_weight),
        "--early-stopping-patience",
        str(stage_a_cfg.early_stopping_patience),
        "--early-stopping-metric",
        stage_a_cfg.early_stopping_metric,
    ]


def stage_a_train_recognizer(
    *,
    manifest_train_csv: Path,
    recognition_run_dir: Path,
    execute: bool,
    stage_a_cfg: StageAConfig,
) -> dict[str, str]:
    command = build_stage_a_command(
        manifest_train_csv=manifest_train_csv,
        recognition_run_dir=recognition_run_dir,
        stage_a_cfg=stage_a_cfg,
    )
    result = _run_stage_command(command=command, execute=execute)
    result["artifact_path"] = str(recognition_run_dir / "recognizer.ckpt")
    return result


def stage_b_train_detector(*, manifest_train_csv: Path, detection_run_dir: Path, execute: bool) -> dict[str, str]:
    command = [
        "python3",
        "-m",
        "paddleocr.tools.train",
        "--task",
        "detection",
        "--manifest",
        str(manifest_train_csv),
        "--output-dir",
        str(detection_run_dir),
    ]
    result = _run_stage_command(command=command, execute=execute)
    result["artifact_path"] = str(detection_run_dir / "detector.ckpt")
    return result


def evaluate(
    *,
    evaluation_report_path: Path,
    metrics_payload: dict[str, object],
) -> dict[str, str]:
    evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_report_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "completed", "artifact_path": str(evaluation_report_path)}


def export(
    *,
    export_bundle_path: Path,
    recognition_checkpoint_path: Path,
    detection_checkpoint_path: Path,
    metadata_bundle_path: Path,
    evaluation_report_path: Path,
    baseline_comparison_path: Path,
    training_config_path: Path,
) -> dict[str, str]:
    required_paths = (
        recognition_checkpoint_path,
        detection_checkpoint_path,
        metadata_bundle_path,
        evaluation_report_path,
        baseline_comparison_path,
        training_config_path,
    )
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"required export artifact(s) missing: {', '.join(missing_paths)}")

    export_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(export_bundle_path, "w:gz") as archive:
        archive.add(recognition_checkpoint_path, arcname="recognition/recognizer.ckpt")
        archive.add(detection_checkpoint_path, arcname="detection/detector.ckpt")
        archive.add(metadata_bundle_path, arcname="metadata/metadata_bundle.json")
        archive.add(evaluation_report_path, arcname="metrics/evaluation_report.json")
        archive.add(baseline_comparison_path, arcname="metrics/baseline_vs_best.json")
        archive.add(training_config_path, arcname="config/training_config.json")
    return {"status": "completed", "artifact_path": str(export_bundle_path)}
