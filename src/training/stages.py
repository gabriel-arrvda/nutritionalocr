from __future__ import annotations

import subprocess
from pathlib import Path


def _run_stage_command(*, command: list[str], execute: bool) -> dict[str, str]:
    if execute:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return {"status": "completed", "command": " ".join(command)}
    return {"status": "skipped_dry_run", "command": " ".join(command)}


def stage_a_train_recognizer(*, manifest_train_csv: Path, recognition_run_dir: Path, execute: bool) -> dict[str, str]:
    command = [
        "python3",
        "-m",
        "src.training.runners",
        "stage-a",
        "--manifest",
        str(manifest_train_csv),
        "--output-dir",
        str(recognition_run_dir),
    ]
    result = _run_stage_command(command=command, execute=execute)
    result["artifact_path"] = str(recognition_run_dir / "recognizer.ckpt")
    return result


def stage_b_train_detector(*, manifest_train_csv: Path, detection_run_dir: Path, execute: bool) -> dict[str, str]:
    command = [
        "python3",
        "-m",
        "src.training.runners",
        "stage-b",
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
    manifest_val_csv: Path,
    recognition_run_dir: Path,
    detection_run_dir: Path,
    evaluation_report_path: Path,
    execute: bool,
) -> dict[str, str]:
    command = [
        "python3",
        "-m",
        "src.training.runners",
        "evaluate",
        "--manifest",
        str(manifest_val_csv),
        "--recognizer-dir",
        str(recognition_run_dir),
        "--detector-dir",
        str(detection_run_dir),
        "--output",
        str(evaluation_report_path),
    ]
    result = _run_stage_command(command=command, execute=execute)
    result["artifact_path"] = str(evaluation_report_path)
    return result


def export(*, recognition_run_dir: Path, detection_run_dir: Path, export_bundle_path: Path, execute: bool) -> dict[str, str]:
    command = [
        "python3",
        "-m",
        "src.training.runners",
        "export",
        "--recognizer-dir",
        str(recognition_run_dir),
        "--detector-dir",
        str(detection_run_dir),
        "--output",
        str(export_bundle_path),
    ]
    result = _run_stage_command(command=command, execute=execute)
    result["artifact_path"] = str(export_bundle_path)
    return result
