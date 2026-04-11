from __future__ import annotations

import hashlib
from pathlib import Path
from collections.abc import Mapping
from typing import TypedDict

import pandas as pd

from src.training.config import TrainingConfig
from src.training.dataset import validate_training_dataset, write_training_manifests
from src.training.io import ensure_training_dirs
from src.training.pseudo_labeling import (
    compute_pseudo_ratio_stats_by_language,
    validate_pseudo_ratio_stats_by_language,
)
from src.training import stages


class PipelineArtifacts(TypedDict):
    processed_csv: Path
    image_root: Path
    recognition_run_dir: Path
    detection_run_dir: Path
    manifest_train_csv: Path
    manifest_val_csv: Path
    manifest_test_csv: Path
    evaluation_report_path: Path
    export_bundle_path: Path


class PipelineStep(TypedDict):
    name: str
    status: str


class PipelineReport(TypedDict):
    status: str
    steps: list[PipelineStep]
    artifacts: PipelineArtifacts


def select_stage_b_hard_examples(
    *,
    manifest_train_csv: Path,
    stage_a_artifact_path: Path,
    output_manifest_csv: Path,
    hard_example_ratio: float = 0.5,
) -> Path:
    if not 0 < hard_example_ratio <= 1:
        raise ValueError("hard_example_ratio must satisfy 0 < hard_example_ratio <= 1")

    manifest_df = pd.read_csv(manifest_train_csv)
    if manifest_df.empty:
        raise ValueError("manifest_train_csv cannot be empty")

    stage_a_signal = (
        stage_a_artifact_path.read_text(encoding="utf-8")
        if stage_a_artifact_path.is_file()
        else str(stage_a_artifact_path)
    )

    def _score_row(row: pd.Series) -> int:
        payload = (
            f"{stage_a_signal}|{row['image_path']}|{row['label_text']}|"
            f"{row['language']}|{row['source_kind']}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(digest[:15], 16)

    scored_df = manifest_df.copy()
    scored_df["_hard_score"] = scored_df.apply(_score_row, axis=1)
    hard_example_count = max(1, int(len(scored_df) * hard_example_ratio))
    hard_examples_df = scored_df.nlargest(hard_example_count, "_hard_score").drop(columns=["_hard_score"])
    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    hard_examples_df.to_csv(output_manifest_csv, index=False)
    return output_manifest_csv


def run_training_pipeline(
    config: TrainingConfig,
    processed_csv: Path,
    image_root: Path,
    dry_run: bool = False,
    pseudo_ratio_stats_by_language: Mapping[str, Mapping[str, int | float]] | None = None,
) -> PipelineReport:
    steps: list[PipelineStep] = []
    artifacts = ensure_training_dirs(logs_dir=config.logs_dir, data_dir=config.data_dir)
    validation_report, dataset_df = validate_training_dataset(
        config=config,
        processed_csv=processed_csv,
        image_root=image_root,
    )
    steps.append({"name": "validate_dataset", "status": "completed" if not validation_report["errors"] else "failed"})

    if validation_report["errors"] or dataset_df is None:
        raise ValueError("; ".join(validation_report["errors"]))

    stats_by_language = pseudo_ratio_stats_by_language
    if stats_by_language is None:
        stats_by_language = compute_pseudo_ratio_stats_by_language(dataset_df)

    validate_pseudo_ratio_stats_by_language(stats_by_language, config.pseudo_label)
    manifest_paths = write_training_manifests(validated_rows=dataset_df, output_dir=config.data_dir)
    steps.append({"name": "build_manifests", "status": "completed"})

    execute = not dry_run
    stage_a_result = stages.stage_a_train_recognizer(
        manifest_train_csv=manifest_paths["train"],
        recognition_run_dir=artifacts["recognition_run_dir"],
        execute=execute,
    )
    steps.append({"name": "stage_a_train_recognizer", "status": stage_a_result["status"]})

    stage_b_manifest_csv = select_stage_b_hard_examples(
        manifest_train_csv=manifest_paths["train"],
        stage_a_artifact_path=Path(stage_a_result["artifact_path"]),
        output_manifest_csv=config.data_dir / "manifest_stage_b_hard_examples.csv",
    )

    stage_b_result = stages.stage_b_train_detector(
        manifest_train_csv=stage_b_manifest_csv,
        detection_run_dir=artifacts["detection_run_dir"],
        execute=execute,
    )
    steps.append({"name": "stage_b_train_detector", "status": stage_b_result["status"]})

    evaluation_report_path = config.logs_dir / "evaluation_report.json"
    evaluate_result = stages.evaluate(
        manifest_val_csv=manifest_paths["val"],
        recognition_run_dir=artifacts["recognition_run_dir"],
        detection_run_dir=artifacts["detection_run_dir"],
        evaluation_report_path=evaluation_report_path,
        execute=execute,
    )
    steps.append({"name": "evaluate", "status": evaluate_result["status"]})

    export_bundle_path = config.logs_dir / "model_export.tar.gz"
    export_result = stages.export(
        recognition_run_dir=artifacts["recognition_run_dir"],
        detection_run_dir=artifacts["detection_run_dir"],
        export_bundle_path=export_bundle_path,
        execute=execute,
    )
    steps.append({"name": "export", "status": export_result["status"]})

    return {
        "status": "dry_run_ready" if dry_run else "completed",
        "steps": steps,
        "artifacts": {
            "processed_csv": processed_csv,
            "image_root": image_root,
            "recognition_run_dir": artifacts["recognition_run_dir"],
            "detection_run_dir": artifacts["detection_run_dir"],
            "manifest_train_csv": manifest_paths["train"],
            "manifest_val_csv": manifest_paths["val"],
            "manifest_test_csv": manifest_paths["test"],
            "evaluation_report_path": evaluation_report_path,
            "export_bundle_path": export_bundle_path,
        },
    }
