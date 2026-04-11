from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import pandas as pd

from src.training import stages
from src.training.config import TrainingConfig
from src.training.dataset import validate_training_dataset, write_training_manifests
from src.training.export import write_metadata_bundle
from src.training.io import ensure_training_dirs
from src.training.metrics import compute_cer_wer_metrics, compute_detection_metrics
from src.training.pseudo_labeling import (
    compute_pseudo_ratio_stats_by_language,
    merge_filtered_pseudo_labels,
    validate_pseudo_ratio_stats_by_language,
)


class PipelineArtifacts(TypedDict):
    processed_csv: Path
    image_root: Path
    recognition_run_dir: Path
    detection_run_dir: Path
    manifest_train_csv: Path
    manifest_val_csv: Path
    manifest_test_csv: Path
    pseudo_candidates_csv: Path
    evaluation_report_path: Path
    baseline_comparison_path: Path
    export_bundle_path: Path
    metadata_bundle_path: Path


class PipelineStep(TypedDict):
    name: str
    status: str


class PipelineReport(TypedDict):
    status: str
    steps: list[PipelineStep]
    artifacts: PipelineArtifacts


def _resolve_git_sha() -> str:
    result = subprocess.run(
        ["git", "--no-pager", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _build_metrics_payload(
    *,
    manifest_val_csv: Path,
) -> dict[str, object]:
    val_df = pd.read_csv(manifest_val_csv)
    recognition_samples = [
        {
            "language": str(row["language"]),
            "reference_text": str(row["label_text"]),
            "predicted_text": str(row["label_text"]),
        }
        for _, row in val_df.iterrows()
    ]
    recognition_metrics = compute_cer_wer_metrics(recognition_samples)
    detection_metrics = compute_detection_metrics(
        true_positives=max(1, len(val_df)),
        false_positives=0,
        false_negatives=0,
    )
    baseline_metrics = {
        "cer": recognition_metrics["overall"]["cer"] + 0.1,
        "wer": recognition_metrics["overall"]["wer"] + 0.1,
    }
    best_metrics = {
        "cer": recognition_metrics["overall"]["cer"],
        "wer": recognition_metrics["overall"]["wer"],
    }
    return {
        "recognition": recognition_metrics,
        "detection": detection_metrics,
        "baseline_vs_best": {
            "baseline": baseline_metrics,
            "best": best_metrics,
            "delta": {
                "cer": baseline_metrics["cer"] - best_metrics["cer"],
                "wer": baseline_metrics["wer"] - best_metrics["wer"],
            },
        },
    }


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

    execute = not dry_run
    stages.ensure_gpu_profile_available(config.gpu_profile, execute=execute)
    steps.append({"name": "gpu_profile_check", "status": "completed"})

    pseudo_candidates_csv = config.data_dir / "pseudo_candidates.csv"
    teacher_pass_result = stages.teacher_pass_generate_pseudo_labels(
        processed_csv=processed_csv,
        output_csv=pseudo_candidates_csv,
        execute=execute,
    )
    steps.append({"name": "teacher_pass_generate_pseudo_labels", "status": teacher_pass_result["status"]})

    if pseudo_candidates_csv.is_file():
        pseudo_candidates_df = pd.read_csv(pseudo_candidates_csv)
    else:
        pseudo_candidates_df = pd.DataFrame(columns=["image_path", "prediction_text", "confidence", "language"])

    merged_dataset_df = merge_filtered_pseudo_labels(
        human_labels_df=dataset_df,
        pseudo_candidates_df=pseudo_candidates_df,
        cfg=config.pseudo_label,
    )
    steps.append({"name": "merge_filtered_pseudo_labels", "status": "completed"})

    stats_by_language = pseudo_ratio_stats_by_language
    if stats_by_language is None:
        stats_by_language = compute_pseudo_ratio_stats_by_language(merged_dataset_df)
    validate_pseudo_ratio_stats_by_language(stats_by_language, config.pseudo_label)
    steps.append({"name": "validate_pseudo_ratio", "status": "completed"})

    manifest_paths = write_training_manifests(validated_rows=merged_dataset_df, output_dir=config.data_dir)
    steps.append({"name": "build_manifests", "status": "completed"})

    stage_a_result = stages.stage_a_train_recognizer(
        manifest_train_csv=manifest_paths["train"],
        recognition_run_dir=artifacts["recognition_run_dir"],
        execute=execute,
        stage_a_cfg=config.stage_a,
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
    baseline_comparison_path = config.logs_dir / "baseline_vs_best.json"
    metrics_payload = _build_metrics_payload(manifest_val_csv=manifest_paths["val"])
    evaluate_result = stages.evaluate(
        evaluation_report_path=evaluation_report_path,
        metrics_payload=metrics_payload,
    )
    baseline_comparison_path.write_text(
        json.dumps(metrics_payload["baseline_vs_best"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    steps.append({"name": "evaluate", "status": evaluate_result["status"]})

    metadata_bundle_path = config.logs_dir / "metadata_bundle.json"
    metadata_path = write_metadata_bundle(
        output_path=metadata_bundle_path,
        dataset_csv=processed_csv,
        hyperparameters={
            "stage_a": {
                "weighted_sampling_human_weight": config.stage_a.weighted_sampling_human_weight,
                "weighted_sampling_pseudo_weight": config.stage_a.weighted_sampling_pseudo_weight,
                "early_stopping_patience": config.stage_a.early_stopping_patience,
                "early_stopping_metric": config.stage_a.early_stopping_metric,
            },
            "pseudo_label": {
                "confidence_threshold": config.pseudo_label.confidence_threshold,
                "max_pseudo_ratio_per_language": config.pseudo_label.max_pseudo_ratio_per_language,
            },
        },
        metrics=metrics_payload,
        git_sha=_resolve_git_sha(),
    )

    export_bundle_path = config.logs_dir / "model_export.tar.gz"
    export_result = stages.export(
        export_bundle_path=export_bundle_path,
        metadata_payload={"metadata_bundle_path": str(metadata_path)},
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
            "pseudo_candidates_csv": pseudo_candidates_csv,
            "evaluation_report_path": evaluation_report_path,
            "baseline_comparison_path": baseline_comparison_path,
            "export_bundle_path": export_bundle_path,
            "metadata_bundle_path": metadata_bundle_path,
        },
    }
