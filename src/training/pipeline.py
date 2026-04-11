from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import pandas as pd

from src.training import stages
from src.training.config import PromotionGateConfig, TrainingConfig
from src.training.dataset import validate_training_dataset, write_training_manifests
from src.training.export import write_metadata_bundle
from src.training.io import ensure_training_dirs
from src.training.metrics import compute_cer_wer_metrics, compute_detection_metrics
from src.training.pseudo_labeling import (
    compute_pseudo_ratio_stats_by_language,
    compute_pseudo_ratio_stats_by_language_source_bucket,
    merge_filtered_pseudo_labels,
    validate_pseudo_ratio_stats_by_language,
    validate_pseudo_ratio_stats_by_language_source_bucket,
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
    inference_config_path: Path


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


def _resolve_multilingual_macro_metrics(recognition_metrics: Mapping[str, object]) -> dict[str, float]:
    per_language = recognition_metrics.get("per_language", {})
    if not isinstance(per_language, Mapping) or not per_language:
        return {"cer": 0.0, "wer": 0.0}

    cer_values: list[float] = []
    wer_values: list[float] = []
    for language_metrics in per_language.values():
        if not isinstance(language_metrics, Mapping):
            continue
        if "cer" not in language_metrics or "wer" not in language_metrics:
            continue
        cer_values.append(float(language_metrics["cer"]))
        wer_values.append(float(language_metrics["wer"]))

    if not cer_values or not wer_values:
        return {"cer": 0.0, "wer": 0.0}
    return {
        "cer": sum(cer_values) / len(cer_values),
        "wer": sum(wer_values) / len(wer_values),
    }


def _write_divergence_diagnostics(
    *,
    logs_dir: Path,
    stage_name: str,
    error: Exception,
    context: Mapping[str, str],
) -> Path:
    diagnostics_path = logs_dir / "divergence_diagnostics.json"
    command: str | None = None
    if isinstance(error, subprocess.CalledProcessError):
        if isinstance(error.cmd, (list, tuple)):
            command = " ".join(str(item) for item in error.cmd)
        else:
            command = str(error.cmd)
    payload = {
        "stage": stage_name,
        "error": str(error),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "context": dict(context),
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return diagnostics_path


def _build_metrics_payload(
    *,
    recognition_predictions_csv: Path,
    detection_metrics_json: Path,
    baseline_metrics_json: Path,
    manifest_val_csv: Path,
    execute: bool,
) -> dict[str, object]:
    if execute:
        missing_paths = [
            str(path)
            for path in (recognition_predictions_csv, detection_metrics_json, baseline_metrics_json)
            if not path.is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(f"required metrics artifact(s) missing: {', '.join(missing_paths)}")
    elif not (
        recognition_predictions_csv.is_file()
        and detection_metrics_json.is_file()
        and baseline_metrics_json.is_file()
    ):
        return {
            "recognition": {
                "overall": {"cer": 0.0, "wer": 0.0},
                "macro": {"cer": 0.0, "wer": 0.0},
                "per_language": {},
                "per_source_kind": {},
            },
            "detection": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "baseline_vs_best": {
                "baseline": {},
                "best": {},
                "delta": {},
            },
            "dry_run_status": "skipped_dry_run",
        }

    recognition_df = pd.read_csv(recognition_predictions_csv)
    recognition_samples = []
    for _, row in recognition_df.iterrows():
        recognition_samples.append(
            {
                "language": str(row["language"]),
                "reference_text": str(row["reference_text"]),
                "predicted_text": str(row["predicted_text"]),
            }
        )
    recognition_metrics = compute_cer_wer_metrics(recognition_samples)
    recognition_metrics["macro"] = _resolve_multilingual_macro_metrics(recognition_metrics)
    if "source_kind" not in recognition_df.columns and manifest_val_csv.is_file():
        manifest_df = pd.read_csv(manifest_val_csv)
        if {"image_path", "language", "source_kind"}.issubset(manifest_df.columns):
            recognition_df = recognition_df.merge(
                manifest_df[["image_path", "language", "source_kind"]],
                on=["image_path", "language"],
                how="left",
            )
    per_source_kind: dict[str, object] = {}
    if "source_kind" in recognition_df.columns:
        source_df = recognition_df.dropna(subset=["source_kind"])
        for source_kind, source_rows in source_df.groupby("source_kind"):
            source_samples = []
            for _, row in source_rows.iterrows():
                source_samples.append(
                    {
                        "language": str(row["language"]),
                        "reference_text": str(row["reference_text"]),
                        "predicted_text": str(row["predicted_text"]),
                    }
                )
            per_source_kind[str(source_kind)] = compute_cer_wer_metrics(source_samples)
    recognition_metrics["per_source_kind"] = per_source_kind
    detection_payload = json.loads(detection_metrics_json.read_text(encoding="utf-8"))
    if {"precision", "recall", "f1"}.issubset(detection_payload):
        detection_metrics = {
            "precision": float(detection_payload["precision"]),
            "recall": float(detection_payload["recall"]),
            "f1": float(detection_payload["f1"]),
        }
    else:
        detection_metrics = compute_detection_metrics(
            true_positives=int(detection_payload["true_positives"]),
            false_positives=int(detection_payload["false_positives"]),
            false_negatives=int(detection_payload["false_negatives"]),
        )
    baseline_payload = json.loads(baseline_metrics_json.read_text(encoding="utf-8"))
    baseline_metrics = {"cer": float(baseline_payload["cer"]), "wer": float(baseline_payload["wer"])}
    best_metrics = {
        "cer": float(recognition_metrics["macro"]["cer"]),
        "wer": float(recognition_metrics["macro"]["wer"]),
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


def _resolve_stage_a_predictions_path(stage_a_artifact_path: Path) -> Path:
    if stage_a_artifact_path.is_file() and stage_a_artifact_path.suffix == ".csv":
        return stage_a_artifact_path
    return stage_a_artifact_path.parent / "val_predictions.csv"


def _evaluate_promotion_gates(
    *,
    metrics_payload: dict[str, object],
    gate_cfg: PromotionGateConfig,
) -> tuple[dict[str, object], list[str]]:
    gate_failures: list[str] = []
    baseline_vs_best = metrics_payload.get("baseline_vs_best", {})
    if not isinstance(baseline_vs_best, Mapping):
        baseline_vs_best = {}
    baseline_metrics = baseline_vs_best.get("baseline", {})
    best_metrics = baseline_vs_best.get("best", {})

    baseline_gate: dict[str, object]
    if (
        isinstance(baseline_metrics, Mapping)
        and isinstance(best_metrics, Mapping)
        and {"cer", "wer"}.issubset(baseline_metrics)
        and {"cer", "wer"}.issubset(best_metrics)
    ):
        baseline_cer = float(baseline_metrics["cer"])
        baseline_wer = float(baseline_metrics["wer"])
        best_cer = float(best_metrics["cer"])
        best_wer = float(best_metrics["wer"])
        cer_regression = best_cer - baseline_cer
        wer_regression = best_wer - baseline_wer
        if (
            cer_regression > gate_cfg.max_baseline_cer_regression
            or wer_regression > gate_cfg.max_baseline_wer_regression
        ):
            message = (
                "Current best multilingual CER/WER regressed beyond baseline threshold "
                f"(cer_regression={cer_regression:.6f}, wer_regression={wer_regression:.6f}, "
                f"allowed_cer={gate_cfg.max_baseline_cer_regression:.6f}, "
                f"allowed_wer={gate_cfg.max_baseline_wer_regression:.6f})"
            )
            baseline_gate = {"status": "failed", "message": message}
            gate_failures.append(f"Baseline regression gate failed: {message}")
        else:
            baseline_gate = {"status": "passed", "message": "Best multilingual CER/WER is within baseline threshold."}
    else:
        baseline_gate = {"status": "skipped", "message": "baseline_vs_best metrics unavailable for gate evaluation."}

    source_gate: dict[str, object]
    recognition_metrics = metrics_payload.get("recognition", {})
    if not isinstance(recognition_metrics, Mapping):
        recognition_metrics = {}
    per_source_kind = recognition_metrics.get("per_source_kind", {})
    if (
        isinstance(per_source_kind, Mapping)
        and "human_label" in per_source_kind
        and "pseudo_label" in per_source_kind
        and isinstance(per_source_kind["human_label"], Mapping)
        and isinstance(per_source_kind["pseudo_label"], Mapping)
    ):
        human_metrics = per_source_kind["human_label"].get("overall", {})
        pseudo_metrics = per_source_kind["pseudo_label"].get("overall", {})
        if (
            isinstance(human_metrics, Mapping)
            and isinstance(pseudo_metrics, Mapping)
            and {"cer", "wer"}.issubset(human_metrics)
            and {"cer", "wer"}.issubset(pseudo_metrics)
        ):
            cer_degradation = float(pseudo_metrics["cer"]) - float(human_metrics["cer"])
            wer_degradation = float(pseudo_metrics["wer"]) - float(human_metrics["wer"])
            if (
                cer_degradation > gate_cfg.max_source_cer_degradation
                or wer_degradation > gate_cfg.max_source_wer_degradation
            ):
                message = (
                    "pseudo_label validation metrics degraded beyond configured threshold "
                    f"(cer_degradation={cer_degradation:.6f}, wer_degradation={wer_degradation:.6f}, "
                    f"allowed_cer={gate_cfg.max_source_cer_degradation:.6f}, "
                    f"allowed_wer={gate_cfg.max_source_wer_degradation:.6f})"
                )
                source_gate = {"status": "failed", "message": message}
                gate_failures.append(f"Source-segmentation degradation gate failed: {message}")
            else:
                source_gate = {
                    "status": "passed",
                    "message": "pseudo_label validation metrics are within degradation threshold.",
                }
        else:
            source_gate = {"status": "skipped", "message": "Source-kind overall CER/WER metrics unavailable."}
    else:
        source_gate = {"status": "skipped", "message": "human_label and pseudo_label metrics unavailable."}

    overall_status = "failed" if gate_failures else "passed"
    promotion_gates = {
        "status": overall_status,
        "baseline_regression": baseline_gate,
        "source_segmentation_degradation": source_gate,
    }
    return promotion_gates, gate_failures


def select_stage_b_hard_examples(
    *,
    manifest_train_csv: Path,
    stage_a_artifact_path: Path,
    output_manifest_csv: Path,
    hard_example_ratio: float = 0.5,
    stage_a_failure_confidence_threshold: float = 0.7,
    require_stage_a_failures: bool = False,
) -> Path:
    if not 0 < hard_example_ratio <= 1:
        raise ValueError("hard_example_ratio must satisfy 0 < hard_example_ratio <= 1")
    if not 0 <= stage_a_failure_confidence_threshold <= 1:
        raise ValueError("stage_a_failure_confidence_threshold must be between 0 and 1")

    manifest_df = pd.read_csv(manifest_train_csv)
    if manifest_df.empty:
        raise ValueError("manifest_train_csv cannot be empty")

    stage_a_predictions_path = _resolve_stage_a_predictions_path(stage_a_artifact_path)
    if not stage_a_predictions_path.is_file():
        if require_stage_a_failures:
            raise FileNotFoundError(
                "stage_a recognition failures unavailable: expected predictions artifact at "
                f"{stage_a_predictions_path}"
            )
        scored_df = manifest_df.copy()
        scored_df["_failure_mismatch"] = 0
        scored_df["_failure_low_confidence"] = 0
        scored_df["_confidence_sort"] = 1.0
    else:
        stage_a_predictions_df = pd.read_csv(stage_a_predictions_path)
        required_prediction_columns = {"image_path", "reference_text", "predicted_text"}
        if not required_prediction_columns.issubset(stage_a_predictions_df.columns):
            raise ValueError(
                "stage_a predictions missing required columns: "
                f"{', '.join(sorted(required_prediction_columns - set(stage_a_predictions_df.columns)))}"
            )
        stage_a_predictions_df = stage_a_predictions_df.copy()
        if "language" not in stage_a_predictions_df.columns:
            stage_a_predictions_df["language"] = ""
        if "confidence" not in stage_a_predictions_df.columns:
            stage_a_predictions_df["confidence"] = 1.0
        stage_a_predictions_df["_failure_mismatch"] = (
            stage_a_predictions_df["reference_text"].astype(str) != stage_a_predictions_df["predicted_text"].astype(str)
        ).astype(int)
        stage_a_predictions_df["_failure_low_confidence"] = (
            stage_a_predictions_df["confidence"].astype(float) < stage_a_failure_confidence_threshold
        ).astype(int)
        stage_a_predictions_df["_confidence_sort"] = stage_a_predictions_df["confidence"].astype(float)
        scored_df = manifest_df.merge(
            stage_a_predictions_df[
                [
                    "image_path",
                    "language",
                    "_failure_mismatch",
                    "_failure_low_confidence",
                    "_confidence_sort",
                ]
            ],
            on=["image_path", "language"],
            how="left",
        )
        scored_df["_failure_mismatch"] = scored_df["_failure_mismatch"].fillna(0).astype(int)
        scored_df["_failure_low_confidence"] = scored_df["_failure_low_confidence"].fillna(0).astype(int)
        scored_df["_confidence_sort"] = scored_df["_confidence_sort"].fillna(1.0).astype(float)

    scored_df["_hard_rank"] = range(len(scored_df))
    hard_example_count = max(1, int(len(scored_df) * hard_example_ratio))
    hard_examples_df = (
        scored_df.sort_values(
            by=["_failure_mismatch", "_failure_low_confidence", "_confidence_sort", "_hard_rank"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .head(hard_example_count)
        .drop(columns=["_failure_mismatch", "_failure_low_confidence", "_confidence_sort", "_hard_rank"])
    )
    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    hard_examples_df.to_csv(output_manifest_csv, index=False)
    return output_manifest_csv


def run_training_pipeline(
    config: TrainingConfig,
    processed_csv: Path,
    image_root: Path,
    dry_run: bool = False,
    pseudo_ratio_stats_by_language: Mapping[str, Mapping[str, int | float]] | None = None,
    baseline_metrics_path: Path | None = None,
) -> PipelineReport:
    steps: list[PipelineStep] = []
    artifacts = ensure_training_dirs(logs_dir=config.logs_dir, data_dir=config.data_dir)
    validation_report, dataset_df = validate_training_dataset(
        config=config,
        processed_csv=processed_csv,
        image_root=image_root,
        allow_unlabeled_rows=True,
    )
    steps.append({"name": "validate_dataset", "status": "completed" if not validation_report["errors"] else "failed"})

    if validation_report["errors"] or dataset_df is None:
        raise ValueError("; ".join(validation_report["errors"]))

    labeled_mask = dataset_df["label_text"].notna() & dataset_df["label_text"].astype(str).str.strip().ne("")
    labeled_dataset_df = dataset_df.loc[labeled_mask].copy()

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
    if execute and not pseudo_candidates_csv.is_file():
        raise FileNotFoundError(f"required artifact missing after teacher pass: {pseudo_candidates_csv}")

    if pseudo_candidates_csv.is_file():
        pseudo_candidates_df = pd.read_csv(pseudo_candidates_csv)
    else:
        pseudo_candidates_df = pd.DataFrame(columns=["image_path", "prediction_text", "confidence", "language"])

    merged_dataset_df = merge_filtered_pseudo_labels(
        human_labels_df=labeled_dataset_df,
        pseudo_candidates_df=pseudo_candidates_df,
        cfg=config.pseudo_label,
    )
    steps.append({"name": "merge_filtered_pseudo_labels", "status": "completed"})

    if merged_dataset_df["label_text"].isna().any() or merged_dataset_df["label_text"].astype(str).str.strip().eq("").any():
        raise ValueError("required field 'label_text' must be non-empty")
    if merged_dataset_df.empty:
        raise ValueError("required field 'label_text' must be non-empty")

    stats_by_language = pseudo_ratio_stats_by_language
    if stats_by_language is None:
        stats_by_language = compute_pseudo_ratio_stats_by_language(merged_dataset_df)
    validate_pseudo_ratio_stats_by_language(stats_by_language, config.pseudo_label)
    stats_by_bucket = compute_pseudo_ratio_stats_by_language_source_bucket(merged_dataset_df)
    validate_pseudo_ratio_stats_by_language_source_bucket(stats_by_bucket, config.pseudo_label)
    steps.append({"name": "validate_pseudo_ratio", "status": "completed"})

    manifest_paths = write_training_manifests(validated_rows=merged_dataset_df, output_dir=config.data_dir)
    steps.append({"name": "build_manifests", "status": "completed"})

    try:
        stage_a_result = stages.stage_a_train_recognizer(
            manifest_train_csv=manifest_paths["train"],
            recognition_run_dir=artifacts["recognition_run_dir"],
            execute=execute,
            stage_a_cfg=config.stage_a,
        )
    except Exception as exc:
        _write_divergence_diagnostics(
            logs_dir=config.logs_dir,
            stage_name="stage_a_train_recognizer",
            error=exc,
            context={
                "processed_csv": str(processed_csv),
                "manifest_train_csv": str(manifest_paths["train"]),
            },
        )
        raise
    steps.append({"name": "stage_a_train_recognizer", "status": stage_a_result["status"]})

    stage_b_manifest_csv = select_stage_b_hard_examples(
        manifest_train_csv=manifest_paths["train"],
        stage_a_artifact_path=Path(stage_a_result["artifact_path"]),
        output_manifest_csv=config.data_dir / "manifest_stage_b_hard_examples.csv",
        stage_a_failure_confidence_threshold=config.promotion_gate.hard_example_confidence_threshold,
        require_stage_a_failures=execute,
    )

    try:
        stage_b_result = stages.stage_b_train_detector(
            manifest_train_csv=stage_b_manifest_csv,
            detection_run_dir=artifacts["detection_run_dir"],
            execute=execute,
        )
    except Exception as exc:
        _write_divergence_diagnostics(
            logs_dir=config.logs_dir,
            stage_name="stage_b_train_detector",
            error=exc,
            context={
                "processed_csv": str(processed_csv),
                "manifest_train_csv": str(stage_b_manifest_csv),
            },
        )
        raise
    steps.append({"name": "stage_b_train_detector", "status": stage_b_result["status"]})

    evaluation_report_path = config.logs_dir / "evaluation_report.json"
    baseline_comparison_path = config.logs_dir / "baseline_vs_best.json"
    recognition_predictions_csv = artifacts["recognition_run_dir"] / "val_predictions.csv"
    detection_metrics_json = artifacts["detection_run_dir"] / "metrics.json"
    resolved_baseline_metrics_path = baseline_metrics_path or (config.logs_dir / "baseline_metrics.json")
    metrics_payload = _build_metrics_payload(
        recognition_predictions_csv=recognition_predictions_csv,
        detection_metrics_json=detection_metrics_json,
        baseline_metrics_json=resolved_baseline_metrics_path,
        manifest_val_csv=manifest_paths["val"],
        execute=execute,
    )
    promotion_gates_payload, promotion_gate_failures = _evaluate_promotion_gates(
        metrics_payload=metrics_payload,
        gate_cfg=config.promotion_gate,
    )
    metrics_payload["promotion_gates"] = promotion_gates_payload
    evaluate_result = stages.evaluate(
        evaluation_report_path=evaluation_report_path,
        metrics_payload=metrics_payload,
    )
    baseline_comparison_path.write_text(
        json.dumps(metrics_payload["baseline_vs_best"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    steps.append({"name": "evaluate", "status": evaluate_result["status"]})
    if promotion_gate_failures:
        raise ValueError("; ".join(promotion_gate_failures))

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
            "promotion_gate": {
                "max_baseline_cer_regression": config.promotion_gate.max_baseline_cer_regression,
                "max_baseline_wer_regression": config.promotion_gate.max_baseline_wer_regression,
                "max_source_cer_degradation": config.promotion_gate.max_source_cer_degradation,
                "max_source_wer_degradation": config.promotion_gate.max_source_wer_degradation,
                "hard_example_confidence_threshold": config.promotion_gate.hard_example_confidence_threshold,
            },
        },
        metrics=metrics_payload,
        git_sha=_resolve_git_sha(),
    )

    export_bundle_path = config.logs_dir / "model_export.tar.gz"
    training_config_path = config.logs_dir / "training_config.json"
    inference_config_path = config.logs_dir / "inference_config.json"
    training_config_path.write_text(
        json.dumps(
            {
                "data_dir": str(config.data_dir),
                "logs_dir": str(config.logs_dir),
                "languages": list(config.languages),
                "min_image_width": config.min_image_width,
                "min_image_height": config.min_image_height,
                "max_language_imbalance_ratio": config.max_language_imbalance_ratio,
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
                "promotion_gate": {
                    "max_baseline_cer_regression": config.promotion_gate.max_baseline_cer_regression,
                    "max_baseline_wer_regression": config.promotion_gate.max_baseline_wer_regression,
                    "max_source_cer_degradation": config.promotion_gate.max_source_cer_degradation,
                    "max_source_wer_degradation": config.promotion_gate.max_source_wer_degradation,
                    "hard_example_confidence_threshold": config.promotion_gate.hard_example_confidence_threshold,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    recognition_checkpoint_path = Path(stage_a_result["artifact_path"])
    detection_checkpoint_path = Path(stage_b_result["artifact_path"])
    if not execute:
        recognition_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        detection_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        recognition_checkpoint_path.write_text("skipped_dry_run", encoding="utf-8")
        detection_checkpoint_path.write_text("skipped_dry_run", encoding="utf-8")
    inference_config_path.write_text(
        json.dumps(
            {
                "recognition_checkpoint_path": str(recognition_checkpoint_path),
                "detection_checkpoint_path": str(detection_checkpoint_path),
                "languages": list(config.languages),
                "image_constraints": {
                    "min_image_width": config.min_image_width,
                    "min_image_height": config.min_image_height,
                },
                "pseudo_label": {
                    "confidence_threshold": config.pseudo_label.confidence_threshold,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    export_result = stages.export(
        export_bundle_path=export_bundle_path,
        recognition_checkpoint_path=recognition_checkpoint_path,
        detection_checkpoint_path=detection_checkpoint_path,
        metadata_bundle_path=metadata_path,
        evaluation_report_path=evaluation_report_path,
        baseline_comparison_path=baseline_comparison_path,
        training_config_path=training_config_path,
        inference_config_path=inference_config_path,
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
            "inference_config_path": inference_config_path,
        },
    }
