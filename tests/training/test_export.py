from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from src.training.export import write_metadata_bundle
from src.training.stages import export as export_stage


def test_write_metadata_bundle_persists_dataset_hash_hparams_metrics_and_git_sha(tmp_path: Path):
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text("image_path,label_text\na.png,abc\n", encoding="utf-8")
    output_path = tmp_path / "metadata_bundle.json"

    metrics = {"overall": {"cer": 0.12, "wer": 0.2}}
    hyperparameters = {"learning_rate": 0.001, "epochs": 20}
    git_sha = "deadbeef1234"

    written_path = write_metadata_bundle(
        output_path=output_path,
        dataset_csv=dataset_csv,
        hyperparameters=hyperparameters,
        metrics=metrics,
        git_sha=git_sha,
    )

    assert written_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dataset_hash"] == hashlib.sha256(dataset_csv.read_bytes()).hexdigest()
    assert payload["hyperparameters"] == hyperparameters
    assert payload["metrics"] == metrics
    assert payload["git_sha"] == git_sha


def test_export_stage_writes_real_tar_gz_with_expected_artifacts(tmp_path: Path):
    recognition_checkpoint = tmp_path / "recognition" / "recognizer.ckpt"
    detection_checkpoint = tmp_path / "detection" / "detector.ckpt"
    metadata_bundle = tmp_path / "metadata" / "metadata_bundle.json"
    evaluation_report = tmp_path / "metrics" / "evaluation_report.json"
    baseline_comparison = tmp_path / "metrics" / "baseline_vs_best.json"
    training_config = tmp_path / "config" / "training_config.json"
    inference_config = tmp_path / "config" / "inference_config.json"
    export_bundle = tmp_path / "model_export.tar.gz"

    recognition_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    detection_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata_bundle.parent.mkdir(parents=True, exist_ok=True)
    evaluation_report.parent.mkdir(parents=True, exist_ok=True)
    training_config.parent.mkdir(parents=True, exist_ok=True)

    recognition_checkpoint.write_text("recognizer-weights", encoding="utf-8")
    detection_checkpoint.write_text("detector-weights", encoding="utf-8")
    metadata_bundle.write_text(json.dumps({"git_sha": "abc123"}), encoding="utf-8")
    evaluation_report.write_text(json.dumps({"recognition": {"overall": {"cer": 0.1, "wer": 0.2}}}), encoding="utf-8")
    baseline_comparison.write_text(
        json.dumps({"baseline": {"cer": 0.5, "wer": 0.6}, "best": {"cer": 0.1, "wer": 0.2}, "delta": {"cer": 0.4, "wer": 0.4}}),
        encoding="utf-8",
    )
    training_config.write_text(json.dumps({"languages": ["pt", "en"]}), encoding="utf-8")
    inference_config.write_text(json.dumps({"languages": ["pt", "en"]}), encoding="utf-8")

    export_stage(
        export_bundle_path=export_bundle,
        recognition_checkpoint_path=recognition_checkpoint,
        detection_checkpoint_path=detection_checkpoint,
        metadata_bundle_path=metadata_bundle,
        evaluation_report_path=evaluation_report,
        baseline_comparison_path=baseline_comparison,
        training_config_path=training_config,
        inference_config_path=inference_config,
    )

    assert export_bundle.read_bytes()[:2] == b"\x1f\x8b"
    with tarfile.open(export_bundle, "r:gz") as archive:
        names = set(archive.getnames())
        assert "recognition/recognizer.ckpt" in names
        assert "detection/detector.ckpt" in names
        assert "metadata/metadata_bundle.json" in names
        assert "metrics/evaluation_report.json" in names
        assert "metrics/baseline_vs_best.json" in names
        assert "config/training_config.json" in names
        assert "config/inference_config.json" in names

        exported_metadata = json.loads(
            archive.extractfile("metadata/metadata_bundle.json").read().decode("utf-8")  # type: ignore[union-attr]
        )
        assert exported_metadata["git_sha"] == "abc123"
