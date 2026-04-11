from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.training.export import write_metadata_bundle


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
