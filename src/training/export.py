from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_metadata_bundle(
    *,
    output_path: Path,
    dataset_csv: Path,
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    git_sha: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_hash": hashlib.sha256(dataset_csv.read_bytes()).hexdigest(),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "git_sha": git_sha,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
