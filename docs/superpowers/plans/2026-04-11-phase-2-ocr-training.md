# Phase 2 OCR Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible multilingual PaddleOCR training pipeline (notebook + scripts) that uses labeled and pseudo-labeled images, exports the best checkpoint, and reports CER/WER plus detection metrics.

**Architecture:** Add a focused `src/training/` package for dataset prep, pseudo-label filtering, and pipeline orchestration; keep notebook usage in `notebooks/02_ocr_training.ipynb` as a guided execution layer over scriptable modules. Training runs in two stages (recognition then detection), with explicit quality gates for pseudo-label confidence and source balance. Tests in `tests/training/` lock behavior for normalization, split balance, pseudo-label acceptance, and smoke execution.

**Tech Stack:** Python 3, PaddleOCR/PaddlePaddle, pandas, numpy, pytest, Jupyter

---

## File Structure

- Create: `src/training/__init__.py` — public exports for training utilities.
- Create: `src/training/config.py` — dataclasses for run config and thresholds.
- Create: `src/training/dataset.py` — validation, label normalization, manifest build, stratified split.
- Create: `src/training/pseudo_labeling.py` — teacher prediction model, confidence filtering, merge policy.
- Create: `src/training/pipeline.py` — orchestration of Stage A/Stage B flow and report payload.
- Create: `src/training/io.py` — filesystem/report helpers (`logs/training`, `data/processed/training`).
- Create: `scripts/train_ocr.py` — CLI entrypoint for reproducible runs.
- Create: `tests/training/test_dataset.py` — tests for normalization, manifests, stratified splits.
- Create: `tests/training/test_pseudo_labeling.py` — tests for pseudo-label thresholds and source caps.
- Create: `tests/training/test_pipeline_smoke.py` — smoke tests for pipeline orchestration behavior.
- Create: `notebooks/02_ocr_training.ipynb` — notebook that calls `src/training` modules (no duplicated core logic).
- Modify: `requirements.txt` — add PaddleOCR/PaddlePaddle and utilities required by pipeline.
- Modify: `README.md` — add Phase 2 run instructions and expected artifacts.

### Task 1: Training package skeleton + dependencies

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/config.py`
- Modify: `requirements.txt`
- Test: `tests/training/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_dataset.py
from src.training.config import TrainingConfig, PseudoLabelConfig


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.languages == ["pt", "en", "es", "fr", "de"]
    assert cfg.min_image_width == 200
    assert cfg.min_image_height == 200
    assert isinstance(cfg.pseudo_label, PseudoLabelConfig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_dataset.py::test_training_config_defaults -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/training/config.py
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PseudoLabelConfig:
    min_confidence: float = 0.80
    max_pseudo_ratio_per_language: float = 0.40


@dataclass(frozen=True)
class TrainingConfig:
    languages: list[str] = field(default_factory=lambda: ["pt", "en", "es", "fr", "de"])
    min_image_width: int = 200
    min_image_height: int = 200
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
```

```python
# src/training/__init__.py
from .config import TrainingConfig, PseudoLabelConfig

__all__ = ["TrainingConfig", "PseudoLabelConfig"]
```

```text
# requirements.txt (append)
paddleocr>=2.8.1
# choose one paddle build compatible with target GPU backend
paddlepaddle>=2.6.1
scikit-learn>=1.4.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_dataset.py::test_training_config_defaults -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/training/__init__.py src/training/config.py tests/training/test_dataset.py
git commit -m "feat: add training config skeleton for phase 2"
```

### Task 2: Dataset normalization + stratified split

**Files:**
- Create: `src/training/dataset.py`
- Modify: `tests/training/test_dataset.py`
- Test: `tests/training/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
from src.training.dataset import normalize_label_text, stratified_split


def test_normalize_label_text_standardizes_unicode_and_spaces():
    raw = "  Proteína\u00a0 total   "
    assert normalize_label_text(raw) == "Proteína total"


def test_stratified_split_preserves_language_presence():
    df = pd.DataFrame(
        {
            "image_path": [f"img_{i}.jpg" for i in range(12)],
            "label_text": [f"txt_{i}" for i in range(12)],
            "language": ["pt"] * 4 + ["en"] * 4 + ["es"] * 4,
            "source_kind": ["human_label"] * 12,
        }
    )
    split = stratified_split(df, val_ratio=0.2, test_ratio=0.2, seed=42)
    assert set(split.keys()) == {"train", "val", "test"}
    for part in split.values():
        assert {"pt", "en", "es"}.issubset(set(part["language"]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/training/test_dataset.py -v`  
Expected: FAIL with `ImportError` for `normalize_label_text` / `stratified_split`

- [ ] **Step 3: Write minimal implementation**

```python
# src/training/dataset.py
from __future__ import annotations

import unicodedata
import pandas as pd


def normalize_label_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    return " ".join(normalized.split())


def stratified_split(
    df: pd.DataFrame, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42
) -> dict[str, pd.DataFrame]:
    if df.empty:
        raise ValueError("Input dataframe is empty")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    groups = [g for _, g in shuffled.groupby(["language", "source_kind"], sort=False)]

    train_parts, val_parts, test_parts = [], [], []
    for group in groups:
        n = len(group)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        test_parts.append(group.iloc[:n_test])
        val_parts.append(group.iloc[n_test : n_test + n_val])
        train_parts.append(group.iloc[n_test + n_val :])

    return {
        "train": pd.concat(train_parts).reset_index(drop=True),
        "val": pd.concat(val_parts).reset_index(drop=True),
        "test": pd.concat(test_parts).reset_index(drop=True),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_dataset.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/dataset.py tests/training/test_dataset.py
git commit -m "feat: add dataset normalization and stratified split for ocr training"
```

### Task 3: Pseudo-label filtering and merge policy

**Files:**
- Create: `src/training/pseudo_labeling.py`
- Create: `tests/training/test_pseudo_labeling.py`
- Test: `tests/training/test_pseudo_labeling.py`

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
import pytest
from src.training.config import PseudoLabelConfig
from src.training.pseudo_labeling import filter_pseudo_labels, enforce_pseudo_ratio_cap


def test_filter_pseudo_labels_keeps_only_high_confidence_rows():
    df = pd.DataFrame(
        {
            "language": ["pt", "pt", "en"],
            "prediction_text": ["A", "B", "C"],
            "confidence": [0.92, 0.61, 0.88],
        }
    )
    cfg = PseudoLabelConfig(min_confidence=0.80, max_pseudo_ratio_per_language=0.40)
    filtered = filter_pseudo_labels(df, cfg)
    assert list(filtered["label_text"]) == ["A", "C"]


def test_enforce_pseudo_ratio_cap_raises_when_exceeds_limit():
    stats = {"pt": {"pseudo": 8, "human": 10}}
    cfg = PseudoLabelConfig(min_confidence=0.80, max_pseudo_ratio_per_language=0.40)
    with pytest.raises(ValueError, match="exceeds max pseudo ratio"):
        enforce_pseudo_ratio_cap(stats, cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/training/test_pseudo_labeling.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training.pseudo_labeling'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/training/pseudo_labeling.py
from __future__ import annotations

import pandas as pd
from .config import PseudoLabelConfig


def filter_pseudo_labels(df: pd.DataFrame, cfg: PseudoLabelConfig) -> pd.DataFrame:
    if "confidence" not in df.columns:
        raise ValueError("Missing confidence column in pseudo-label dataframe")
    accepted = df[df["confidence"] >= cfg.min_confidence].copy()
    accepted["source_kind"] = "pseudo_label"
    accepted = accepted.rename(columns={"prediction_text": "label_text"})
    return accepted


def enforce_pseudo_ratio_cap(stats: dict[str, dict[str, int]], cfg: PseudoLabelConfig) -> None:
    for language, values in stats.items():
        pseudo = int(values.get("pseudo", 0))
        human = int(values.get("human", 0))
        total = pseudo + human
        if total == 0:
            continue
        ratio = pseudo / total
        if ratio > cfg.max_pseudo_ratio_per_language:
            raise ValueError(
                f"Language '{language}' exceeds max pseudo ratio: {ratio:.2f} > {cfg.max_pseudo_ratio_per_language:.2f}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_pseudo_labeling.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/pseudo_labeling.py tests/training/test_pseudo_labeling.py
git commit -m "feat: add pseudo-label confidence filtering and ratio guardrails"
```

### Task 4: Pipeline orchestration + smoke test

**Files:**
- Create: `src/training/pipeline.py`
- Create: `src/training/io.py`
- Create: `tests/training/test_pipeline_smoke.py`
- Test: `tests/training/test_pipeline_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
from pathlib import Path
from src.training.config import TrainingConfig
from src.training.pipeline import run_training_pipeline


def test_pipeline_smoke_returns_expected_report_shape(tmp_path: Path):
    cfg = TrainingConfig()
    report = run_training_pipeline(
        config=cfg,
        processed_csv=tmp_path / "consolidated_dataset.csv",
        image_root=tmp_path / "images",
        dry_run=True,
    )
    assert report["status"] == "dry_run_ok"
    assert "stages" in report and report["stages"] == ["recognition", "detection"]
    assert "artifacts" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_pipeline_smoke.py -v`  
Expected: FAIL with `ImportError` for `run_training_pipeline`

- [ ] **Step 3: Write minimal implementation**

```python
# src/training/io.py
from __future__ import annotations

from pathlib import Path


def ensure_training_dirs(project_root: Path) -> dict[str, Path]:
    logs_dir = project_root / "logs" / "training"
    processed_dir = project_root / "data" / "processed" / "training"
    logs_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"logs_dir": logs_dir, "processed_dir": processed_dir}
```

```python
# src/training/pipeline.py
from __future__ import annotations

from pathlib import Path
from .config import TrainingConfig
from .io import ensure_training_dirs


def run_training_pipeline(
    config: TrainingConfig,
    processed_csv: Path,
    image_root: Path,
    dry_run: bool = False,
) -> dict:
    project_root = Path.cwd()
    dirs = ensure_training_dirs(project_root)
    if dry_run:
        return {
            "status": "dry_run_ok",
            "stages": ["recognition", "detection"],
            "artifacts": {
                "logs_dir": str(dirs["logs_dir"]),
                "processed_dir": str(dirs["processed_dir"]),
                "processed_csv": str(processed_csv),
                "image_root": str(image_root),
            },
        }
    raise NotImplementedError("Non-dry training will be implemented in subsequent iterations")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_pipeline_smoke.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/io.py src/training/pipeline.py tests/training/test_pipeline_smoke.py
git commit -m "feat: add phase 2 pipeline orchestration smoke path"
```

### Task 5: CLI + notebook wiring + documentation

**Files:**
- Create: `scripts/train_ocr.py`
- Create: `notebooks/02_ocr_training.ipynb`
- Modify: `README.md`
- Test: `tests/training/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing CLI-focused test**

```python
import subprocess


def test_train_ocr_cli_supports_dry_run_flag():
    proc = subprocess.run(
        ["python3", "scripts/train_ocr.py", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "dry_run_ok" in proc.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_pipeline_smoke.py::test_train_ocr_cli_supports_dry_run_flag -v`  
Expected: FAIL because CLI script does not exist

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/train_ocr.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.config import TrainingConfig
from src.training.pipeline import run_training_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 OCR training pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Validate orchestration without model training")
    parser.add_argument("--processed-csv", default="data/processed/consolidated_dataset.csv")
    parser.add_argument("--image-root", default="data/images")
    args = parser.parse_args()

    report = run_training_pipeline(
        config=TrainingConfig(),
        processed_csv=Path(args.processed_csv),
        image_root=Path(args.image_root),
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# notebooks/02_ocr_training.ipynb (code cell outline)
from pathlib import Path
from src.training.config import TrainingConfig
from src.training.pipeline import run_training_pipeline

cfg = TrainingConfig()
report = run_training_pipeline(
    config=cfg,
    processed_csv=Path("data/processed/consolidated_dataset.csv"),
    image_root=Path("data/images"),
    dry_run=True,
)
report
```

```markdown
<!-- README.md additions -->
## Phase 2: OCR Training

Run smoke orchestration:

```bash
python3 scripts/train_ocr.py --dry-run
```

Expected artifacts:
- `logs/training/`
- `data/processed/training/`
- JSON report with `status: dry_run_ok`
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_pipeline_smoke.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_ocr.py notebooks/02_ocr_training.ipynb README.md tests/training/test_pipeline_smoke.py
git commit -m "feat: add phase 2 ocr training CLI and notebook entrypoint"
```

### Task 6: Full verification gate for Phase 2 scaffolding

**Files:**
- Modify: none (verification-only)
- Test: `tests/training/test_dataset.py`
- Test: `tests/training/test_pseudo_labeling.py`
- Test: `tests/training/test_pipeline_smoke.py`

- [ ] **Step 1: Run targeted training tests**

Run: `pytest tests/training -v`  
Expected: PASS for dataset, pseudo-label, and smoke suites

- [ ] **Step 2: Run existing regression tests**

Run: `pytest tests/test_data_collection.py tests/test_consolidation_utils.py tests/test_review_fixes.py -v`  
Expected: PASS to confirm Phase 1 behavior is unchanged

- [ ] **Step 3: Commit verification snapshot**

```bash
git add -A
git commit -m "test: verify phase 2 scaffolding without regressions"
```
