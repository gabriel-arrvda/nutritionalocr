# OCR Training Design (Phase 2)

## Context and Goal

The data collection and augmentation phase is complete. This design defines Phase 2 to train an OCR model focused on **maximum recognition quality**, using **PaddleOCR**, **local GPU (MPS/CUDA)**, and **broad multilingual support from the first iteration**.

Primary goal: deliver a reproducible training pipeline (notebook + scripts) that produces an exportable best checkpoint with multilingual quality metrics.

## Scope

In scope:
- Build reproducible OCR training workflow for multilingual labels.
- Train in two stages:
  - Stage A: recognition fine-tuning first.
  - Stage B: detection refinement second.
- Produce versioned metrics and model artifacts.

Out of scope:
- API integration and frontend work.
- Production deployment orchestration.

## Architecture

Pipeline flow:

1. `data/processed` + `data/images/{original,augmented}` as source.
2. Dataset validation and text normalization.
3. Stratified train/validation/test manifest generation by language and source.
4. Stage A PaddleOCR recognizer training (multilingual).
5. Stage B PaddleOCR detector refinement with hard examples.
6. Consolidated evaluation (CER/WER + detection metrics).
7. Export of best checkpoint + inference configuration + run metadata.

Project structure:
- `notebooks/02_ocr_training.ipynb` for guided execution and analysis.
- `src/training/` for reproducible pipeline scripts.
- `data/processed/training/` for manifests, normalized labels, and split versions.
- `logs/training/` for experiment metrics and diagnostic artifacts.

## Components

### 1) `dataset_validator`
- Validates image-label integrity.
- Enforces minimum image quality constraints.
- Detects charset/language anomalies.
- Produces structured validation report.

### 2) `label_normalizer`
- Applies Unicode normalization (NFC).
- Standardizes spacing/punctuation policies.
- Ensures consistent tokenization for OCR targets.

### 3) `split_builder`
- Creates train/val/test splits with language and source stratification.
- Preserves multilingual representation.

### 4) `train_recognizer`
- Fine-tunes multilingual text recognition.
- Uses early stopping on validation CER/WER.

### 5) `train_detector`
- Refines text-region detection.
- Prioritizes hard examples from recognition-stage failures.

### 6) `evaluator`
- Computes CER/WER per language and global macro/micro summaries.
- Computes detection precision/recall/F1.
- Produces baseline vs best-run comparison.

### 7) `exporter`
- Exports selected best checkpoint.
- Saves configuration and metadata (dataset hash, hyperparameters, metrics, git commit SHA).

## Data Flow

1. Load processed and augmented assets.
2. Validate and normalize labels.
3. Build stratified manifests.
4. Train Stage A recognizer.
5. Build hard-set from Stage A errors.
6. Train Stage B detector.
7. Evaluate multilingual quality and detection quality.
8. Export best artifacts and reports.

## Error Handling

No silent failure behavior:

- Inconsistent dataset (missing pairs, unreadable files, invalid labels) -> fail fast and write `logs/training/dataset_validation_report.json`.
- Severe language imbalance -> block training and report required rebalancing actions.
- Training divergence (NaN/exploding loss) -> abort run and save diagnostics.
- Missing required GPU profile for configured mode -> stop with explicit actionable message.

## Testing Strategy

- Unit tests in `tests/training/` for:
  - dataset validation
  - label normalization
  - split generation and stratification checks
- Training smoke test (minimal batches) to verify pipeline execution path.
- Metrics regression test to prevent promoting models worse than baseline in multilingual CER/WER macro metrics.
- Final run report persisted with baseline vs best-run comparison.

## Done Criteria for Phase 2

1. Notebook + script pipeline is reproducible.
2. Best checkpoint is exported with full metadata.
3. Multilingual evaluation report is generated and approved.

