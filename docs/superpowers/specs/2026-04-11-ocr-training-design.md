# OCR Training Design (Phase 2)

## Context and Goal

The data collection and augmentation phase is complete. This design defines Phase 2 to train an OCR model focused on **maximum recognition quality**, using **PaddleOCR**, **local GPU (MPS/CUDA)**, and **broad multilingual support from the first iteration**.

Primary goal: deliver a reproducible training pipeline (notebook + scripts) that produces an exportable best checkpoint with multilingual quality metrics.

## Scope

In scope:
- Build reproducible OCR training workflow for multilingual labels.
- Use unlabeled images through pseudo-labeling with confidence control.
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
4. Teacher-pass inference on unlabeled images to generate pseudo-label candidates.
5. Pseudo-label filtering by confidence and consistency rules.
6. Stage A PaddleOCR recognizer training (multilingual) with labeled + approved pseudo-labeled samples.
7. Stage B PaddleOCR detector refinement with hard examples.
8. Consolidated evaluation (CER/WER + detection metrics).
9. Export of best checkpoint + inference configuration + run metadata.

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

### 4) `pseudo_labeler`
- Runs teacher inference on unlabeled images.
- Produces pseudo-label candidates with confidence metadata.
- Tracks source provenance (`human_label` vs `pseudo_label`).

### 5) `pseudo_label_filter`
- Accepts pseudo-labels only above configured confidence threshold.
- Rejects low-confidence, empty, or inconsistent text predictions.
- Caps pseudo-label ratio per language/source to avoid drift.

### 6) `train_recognizer`
- Fine-tunes multilingual text recognition.
- Uses early stopping on validation CER/WER.
- Uses weighted sampling so human labels keep higher training influence.

### 7) `train_detector`
- Refines text-region detection.
- Prioritizes hard examples from recognition-stage failures.

### 8) `evaluator`
- Computes CER/WER per language and global macro/micro summaries.
- Computes detection precision/recall/F1.
- Reports metrics segmented by data source (human-labeled vs pseudo-labeled contribution).
- Produces baseline vs best-run comparison.

### 9) `exporter`
- Exports selected best checkpoint.
- Saves configuration and metadata (dataset hash, hyperparameters, metrics, git commit SHA).

## Data Flow

1. Load processed and augmented assets.
2. Validate and normalize labels.
3. Run teacher inference on unlabeled images.
4. Filter pseudo-labels by confidence and consistency.
5. Build stratified manifests with source-aware balancing.
6. Train Stage A recognizer with weighted labeled/pseudo-labeled mix.
7. Build hard-set from Stage A errors.
8. Train Stage B detector.
9. Evaluate multilingual quality and detection quality.
10. Export best artifacts and reports.

## Error Handling

No silent failure behavior:

- Inconsistent dataset (missing pairs, unreadable files, invalid labels) -> fail fast and write `logs/training/dataset_validation_report.json`.
- Severe language imbalance -> block training and report required rebalancing actions.
- Pseudo-label quality below threshold floor -> block pseudo-label merge and continue with labeled-only subset.
- Pseudo-label dominance over human-labeled data -> block run and require reweight/rebalance config.
- Training divergence (NaN/exploding loss) -> abort run and save diagnostics.
- Missing required GPU profile for configured mode -> stop with explicit actionable message.

## Testing Strategy

- Unit tests in `tests/training/` for:
  - dataset validation
  - label normalization
  - pseudo-label filtering and acceptance/rejection rules
  - split generation and stratification checks
- Training smoke test (minimal batches) to verify pipeline execution path.
- Metrics regression test to prevent promoting models worse than baseline in multilingual CER/WER macro metrics.
- Source-segmentation check to ensure pseudo-label usage does not degrade labeled-set validation metrics.
- Final run report persisted with baseline vs best-run comparison.

## Done Criteria for Phase 2

1. Notebook + script pipeline is reproducible.
2. Best checkpoint is exported with full metadata.
3. Multilingual evaluation report is generated and approved.
