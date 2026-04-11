import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.config import TrainingConfig
from src.training.pipeline import run_training_pipeline


def _to_json_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_serializable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR training pipeline")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full pipeline planning with strict validation and no heavy training (default)",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        help="Run executable full Phase 2 orchestration flow",
    )
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("data/processed/training/merged.csv"),
        help="Path to processed dataset CSV",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/images"),
        help="Path to image root directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainingConfig()
    report = run_training_pipeline(
        config=config,
        processed_csv=args.processed_csv,
        image_root=args.image_root,
        dry_run=not args.execute,
    )
    print(json.dumps(_to_json_serializable(report), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
