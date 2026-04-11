from __future__ import annotations

import sys
from pathlib import Path

from scripts.train_ocr import parse_args


def test_parse_args_defaults_use_canonical_processed_csv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ocr.py"])

    args = parse_args()

    assert args.processed_csv == Path("data/processed/consolidated_dataset.csv")


def test_readme_dry_run_command_uses_canonical_processed_csv():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "--dry-run --processed-csv data/processed/consolidated_dataset.csv" in readme
