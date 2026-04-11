from __future__ import annotations

import argparse
import json
from pathlib import Path


def _touch_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Placeholder training stage runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage_a = subparsers.add_parser("stage-a")
    stage_a.add_argument("--manifest", type=Path, required=True)
    stage_a.add_argument("--output-dir", type=Path, required=True)

    stage_b = subparsers.add_parser("stage-b")
    stage_b.add_argument("--manifest", type=Path, required=True)
    stage_b.add_argument("--output-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--recognizer-dir", type=Path, required=True)
    evaluate.add_argument("--detector-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--recognizer-dir", type=Path, required=True)
    export.add_argument("--detector-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "stage-a":
        _touch_file(args.output_dir / "recognizer.ckpt", f"manifest={args.manifest}")
    elif args.command == "stage-b":
        _touch_file(args.output_dir / "detector.ckpt", f"manifest={args.manifest}")
    elif args.command == "evaluate":
        payload = {
            "manifest": str(args.manifest),
            "recognizer_dir": str(args.recognizer_dir),
            "detector_dir": str(args.detector_dir),
        }
        _touch_file(args.output, json.dumps(payload, ensure_ascii=False))
    elif args.command == "export":
        _touch_file(
            args.output,
            f"recognizer_dir={args.recognizer_dir}\ndetector_dir={args.detector_dir}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
