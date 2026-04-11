from pathlib import Path


def ensure_training_dirs(*, logs_dir: Path, data_dir: Path) -> dict[str, Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    recognition_run_dir = logs_dir / "recognition"
    detection_run_dir = logs_dir / "detection"
    recognition_run_dir.mkdir(parents=True, exist_ok=True)
    detection_run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "logs_dir": logs_dir,
        "data_dir": data_dir,
        "recognition_run_dir": recognition_run_dir,
        "detection_run_dir": detection_run_dir,
    }
