from pathlib import Path


def ensure_training_dirs(project_root: Path) -> dict[str, Path]:
    logs_dir = project_root / "logs" / "training"
    data_dir = project_root / "data" / "processed" / "training"

    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "logs_dir": logs_dir,
        "data_dir": data_dir,
    }
