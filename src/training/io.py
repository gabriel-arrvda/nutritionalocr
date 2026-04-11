from pathlib import Path


def ensure_training_dirs(*, logs_dir: Path, data_dir: Path) -> dict[str, Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "logs_dir": logs_dir,
        "data_dir": data_dir,
    }
