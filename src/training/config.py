from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path = Path("data/processed/training")
    logs_dir: Path = Path("logs/training")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    pseudo_label_confidence_threshold: float = 0.8
    random_seed: int = 42
