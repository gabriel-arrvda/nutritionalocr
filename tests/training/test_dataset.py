from pathlib import Path

from src.training.config import TrainingConfig


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.data_dir == Path("data/processed/training")
    assert config.logs_dir == Path("logs/training")
    assert config.train_ratio == 0.8
    assert config.val_ratio == 0.1
    assert config.test_ratio == 0.1
    assert config.pseudo_label_confidence_threshold == 0.8
    assert config.random_seed == 42
