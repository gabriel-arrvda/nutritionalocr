from pathlib import Path

import pandas as pd
import pytest

from src.training.config import PseudoLabelConfig, TrainingConfig
from src.training.dataset import normalize_label_text, stratified_split


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.data_dir == Path("data/processed/training")
    assert config.logs_dir == Path("logs/training")
    assert config.languages == ("pt", "en", "es", "fr", "de")
    assert config.min_image_width == 200
    assert config.min_image_height == 200
    assert isinstance(config.pseudo_label, PseudoLabelConfig)
    assert config.pseudo_label.confidence_threshold == 0.8


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_pseudo_label_config_rejects_invalid_confidence_threshold(value: float):
    with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
        PseudoLabelConfig(confidence_threshold=value)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0, 200),
        (-1, 200),
        (200, 0),
        (200, -1),
    ],
)
def test_training_config_rejects_non_positive_image_dimensions(width: int, height: int):
    with pytest.raises(
        ValueError,
        match="min_image_width and min_image_height must be greater than 0",
    ):
        TrainingConfig(min_image_width=width, min_image_height=height)


def test_training_config_rejects_empty_languages():
    with pytest.raises(ValueError, match="languages must not be empty"):
        TrainingConfig(languages=())


@pytest.mark.parametrize("languages", [("pt", ""), ("pt", " ")])
def test_training_config_rejects_blank_language_entries(languages: tuple[str, ...]):
    with pytest.raises(ValueError, match="languages must contain only non-blank strings"):
        TrainingConfig(languages=languages)


def test_normalize_label_text_applies_nfc_and_collapses_whitespace():
    text = "Cafe\u0301   com\t  leite\nintegral "

    normalized = normalize_label_text(text)

    assert normalized == "Café com leite integral"


def test_stratified_split_preserves_languages_in_all_splits():
    rows: list[dict[str, str | int]] = []
    idx = 0
    for language in ("pt", "en", "es"):
        for source_kind in ("manual", "pseudo"):
            for _ in range(10):
                rows.append(
                    {
                        "id": idx,
                        "language": language,
                        "source_kind": source_kind,
                    }
                )
                idx += 1
    df = pd.DataFrame(rows)

    splits = stratified_split(df, val_ratio=0.2, test_ratio=0.2, seed=42)

    expected_languages = {"pt", "en", "es"}
    for split_name in ("train", "val", "test"):
        assert set(splits[split_name]["language"]) == expected_languages


def test_stratified_split_rejects_empty_dataframe():
    df = pd.DataFrame(columns=["language", "source_kind"])

    with pytest.raises(ValueError, match="input dataframe cannot be empty"):
        stratified_split(df)
