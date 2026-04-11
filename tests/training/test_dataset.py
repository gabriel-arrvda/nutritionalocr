from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from src.training.config import PseudoLabelConfig, TrainingConfig
from src.training.dataset import normalize_label_text, stratified_split


def _build_dataframe(stratum_sizes: dict[tuple[str, str], int]) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    idx = 0
    for (language, source_kind), size in stratum_sizes.items():
        for _ in range(size):
            rows.append(
                {
                    "id": idx,
                    "language": language,
                    "source_kind": source_kind,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.data_dir == Path("data/processed/training")
    assert config.logs_dir == Path("logs/training")
    assert config.languages == ("pt", "en", "es", "fr", "de")
    assert config.min_image_width == 200
    assert config.min_image_height == 200
    assert isinstance(config.pseudo_label, PseudoLabelConfig)
    assert config.pseudo_label.confidence_threshold == 0.8
    assert config.pseudo_label.max_pseudo_ratio_per_language == 0.7


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_pseudo_label_config_rejects_invalid_confidence_threshold(value: float):
    with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
        PseudoLabelConfig(confidence_threshold=value)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_pseudo_label_config_rejects_invalid_max_pseudo_ratio_per_language(value: float):
    with pytest.raises(
        ValueError,
        match="max_pseudo_ratio_per_language must be between 0 and 1",
    ):
        PseudoLabelConfig(max_pseudo_ratio_per_language=value)


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


def test_training_config_rejects_languages_as_plain_string():
    with pytest.raises(
        ValueError, match="languages must be a sequence of non-blank strings"
    ):
        TrainingConfig(languages="pt")


def test_normalize_label_text_applies_nfc_and_collapses_whitespace():
    text = "Cafe\u0301   com\t  leite\nintegral "

    normalized = normalize_label_text(text)

    assert normalized == "Café com leite integral"


def test_stratified_split_preserves_languages_in_all_splits():
    df = _build_dataframe(
        {
            ("pt", "manual"): 10,
            ("pt", "pseudo"): 10,
            ("en", "manual"): 10,
            ("en", "pseudo"): 10,
            ("es", "manual"): 10,
            ("es", "pseudo"): 10,
        }
    )

    splits = stratified_split(df, val_ratio=0.2, test_ratio=0.2, seed=42)

    expected_languages = {"pt", "en", "es"}
    for split_name in ("train", "val", "test"):
        assert set(splits[split_name]["language"]) == expected_languages


def test_stratified_split_has_no_overlap_and_full_row_coverage():
    df = _build_dataframe(
        {
            ("pt", "manual"): 12,
            ("pt", "pseudo"): 12,
            ("en", "manual"): 12,
            ("en", "pseudo"): 12,
        }
    )

    splits = stratified_split(df, val_ratio=0.25, test_ratio=0.25, seed=42)

    train_ids = set(splits["train"]["id"])
    val_ids = set(splits["val"]["id"])
    test_ids = set(splits["test"]["id"])
    original_ids = set(df["id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == original_ids


def test_stratified_split_preserves_source_kind_distribution_by_language():
    df = _build_dataframe(
        {
            ("pt", "manual"): 10,
            ("pt", "pseudo"): 10,
            ("en", "manual"): 10,
            ("en", "pseudo"): 10,
        }
    )

    splits = stratified_split(df, val_ratio=0.2, test_ratio=0.2, seed=42)

    for language in ("pt", "en"):
        for source_kind in ("manual", "pseudo"):
            assert len(
                splits["train"].query(
                    "language == @language and source_kind == @source_kind"
                )
            ) == 6
            assert len(
                splits["val"].query("language == @language and source_kind == @source_kind")
            ) == 2
            assert len(
                splits["test"].query(
                    "language == @language and source_kind == @source_kind"
                )
            ) == 2


def test_stratified_split_rejects_empty_dataframe():
    df = pd.DataFrame(columns=["language", "source_kind"])

    with pytest.raises(ValueError, match="input dataframe cannot be empty"):
        stratified_split(df)


@pytest.mark.parametrize("val_ratio", [-0.1, 1.0, 1.2])
def test_stratified_split_rejects_invalid_val_ratio(val_ratio: float):
    df = pd.DataFrame([{"language": "pt", "source_kind": "manual"}])

    with pytest.raises(ValueError, match="val_ratio must satisfy 0 <= val_ratio < 1"):
        stratified_split(df, val_ratio=val_ratio, test_ratio=0.2)


@pytest.mark.parametrize("test_ratio", [-0.1, 1.0, 1.2])
def test_stratified_split_rejects_invalid_test_ratio(test_ratio: float):
    df = pd.DataFrame([{"language": "pt", "source_kind": "manual"}])

    with pytest.raises(ValueError, match="test_ratio must satisfy 0 <= test_ratio < 1"):
        stratified_split(df, val_ratio=0.2, test_ratio=test_ratio)


def test_stratified_split_rejects_sum_of_ratios_greater_or_equal_to_one():
    df = pd.DataFrame([{"language": "pt", "source_kind": "manual"}])

    with pytest.raises(ValueError, match="val_ratio \\+ test_ratio must be less than 1"):
        stratified_split(df, val_ratio=0.5, test_ratio=0.5)


@pytest.mark.parametrize(
    ("columns", "expected_missing"),
    [
        (["source_kind"], "language"),
        (["language"], "source_kind"),
        (["id"], "language, source_kind"),
    ],
)
def test_stratified_split_rejects_missing_required_columns(
    columns: list[str], expected_missing: str
):
    df = pd.DataFrame(columns=columns)

    with pytest.raises(
        ValueError, match=f"missing required columns: {expected_missing}"
    ):
        stratified_split(df)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("language", None),
        ("language", ""),
        ("language", " "),
        ("source_kind", None),
        ("source_kind", ""),
        ("source_kind", " "),
    ],
)
def test_stratified_split_rejects_null_or_blank_stratification_values(
    column: str, value: Optional[str]
):
    df = _build_dataframe(
        {
            ("pt", "manual"): 4,
            ("en", "pseudo"): 4,
        }
    )
    df.loc[0, column] = value

    with pytest.raises(
        ValueError, match="language and source_kind must contain only non-blank values"
    ):
        stratified_split(df, val_ratio=0.2, test_ratio=0.2)


def test_stratified_split_ensures_small_strata_representation_when_feasible():
    df = _build_dataframe(
        {
            ("pt", "manual"): 3,
            ("en", "pseudo"): 3,
        }
    )

    splits = stratified_split(df, val_ratio=0.2, test_ratio=0.2, seed=42)

    for language, source_kind in (("pt", "manual"), ("en", "pseudo")):
        assert len(
            splits["train"].query(
                "language == @language and source_kind == @source_kind"
            )
        ) == 1
        assert len(
            splits["val"].query("language == @language and source_kind == @source_kind")
        ) == 1
        assert len(
            splits["test"].query(
                "language == @language and source_kind == @source_kind"
            )
        ) == 1


def test_stratified_split_rejects_small_strata_when_representation_is_impossible():
    df = _build_dataframe(
        {
            ("pt", "manual"): 2,
            ("en", "pseudo"): 2,
        }
    )

    with pytest.raises(
        ValueError,
        match="cannot preserve stratified representation for stratum",
    ):
        stratified_split(df, val_ratio=0.2, test_ratio=0.2, seed=42)
