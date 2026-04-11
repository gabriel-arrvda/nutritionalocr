from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PseudoLabelConfig:
    confidence_threshold: float = 0.8
    max_pseudo_ratio_per_language: float = 0.7

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.max_pseudo_ratio_per_language <= 1:
            raise ValueError("max_pseudo_ratio_per_language must be between 0 and 1")


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path = Path("data/processed/training")
    logs_dir: Path = Path("logs/training")
    languages: tuple[str, ...] = ("pt", "en", "es", "fr", "de")
    min_image_width: int = 200
    min_image_height: int = 200
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)

    def __post_init__(self) -> None:
        if isinstance(self.languages, str):
            raise ValueError("languages must be a sequence of non-blank strings")

        normalized_languages = tuple(self.languages)
        object.__setattr__(self, "languages", normalized_languages)

        if not normalized_languages:
            raise ValueError("languages must not be empty")

        if any(not isinstance(language, str) or not language.strip() for language in normalized_languages):
            raise ValueError("languages must contain only non-blank strings")

        if self.min_image_width <= 0 or self.min_image_height <= 0:
            raise ValueError("min_image_width and min_image_height must be greater than 0")
