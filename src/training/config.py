from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StageAConfig:
    weighted_sampling_human_weight: float = 2.0
    weighted_sampling_pseudo_weight: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_metric: str = "cer"

    def __post_init__(self) -> None:
        if self.weighted_sampling_human_weight <= 0:
            raise ValueError("weighted_sampling_human_weight must be greater than 0")
        if self.weighted_sampling_pseudo_weight <= 0:
            raise ValueError("weighted_sampling_pseudo_weight must be greater than 0")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be greater than 0")
        if not self.early_stopping_metric.strip():
            raise ValueError("early_stopping_metric must be non-empty")


@dataclass(frozen=True)
class GPUProfileConfig:
    required: bool = False
    backend: str = "cuda"
    min_devices: int = 1

    def __post_init__(self) -> None:
        if self.min_devices <= 0:
            raise ValueError("min_devices must be greater than 0")
        if self.backend not in {"cuda", "mps"}:
            raise ValueError("backend must be one of: cuda, mps")


@dataclass(frozen=True)
class PseudoLabelConfig:
    confidence_threshold: float = 0.8
    max_pseudo_ratio_per_language: float = 0.4

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.max_pseudo_ratio_per_language <= 1:
            raise ValueError("max_pseudo_ratio_per_language must be between 0 and 1")


@dataclass(frozen=True)
class PromotionGateConfig:
    max_baseline_cer_regression: float = 0.0
    max_baseline_wer_regression: float = 0.0
    max_source_cer_degradation: float = 0.05
    max_source_wer_degradation: float = 0.05
    hard_example_confidence_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.max_baseline_cer_regression < 0:
            raise ValueError("max_baseline_cer_regression must be greater than or equal to 0")
        if self.max_baseline_wer_regression < 0:
            raise ValueError("max_baseline_wer_regression must be greater than or equal to 0")
        if self.max_source_cer_degradation < 0:
            raise ValueError("max_source_cer_degradation must be greater than or equal to 0")
        if self.max_source_wer_degradation < 0:
            raise ValueError("max_source_wer_degradation must be greater than or equal to 0")
        if not 0 <= self.hard_example_confidence_threshold <= 1:
            raise ValueError("hard_example_confidence_threshold must be between 0 and 1")


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path = Path("data/processed/training")
    logs_dir: Path = Path("logs/training")
    languages: tuple[str, ...] = ("pt", "en", "es", "fr", "de")
    min_image_width: int = 200
    min_image_height: int = 200
    max_language_imbalance_ratio: float = 3.0
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    gpu_profile: GPUProfileConfig = field(default_factory=GPUProfileConfig)
    promotion_gate: PromotionGateConfig = field(default_factory=PromotionGateConfig)

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
        if self.max_language_imbalance_ratio <= 1:
            raise ValueError("max_language_imbalance_ratio must be greater than 1")
