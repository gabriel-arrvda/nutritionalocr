"""Utility functions for nutrition label OCR data collection"""

from .data_collection import (
    validate_image,
    download_image,
    apply_augmentation,
    translate_nutrients,
)
from .consolidation import (
    match_original_row,
    get_mapping_warning_reason,
    count_unmatched_mappings,
    assert_quality_gates,
    serialize_translated_nutrients,
)

__all__ = [
    'validate_image',
    'download_image',
    'apply_augmentation',
    'translate_nutrients',
    'match_original_row',
    'get_mapping_warning_reason',
    'count_unmatched_mappings',
    'assert_quality_gates',
    'serialize_translated_nutrients',
]
