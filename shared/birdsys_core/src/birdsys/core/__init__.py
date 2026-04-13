"""Purpose: Expose the shared core helpers and contracts used across active subprojects."""
"""Shared BirdSys core contracts and runtime helpers."""

from .attributes import (
    BEHAVIOR_TO_ID,
    READABILITY_TO_ID,
    SPECIE_TO_ID,
    STANCE_TO_ID,
    SUBSTRATE_TO_ID,
    compute_head_masks,
    encode_labels,
    normalize_behavior,
    normalize_choice,
    normalize_stance,
    normalize_substrate,
)
from .contracts import LabelStudioBatchSummary, ModelHealthPayload, PromotionMetadata
from .paths import ProjectLayout, ensure_layout, next_version_dir
from .reporting import diff_numeric_dict, find_previous_version_dir

__all__ = [
    "BEHAVIOR_TO_ID",
    "LabelStudioBatchSummary",
    "ModelHealthPayload",
    "ProjectLayout",
    "PromotionMetadata",
    "READABILITY_TO_ID",
    "SPECIE_TO_ID",
    "STANCE_TO_ID",
    "SUBSTRATE_TO_ID",
    "compute_head_masks",
    "diff_numeric_dict",
    "encode_labels",
    "ensure_layout",
    "find_previous_version_dir",
    "next_version_dir",
    "normalize_behavior",
    "normalize_choice",
    "normalize_stance",
    "normalize_substrate",
]
