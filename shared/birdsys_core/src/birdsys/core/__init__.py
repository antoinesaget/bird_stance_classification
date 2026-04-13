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
from .paths import (
    DEFAULT_DATA_HOME,
    DEFAULT_SPECIES_SLUG,
    ProjectLayout,
    build_layout,
    default_data_home,
    default_species_slug,
    ensure_layout,
    next_version_dir,
    normalize_species_slug,
    normalize_relative_path,
    resolve_species_relative_path,
)
from .reporting import diff_numeric_dict, find_previous_version_dir

__all__ = [
    "BEHAVIOR_TO_ID",
    "DEFAULT_DATA_HOME",
    "DEFAULT_SPECIES_SLUG",
    "LabelStudioBatchSummary",
    "ModelHealthPayload",
    "ProjectLayout",
    "PromotionMetadata",
    "READABILITY_TO_ID",
    "SPECIE_TO_ID",
    "STANCE_TO_ID",
    "SUBSTRATE_TO_ID",
    "build_layout",
    "compute_head_masks",
    "default_data_home",
    "default_species_slug",
    "diff_numeric_dict",
    "encode_labels",
    "ensure_layout",
    "find_previous_version_dir",
    "next_version_dir",
    "normalize_behavior",
    "normalize_choice",
    "normalize_relative_path",
    "normalize_species_slug",
    "normalize_stance",
    "normalize_substrate",
    "resolve_species_relative_path",
]
