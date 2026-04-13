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
from .config import AppEnv, load_app_env
from .contracts import LabelStudioBatchSummary, ModelHealthPayload, PromotionMetadata
from .logging import configure_logging
from .model_b_artifacts import (
    DEFAULT_LABELS,
    HEADS,
    LoadedModelBArtifact,
    LoadedModelBMember,
    load_model_b_artifact,
)
from .models import ImageStatusModel, MultiHeadAttributeModel
from .paths import ProjectLayout, ensure_layout, next_version_dir
from .reporting import diff_numeric_dict, find_previous_version_dir

__all__ = [
    "AppEnv",
    "BEHAVIOR_TO_ID",
    "DEFAULT_LABELS",
    "HEADS",
    "ImageStatusModel",
    "LabelStudioBatchSummary",
    "LoadedModelBArtifact",
    "LoadedModelBMember",
    "ModelHealthPayload",
    "MultiHeadAttributeModel",
    "ProjectLayout",
    "PromotionMetadata",
    "READABILITY_TO_ID",
    "SPECIE_TO_ID",
    "STANCE_TO_ID",
    "SUBSTRATE_TO_ID",
    "compute_head_masks",
    "configure_logging",
    "diff_numeric_dict",
    "encode_labels",
    "ensure_layout",
    "find_previous_version_dir",
    "load_app_env",
    "load_model_b_artifact",
    "next_version_dir",
    "normalize_behavior",
    "normalize_choice",
    "normalize_stance",
    "normalize_substrate",
]
