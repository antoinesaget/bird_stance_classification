"""Purpose: Define the small shared payload types used across active subprojects."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionMetadata:
    release_id: str
    promoted_at: str
    label: str
    source_path: str
    source_kind: str
    source_sha256: str
    active_artifact: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelHealthPayload:
    model_a_loaded: bool
    model_b_loaded: bool
    model_a_weights: str
    model_b_checkpoint: str | None
    model_b_artifact: str | None
    model_b_mode: str | None
    model_b_members: list[str]
    model_b_schema_version: str | None
    model_b_supported_labels: dict[str, list[str]]
    model_a_device: str | None = None
    model_a_imgsz: int | None = None
    model_a_max_det: int | None = None
    model_a_release_id: str | None = None
    model_a_source_path: str | None = None
    model_a_source_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelStudioBatchSummary:
    batch_name: str
    data_root: str
    source_root: str
    mirror_root: str
    import_root: str
    dataset_name: str
    images_available: int
    images_selected: int
    jpeg_quality: int
    total_original_bytes: int
    total_served_bytes: int
    size_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
