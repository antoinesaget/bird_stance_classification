"""Purpose: Define the canonical repository and data-root layout helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    root: Path
    raw_images: Path
    metadata: Path
    labelstudio_exports: Path
    labelstudio_normalized: Path
    derived_crops: Path
    derived_datasets: Path
    models_detector: Path
    models_attributes: Path
    models_image_status: Path


def build_layout(data_root: Path) -> ProjectLayout:
    root = data_root.resolve()
    return ProjectLayout(
        root=root,
        raw_images=root / "raw_images",
        metadata=root / "metadata",
        labelstudio_exports=root / "labelstudio" / "exports",
        labelstudio_normalized=root / "labelstudio" / "normalized",
        derived_crops=root / "derived" / "crops",
        derived_datasets=root / "derived" / "datasets",
        models_detector=root / "models" / "detector",
        models_attributes=root / "models" / "attributes",
        models_image_status=root / "models" / "image_status",
    )


def ensure_layout(data_root: Path) -> ProjectLayout:
    layout = build_layout(data_root)
    for path in (
        layout.root,
        layout.raw_images,
        layout.metadata,
        layout.labelstudio_exports,
        layout.labelstudio_normalized,
        layout.derived_crops,
        layout.derived_datasets,
        layout.models_detector,
        layout.models_attributes,
        layout.models_image_status,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def next_version_dir(parent: Path, prefix: str, width: int = 3) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    existing = [path.name for path in parent.iterdir() if path.is_dir() and path.name.startswith(f"{prefix}_v")]
    max_seen = 0
    for name in existing:
        suffix = name.replace(f"{prefix}_v", "", 1)
        if suffix.isdigit():
            max_seen = max(max_seen, int(suffix))
    target = parent / f"{prefix}_v{max_seen + 1:0{width}d}"
    target.mkdir(parents=True, exist_ok=False)
    return target
