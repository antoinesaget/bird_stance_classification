"""Purpose: Define the canonical species-aware data layout helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_HOME = Path("/data/birds")
DEFAULT_SPECIES_SLUG = "black_winged_stilt"


@dataclass(frozen=True)
class ProjectLayout:
    data_home: Path
    species_slug: str
    root: Path
    originals: Path
    metadata: Path
    labelstudio_imports: Path
    labelstudio_images_compressed: Path
    labelstudio_exports: Path
    labelstudio_normalized: Path
    derived_splits: Path
    derived_crops: Path
    derived_datasets: Path
    models_detector: Path
    models_attributes: Path
    models_image_status: Path


def default_data_home() -> Path:
    return Path(os.getenv("BIRD_DATA_HOME", str(DEFAULT_DATA_HOME))).expanduser()


def default_species_slug() -> str:
    raw = os.getenv("BIRD_SPECIES_SLUG", DEFAULT_SPECIES_SLUG).strip()
    if not raw:
        raise ValueError("BIRD_SPECIES_SLUG cannot be empty")
    return raw


def normalize_species_slug(species_slug: str) -> str:
    slug = species_slug.strip()
    if not slug:
        raise ValueError("species_slug cannot be empty")
    if slug.startswith("/") or slug.endswith("/"):
        raise ValueError(f"species_slug must be a slug, got {species_slug!r}")
    if any(token in slug for token in ("/", "\\", "..")):
        raise ValueError(f"species_slug must not contain path separators, got {species_slug!r}")
    return slug


def normalize_relative_path(value: str, *, field_name: str = "relative path") -> str:
    rel = value.strip().strip("/")
    if not rel:
        raise ValueError(f"{field_name} cannot be empty")
    path = Path(rel)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative, got {value!r}")
    if ".." in path.parts:
        raise ValueError(f"{field_name} must stay within the species root, got {value!r}")
    return path.as_posix()


def resolve_species_relative_path(species_root: Path, relative_path: str, *, field_name: str = "relative path") -> Path:
    rel = normalize_relative_path(relative_path, field_name=field_name)
    return species_root / Path(rel)


def build_layout(data_home: Path, species_slug: str) -> ProjectLayout:
    home = data_home.expanduser().resolve()
    slug = normalize_species_slug(species_slug)
    root = home / slug
    return ProjectLayout(
        data_home=home,
        species_slug=slug,
        root=root,
        originals=root / "originals",
        metadata=root / "metadata",
        labelstudio_imports=root / "labelstudio" / "imports",
        labelstudio_images_compressed=root / "labelstudio" / "images_compressed",
        labelstudio_exports=root / "labelstudio" / "exports",
        labelstudio_normalized=root / "labelstudio" / "normalized",
        derived_splits=root / "derived" / "splits",
        derived_crops=root / "derived" / "crops",
        derived_datasets=root / "derived" / "datasets",
        models_detector=root / "models" / "detector",
        models_attributes=root / "models" / "attributes",
        models_image_status=root / "models" / "image_status",
    )


def ensure_layout(data_home: Path, species_slug: str) -> ProjectLayout:
    layout = build_layout(data_home, species_slug)
    for path in (
        layout.data_home,
        layout.root,
        layout.originals,
        layout.metadata,
        layout.labelstudio_imports,
        layout.labelstudio_images_compressed,
        layout.labelstudio_exports,
        layout.labelstudio_normalized,
        layout.derived_splits,
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
