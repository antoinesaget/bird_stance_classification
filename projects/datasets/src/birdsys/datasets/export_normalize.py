#!/usr/bin/env python3
"""Purpose: Normalize strict Label Studio exports into versioned BirdSys annotation tables."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import urllib.parse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any
from xml.etree import ElementTree as ET

from birdsys.core import (
    default_data_home,
    default_species_slug,
    ensure_layout,
    find_previous_version_dir,
    normalize_choice,
    normalize_relative_path,
    normalize_species_slug,
)


CURRENT_SCHEMA_VERSION = "annotation_schema_v2"
NORMALIZATION_POLICY_VERSION = "current_only_v1"
LABEL_FIELDS = ["isbird", "readability", "specie", "behavior", "substrate", "stance"]
BEHAVIOR_VALUES = {
    "flying",
    "moving",
    "foraging",
    "resting",
    "backresting",
    "bathing",
    "calling",
    "preening",
    "display",
    "breeding",
    "other",
    "unsure",
}
SUBSTRATE_VALUES = {"bare_ground", "vegetation", "water", "air", "unsure"}
STANCE_VALUES = {"unipedal", "bipedal", "sitting", "unsure"}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
DEFAULT_LABEL_CONFIG = REPO_ROOT / "projects" / "labelstudio" / "label_config.xml"


@dataclass(frozen=True)
class CurrentAnnotationSchema:
    rectangle_from_name: str
    rectangle_labels: frozenset[str]
    choice_fields: dict[str, frozenset[str]]


@dataclass
class BirdRow:
    annotation_version: str
    task_id: str
    annotation_id: str
    region_id: str
    image_id: str
    bird_id: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    bbox_x_px: int | None
    bbox_y_px: int | None
    bbox_w_px: int | None
    bbox_h_px: int | None
    isbird: str
    readability: str | None
    specie: str | None
    behavior: str | None
    substrate: str | None
    stance: str | None


@dataclass(frozen=True)
class TaskImageMetadata:
    species_slug: str
    image_id: str
    source_filename: str
    original_relpath: str
    compressed_relpath: str
    labelstudio_localfiles_relpath: str
    compression_profile: str
    site_id: str | None


@dataclass(frozen=True)
class NormalizationResult:
    annotation_version: str
    out_dir: pathlib.Path
    birds_out: pathlib.Path
    images_out: pathlib.Path
    manifest_out: pathlib.Path
    extract_report_json: pathlib.Path
    extract_report_md: pathlib.Path
    comparison_json: pathlib.Path
    comparison_md: pathlib.Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a strict Label Studio export into deterministic parquet")
    parser.add_argument("--export-json", required=True, help="Path to the raw export JSON")
    parser.add_argument("--annotation-version", required=True, help="Annotation version, e.g. ann_v001")
    parser.add_argument("--data-home", default=str(default_data_home()))
    parser.add_argument("--species-slug", default=default_species_slug())
    parser.add_argument("--label-config", default=str(DEFAULT_LABEL_CONFIG), help="Path to the current Label Studio label_config.xml")
    parser.add_argument("--raw-metadata-json", default="", help="Optional export metadata JSON to carry into the manifest")
    return parser.parse_args(argv)


def load_current_schema(label_config_path: pathlib.Path) -> CurrentAnnotationSchema:
    root = ET.fromstring(label_config_path.read_text(encoding="utf-8"))
    rectangle_from_name = ""
    rectangle_labels: set[str] = set()
    choice_fields: dict[str, frozenset[str]] = {}

    for node in root.iter():
        tag = node.tag.split("}", 1)[-1]
        if tag == "RectangleLabels":
            rectangle_from_name = normalize_choice(node.attrib.get("name")) or ""
            rectangle_labels = {
                value
                for child in node
                if child.tag.split("}", 1)[-1] == "Label"
                for value in [normalize_choice(child.attrib.get("value"))]
                if value is not None
            }
        elif tag == "Choices":
            field = normalize_choice(node.attrib.get("name"))
            if field is None:
                continue
            values = {
                value
                for child in node
                if child.tag.split("}", 1)[-1] == "Choice"
                for value in [normalize_choice(child.attrib.get("value"))]
                if value is not None
            }
            choice_fields[field] = frozenset(sorted(values))

    if rectangle_from_name != "bird_bbox" or rectangle_labels != {"bird"}:
        raise ValueError(
            f"Unexpected rectangle schema in {label_config_path}: name={rectangle_from_name!r} labels={sorted(rectangle_labels)!r}"
        )
    for field in LABEL_FIELDS:
        if field not in choice_fields:
            raise ValueError(f"Missing choice field {field!r} in {label_config_path}")
    return CurrentAnnotationSchema(
        rectangle_from_name=rectangle_from_name,
        rectangle_labels=frozenset(sorted(rectangle_labels)),
        choice_fields=choice_fields,
    )


def load_raw_metadata(raw_metadata_json: pathlib.Path | None) -> dict[str, Any] | None:
    if raw_metadata_json is None:
        return None
    if not raw_metadata_json.exists():
        raise FileNotFoundError(raw_metadata_json)
    payload = json.loads(raw_metadata_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected metadata object in {raw_metadata_json}")
    return payload


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def bird_row_id(task_id: str, region_id: str) -> str:
    return f"task:{task_id}:region:{region_id}"


def parse_localfiles_relpath(raw: str) -> str:
    parsed = urllib.parse.urlparse(raw)
    query = urllib.parse.parse_qs(parsed.query)
    local = urllib.parse.unquote(query.get("d", [""])[0]).strip()
    if not local:
        raise ValueError("Label Studio local-files URL is missing the d= query value")
    return normalize_relative_path(local, field_name="labelstudio local-files relpath")


def image_metadata_from_task(task: dict[str, Any], *, species_slug: str) -> TaskImageMetadata:
    task_id = task.get("id")
    meta = task.get("meta") or {}
    data = task.get("data") or {}

    task_species_slug = normalize_choice(str(meta.get("species_slug") or ""))
    if task_species_slug is None:
        raise ValueError(f"Task {task_id}: missing required meta.species_slug")
    if task_species_slug != normalize_species_slug(species_slug):
        raise ValueError(
            f"Task {task_id}: task species_slug {task_species_slug!r} does not match requested species {species_slug!r}"
        )

    raw_image_id = data.get("image_id")
    image_id = normalize_choice(str(raw_image_id or ""))
    if image_id is None:
        raise ValueError(f"Task {task_id}: missing required data.image_id")

    raw_source_filename = str(data.get("source_filename") or "").strip()
    if not raw_source_filename:
        raise ValueError(f"Task {task_id}: missing required data.source_filename")

    raw_original_relpath = str(meta.get("original_relative_path") or "").strip()
    if not raw_original_relpath:
        raise ValueError(f"Task {task_id}: missing required meta.original_relative_path")
    original_relpath = normalize_relative_path(raw_original_relpath, field_name="meta.original_relative_path")

    raw_served_relpath = str(meta.get("served_relative_path") or "").strip()
    if not raw_served_relpath:
        raise ValueError(f"Task {task_id}: missing required meta.served_relative_path")
    compressed_relpath = normalize_relative_path(raw_served_relpath, field_name="meta.served_relative_path")

    compression_profile = str(meta.get("compression_profile") or "").strip()
    if not compression_profile:
        raise ValueError(f"Task {task_id}: missing required meta.compression_profile")

    localfiles_image = str(data.get("image") or "").strip()
    if not localfiles_image:
        raise ValueError(f"Task {task_id}: missing required data.image")
    labelstudio_localfiles_relpath = parse_localfiles_relpath(localfiles_image)
    expected_localfiles_relpath = normalize_relative_path(
        f"birds/{task_species_slug}/{compressed_relpath}",
        field_name="expected labelstudio local-files relpath",
    )
    if labelstudio_localfiles_relpath != expected_localfiles_relpath:
        raise ValueError(
            f"Task {task_id}: local-files relpath {labelstudio_localfiles_relpath!r} does not match meta.served_relative_path {compressed_relpath!r}"
        )

    if pathlib.Path(raw_source_filename).name != pathlib.Path(original_relpath).name:
        raise ValueError(
            f"Task {task_id}: data.source_filename {raw_source_filename!r} does not match meta.original_relative_path {original_relpath!r}"
        )
    if pathlib.Path(image_id).name != pathlib.Path(original_relpath).stem:
        raise ValueError(
            f"Task {task_id}: data.image_id {image_id!r} does not match meta.original_relative_path {original_relpath!r}"
        )

    return TaskImageMetadata(
        species_slug=task_species_slug,
        image_id=image_id,
        source_filename=raw_source_filename,
        original_relpath=original_relpath,
        compressed_relpath=compressed_relpath,
        labelstudio_localfiles_relpath=labelstudio_localfiles_relpath,
        compression_profile=compression_profile,
        site_id=normalize_choice(data.get("site_id")),
    )


def bbox_from_region(region: dict[str, Any]) -> tuple[float, float, float, float]:
    value = region.get("value") or {}
    x = float(value.get("x", 0.0)) / 100.0
    y = float(value.get("y", 0.0)) / 100.0
    w = float(value.get("width", 0.0)) / 100.0
    h = float(value.get("height", 0.0)) / 100.0
    return clamp01(x), clamp01(y), clamp01(w), clamp01(h)


def pick_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    annotations = task.get("annotations") or []
    kept = [ann for ann in annotations if isinstance(ann, dict) and not ann.get("was_cancelled")]
    if not kept:
        return None
    kept.sort(key=lambda ann: (str(ann.get("updated_at") or ""), str(ann.get("created_at") or ""), int(ann.get("id") or 0)))
    return kept[-1]


def require_choice_value(item: dict[str, Any], *, field: str, task_id: str, annotation_id: str) -> str:
    values = (((item.get("value") or {}).get("choices")) or [])
    if not values:
        raise ValueError(f"Task {task_id} annotation {annotation_id}: field {field!r} has an empty choices payload")
    choice = normalize_choice(str(values[0]))
    if choice is None:
        raise ValueError(f"Task {task_id} annotation {annotation_id}: field {field!r} has an empty choice value")
    return choice


def validate_choice_value(*, field: str, value: str, schema: CurrentAnnotationSchema, task_id: str, annotation_id: str) -> None:
    allowed = schema.choice_fields.get(field)
    if allowed is None:
        raise ValueError(f"Task {task_id} annotation {annotation_id}: unexpected field {field!r}")
    if value not in allowed:
        raise ValueError(
            f"Task {task_id} annotation {annotation_id}: field {field!r} value {value!r} is not in the current schema {sorted(allowed)!r}"
        )


def extract_task_rows(
    *,
    task: dict[str, Any],
    annotation_version: str,
    species_slug: str,
    schema: CurrentAnnotationSchema,
) -> tuple[dict[str, Any] | None, list[BirdRow]]:
    task_id = normalize_choice(str(task.get("id")))
    if task_id is None:
        raise ValueError("Task id is required")
    annotation = pick_annotation(task)
    if annotation is None:
        return None, []

    annotation_id = normalize_choice(str(annotation.get("id")))
    if annotation_id is None:
        raise ValueError(f"Task {task_id}: latest annotation is missing an id")

    image_meta = image_metadata_from_task(task, species_slug=species_slug)
    results: list[dict[str, Any]] = annotation.get("result") or []
    region_state: dict[str, dict[str, Any]] = {}

    for item in results:
        if not isinstance(item, dict):
            raise ValueError(f"Task {task_id} annotation {annotation_id}: result item is not an object")
        item_type = normalize_choice(item.get("type"))
        from_name = normalize_choice(item.get("from_name"))
        if item_type == "rectanglelabels":
            if from_name != schema.rectangle_from_name:
                raise ValueError(
                    f"Task {task_id} annotation {annotation_id}: unexpected rectangle field {from_name!r}; expected {schema.rectangle_from_name!r}"
                )
            region_id = normalize_choice(str(item.get("id")))
            if region_id is None:
                raise ValueError(f"Task {task_id} annotation {annotation_id}: region is missing an id")
            labels = [normalize_choice(value) for value in (item.get("value") or {}).get("rectanglelabels", [])]
            labels = [value for value in labels if value is not None]
            if not labels:
                raise ValueError(f"Task {task_id} annotation {annotation_id}: region {region_id} has no rectangle label")
            invalid_labels = [value for value in labels if value not in schema.rectangle_labels]
            if invalid_labels:
                raise ValueError(
                    f"Task {task_id} annotation {annotation_id}: region {region_id} uses non-current rectangle labels {invalid_labels!r}"
                )
            region_state[region_id] = {
                "bbox": bbox_from_region(item),
                "isbird": None,
                "readability": None,
                "specie": None,
                "behavior": None,
                "substrate": None,
                "stance": None,
            }
        elif item_type not in {"choices", None}:
            raise ValueError(
                f"Task {task_id} annotation {annotation_id}: unexpected result item type {item.get('type')!r}"
            )

    for item in results:
        if not isinstance(item, dict):
            continue
        item_type = normalize_choice(item.get("type"))
        if item_type != "choices":
            continue
        field = normalize_choice(item.get("from_name"))
        if field is None:
            raise ValueError(f"Task {task_id} annotation {annotation_id}: choice item is missing from_name")
        if field not in schema.choice_fields:
            raise ValueError(f"Task {task_id} annotation {annotation_id}: unexpected field {field!r}")
        parent_id = normalize_choice(str(item.get("parentID") or item.get("parent_id") or item.get("id")))
        if parent_id is None or parent_id not in region_state:
            raise ValueError(
                f"Task {task_id} annotation {annotation_id}: field {field!r} references unknown region {item.get('parentID') or item.get('parent_id') or item.get('id')!r}"
            )
        choice = require_choice_value(item, field=field, task_id=task_id, annotation_id=annotation_id)
        validate_choice_value(field=field, value=choice, schema=schema, task_id=task_id, annotation_id=annotation_id)
        region_state[parent_id][field] = choice

    birds: list[BirdRow] = []
    image_usable = False
    for region_id in sorted(region_state):
        state = region_state[region_id]
        isbird = state["isbird"]
        if isbird is None:
            raise ValueError(f"Task {task_id} annotation {annotation_id}: region {region_id} is missing required field 'isbird'")

        readability = state["readability"]
        specie = state["specie"]
        behavior = state["behavior"]
        substrate = state["substrate"]
        stance = state["stance"]

        if isbird == "yes":
            image_usable = True
            if readability is None:
                raise ValueError(
                    f"Task {task_id} annotation {annotation_id}: region {region_id} is missing required field 'readability'"
                )
            if specie is None:
                raise ValueError(
                    f"Task {task_id} annotation {annotation_id}: region {region_id} is missing required field 'specie'"
                )

            if readability == "unreadable" or specie == "incorrect":
                behavior = None
                substrate = None
                stance = None
            else:
                if behavior is None:
                    behavior = "unsure"
                if substrate is None:
                    substrate = "unsure"
                if behavior not in BEHAVIOR_VALUES:
                    raise ValueError(
                        f"Task {task_id} annotation {annotation_id}: region {region_id} has invalid behavior {behavior!r}"
                    )
                if substrate not in SUBSTRATE_VALUES:
                    raise ValueError(
                        f"Task {task_id} annotation {annotation_id}: region {region_id} has invalid substrate {substrate!r}"
                    )
                if behavior in {"resting", "backresting"} and substrate in {"bare_ground", "water", "unsure"}:
                    if stance is None:
                        stance = "unsure"
                else:
                    stance = None
                if stance is not None and stance not in STANCE_VALUES:
                    raise ValueError(
                        f"Task {task_id} annotation {annotation_id}: region {region_id} has invalid stance {stance!r}"
                    )
        elif isbird == "no":
            readability = None
            specie = None
            behavior = None
            substrate = None
            stance = None
        else:
            raise ValueError(f"Task {task_id} annotation {annotation_id}: region {region_id} has invalid isbird {isbird!r}")

        x, y, w, h = state["bbox"]
        birds.append(
            BirdRow(
                annotation_version=annotation_version,
                task_id=task_id,
                annotation_id=annotation_id,
                region_id=region_id,
                image_id=image_meta.image_id,
                bird_id=bird_row_id(task_id, region_id),
                bbox_x=x,
                bbox_y=y,
                bbox_w=w,
                bbox_h=h,
                bbox_x_px=None,
                bbox_y_px=None,
                bbox_w_px=None,
                bbox_h_px=None,
                isbird=isbird,
                readability=readability,
                specie=specie,
                behavior=behavior,
                substrate=substrate,
                stance=stance,
            )
        )

    image_row = {
        "annotation_version": annotation_version,
        "task_id": task_id,
        "annotation_id": annotation_id,
        "species_slug": image_meta.species_slug,
        "image_id": image_meta.image_id,
        "source_filename": image_meta.source_filename,
        "original_relpath": image_meta.original_relpath,
        "compressed_relpath": image_meta.compressed_relpath,
        "labelstudio_localfiles_relpath": image_meta.labelstudio_localfiles_relpath,
        "compression_profile": image_meta.compression_profile,
        "image_usable": bool(image_usable),
        "site_id": image_meta.site_id,
    }
    return image_row, birds


def write_deterministic_parquet(df, out_path: pathlib.Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for parquet output. Install it before running extraction.") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=False,
        version="2.6",
    )


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pandas_frame(data, columns: list[str]):
    import pandas as pd

    if data:
        return pd.DataFrame(data)
    return pd.DataFrame(columns=columns)


def counts_by_value(frame, column: str) -> dict[str, int]:
    values = frame[column].fillna("<null>").value_counts().sort_index().to_dict()
    return {str(key): int(value) for key, value in values.items()}


def normalize_output_frames(image_rows: list[dict[str, Any]], bird_rows: list[BirdRow]):
    images_df = pandas_frame(
        image_rows,
        [
            "annotation_version",
            "task_id",
            "annotation_id",
            "species_slug",
            "image_id",
            "source_filename",
            "original_relpath",
            "compressed_relpath",
            "labelstudio_localfiles_relpath",
            "compression_profile",
            "image_usable",
            "site_id",
        ],
    )
    if not images_df.empty:
        duplicates = images_df[images_df["image_id"].duplicated(keep=False)]
        if not duplicates.empty:
            duplicate_ids = sorted(set(str(value) for value in duplicates["image_id"].tolist()))
            raise ValueError(f"Duplicate image_id values in annotated tasks are not allowed: {duplicate_ids!r}")
        images_df["task_id"] = images_df["task_id"].astype(str)
        images_df["annotation_id"] = images_df["annotation_id"].astype(str)
        images_df["species_slug"] = images_df["species_slug"].astype(str)
        images_df["image_id"] = images_df["image_id"].astype(str)
        images_df["source_filename"] = images_df["source_filename"].astype(str)
        for column in ("original_relpath", "compressed_relpath", "labelstudio_localfiles_relpath", "compression_profile"):
            images_df[column] = images_df[column].astype(str)
        images_df = images_df.sort_values(["image_id", "task_id", "annotation_id"]).reset_index(drop=True)

    birds_df = pandas_frame(
        [asdict(row) for row in bird_rows],
        [
            "annotation_version",
            "task_id",
            "annotation_id",
            "region_id",
            "image_id",
            "bird_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "bbox_x_px",
            "bbox_y_px",
            "bbox_w_px",
            "bbox_h_px",
            "isbird",
            "readability",
            "specie",
            "behavior",
            "substrate",
            "stance",
        ],
    )
    if not birds_df.empty:
        for column in ("task_id", "annotation_id", "region_id", "image_id", "bird_id"):
            birds_df[column] = birds_df[column].astype(str)
        birds_df = birds_df.sort_values(["image_id", "task_id", "region_id", "bird_id"]).reset_index(drop=True)
    return images_df, birds_df


def is_missing(value: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except Exception:  # noqa: BLE001
        return value is None


def json_value(value: Any) -> Any:
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if is_missing(value):
        return None
    return value


def values_equal(left: Any, right: Any) -> bool:
    if is_missing(left) and is_missing(right):
        return True
    return json_value(left) == json_value(right)


def label_counts(frame) -> dict[str, dict[str, int]]:
    return {field: counts_by_value(frame, field) for field in LABEL_FIELDS}


def applicability_counts(frame) -> dict[str, int]:
    if frame.empty:
        return {
            "all_regions": 0,
            "isbird_yes": 0,
            "visibility_usable": 0,
            "species_usable": 0,
            "behavior_labeled": 0,
            "substrate_labeled": 0,
            "stance_labeled": 0,
        }

    isbird_yes = frame["isbird"] == "yes"
    visibility_usable = isbird_yes & frame["readability"].isin(["readable", "occluded"])
    species_usable = visibility_usable & frame["specie"].isin(["correct", "unsure"])

    return {
        "all_regions": int(len(frame)),
        "isbird_yes": int(isbird_yes.sum()),
        "visibility_usable": int(visibility_usable.sum()),
        "species_usable": int(species_usable.sum()),
        "behavior_labeled": int(frame["behavior"].notna().sum()),
        "substrate_labeled": int(frame["substrate"].notna().sum()),
        "stance_labeled": int(frame["stance"].notna().sum()),
    }


def current_extract_payload(
    *,
    annotation_version: str,
    export_json: pathlib.Path,
    schema: CurrentAnnotationSchema,
    raw_metadata: dict[str, Any] | None,
    stats: dict[str, int],
    images_df,
    birds_df,
) -> dict[str, Any]:
    counts = {
        "tasks_total_raw": int(stats["tasks_total_raw"]),
        "tasks_with_any_annotations": int(stats["tasks_with_any_annotations"]),
        "tasks_with_cancelled_annotations": int(stats["tasks_with_cancelled_annotations"]),
        "tasks_with_kept_annotation": int(stats["tasks_with_kept_annotation"]),
        "tasks_skipped_without_kept_annotation": int(stats["tasks_skipped_without_kept_annotation"]),
        "annotated_tasks_with_zero_regions": int(stats["annotated_tasks_with_zero_regions"]),
        "images_rows": int(len(images_df)),
        "birds_rows": int(len(birds_df)),
        "images_usable_true": int(images_df["image_usable"].astype(bool).sum()) if not images_df.empty else 0,
        "images_usable_false": int((~images_df["image_usable"].astype(bool)).sum()) if not images_df.empty else 0,
    }
    null_counts = {
        field: int(birds_df[field].isna().sum()) if not birds_df.empty else 0
        for field in LABEL_FIELDS
    }
    return {
        "annotation_version": annotation_version,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "export_json": str(export_json),
        "raw_export_metadata": raw_metadata,
        "schema_fields": {
            "rectangle": schema.rectangle_from_name,
            "choices": {key: sorted(values) for key, values in sorted(schema.choice_fields.items())},
        },
        "counts": counts,
        "field_counts": label_counts(birds_df) if not birds_df.empty else {field: {} for field in LABEL_FIELDS},
        "null_counts": null_counts,
        "applicability_counts": applicability_counts(birds_df),
    }


def compare_frames(current_images, current_birds, previous_dir: pathlib.Path | None) -> dict[str, Any]:
    import pandas as pd

    if previous_dir is None:
        return {
            "previous_annotation_version": None,
            "images": {"added": 0, "removed": 0, "changed": 0, "unchanged": int(len(current_images))},
            "birds": {"added": 0, "removed": 0, "changed": 0, "unchanged": int(len(current_birds))},
            "field_change_counts": {"bbox": 0, **{field: 0 for field in LABEL_FIELDS}},
            "label_delta_counts": {field: counts_by_value(current_birds, field) if not current_birds.empty else {} for field in LABEL_FIELDS},
            "image_changes": {"added": [], "removed": [], "changed": []},
            "bird_changes": {"added": [], "removed": [], "changed": []},
        }

    prev_images_path = previous_dir / "images_labels.parquet"
    prev_birds_path = previous_dir / "birds.parquet"
    previous_images = pd.read_parquet(prev_images_path) if prev_images_path.exists() else pd.DataFrame(columns=current_images.columns)
    previous_birds = pd.read_parquet(prev_birds_path) if prev_birds_path.exists() else pd.DataFrame(columns=current_birds.columns)

    image_compare_fields = [
        "species_slug",
        "source_filename",
        "original_relpath",
        "compressed_relpath",
        "labelstudio_localfiles_relpath",
        "compression_profile",
        "image_usable",
        "site_id",
    ]
    current_images_by_key = {str(row["image_id"]): row for _, row in current_images.iterrows()}
    previous_images_by_key = {str(row["image_id"]): row for _, row in previous_images.iterrows()}
    current_birds_by_key = {str(row["bird_id"]): row for _, row in current_birds.iterrows()}
    previous_birds_by_key = {str(row["bird_id"]): row for _, row in previous_birds.iterrows()}

    added_images = sorted(set(current_images_by_key) - set(previous_images_by_key))
    removed_images = sorted(set(previous_images_by_key) - set(current_images_by_key))
    changed_images: list[dict[str, Any]] = []
    for image_id in sorted(set(current_images_by_key) & set(previous_images_by_key)):
        previous_row = previous_images_by_key[image_id]
        current_row = current_images_by_key[image_id]
        changed_fields = {}
        for field in image_compare_fields:
            prev_value = previous_row[field]
            cur_value = current_row[field]
            if not values_equal(prev_value, cur_value):
                changed_fields[field] = {"previous": json_value(prev_value), "current": json_value(cur_value)}
        if changed_fields:
            changed_images.append({"image_id": image_id, "changed_fields": changed_fields})

    bird_compare_fields = ["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", *LABEL_FIELDS]
    added_birds = sorted(set(current_birds_by_key) - set(previous_birds_by_key))
    removed_birds = sorted(set(previous_birds_by_key) - set(current_birds_by_key))
    changed_birds: list[dict[str, Any]] = []
    field_change_counts = {"bbox": 0, **{field: 0 for field in LABEL_FIELDS}}
    for bird_id in sorted(set(current_birds_by_key) & set(previous_birds_by_key)):
        previous_row = previous_birds_by_key[bird_id]
        current_row = current_birds_by_key[bird_id]
        changed_fields = {}
        bbox_changed = False
        for field in bird_compare_fields:
            prev_value = previous_row[field]
            cur_value = current_row[field]
            if values_equal(prev_value, cur_value):
                continue
            changed_fields[field] = {"previous": json_value(prev_value), "current": json_value(cur_value)}
            if field.startswith("bbox_"):
                bbox_changed = True
            elif field in LABEL_FIELDS:
                field_change_counts[field] += 1
        if bbox_changed:
            field_change_counts["bbox"] += 1
        if changed_fields:
            changed_birds.append(
                {
                    "bird_id": bird_id,
                    "image_id": json_value(current_row["image_id"]),
                    "changed_fields": changed_fields,
                }
            )

    current_counts = label_counts(current_birds) if not current_birds.empty else {field: {} for field in LABEL_FIELDS}
    previous_counts = label_counts(previous_birds) if not previous_birds.empty else {field: {} for field in LABEL_FIELDS}
    label_delta_counts: dict[str, dict[str, int]] = {}
    for field in LABEL_FIELDS:
        delta: dict[str, int] = {}
        for label in sorted(set(current_counts[field]) | set(previous_counts[field])):
            delta[label] = int(current_counts[field].get(label, 0)) - int(previous_counts[field].get(label, 0))
        label_delta_counts[field] = delta

    return {
        "previous_annotation_version": previous_dir.name,
        "images": {
            "added": len(added_images),
            "removed": len(removed_images),
            "changed": len(changed_images),
            "unchanged": int(len(set(current_images_by_key) & set(previous_images_by_key)) - len(changed_images)),
        },
        "birds": {
            "added": len(added_birds),
            "removed": len(removed_birds),
            "changed": len(changed_birds),
            "unchanged": int(len(set(current_birds_by_key) & set(previous_birds_by_key)) - len(changed_birds)),
        },
        "field_change_counts": field_change_counts,
        "label_delta_counts": label_delta_counts,
        "image_changes": {
            "added": added_images,
            "removed": removed_images,
            "changed": changed_images,
        },
        "bird_changes": {
            "added": added_birds,
            "removed": removed_birds,
            "changed": changed_birds,
        },
    }


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render extraction plots.") from exc
    return plt


def save_plot(fig, base_path: pathlib.Path, *, tight_rect: tuple[float, float, float, float] | None = None) -> dict[str, str]:
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    fig.tight_layout(rect=tight_rect)
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return {"png": png_path.name, "svg": svg_path.name}


def annotate_bars(ax, bars, texts: list[str], *, padding: float = 3.0, fontsize: int = 9) -> None:
    for bar, text in zip(bars, texts):
        height = bar.get_height()
        baseline = max(height, 0)
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2.0, baseline),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def format_count(count: int) -> str:
    return f"{count:,}"


def format_count_and_pct(count: int, total: int) -> str:
    if total <= 0:
        return format_count(count)
    return f"{count:,}\n{(count / total) * 100:.1f}%"


def add_bar_headroom(ax, values: list[int | float], *, extra_ratio: float = 0.18, min_top: float = 1.0) -> None:
    if not values:
        return
    top = max(float(max(values)) * (1.0 + extra_ratio), min_top)
    bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom, max(top, current_top))


def render_current_plots(*, plots_dir: pathlib.Path, current_payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    plt = import_pyplot()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, str]] = {}

    counts = current_payload["counts"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    keys = [
        "tasks_total_raw",
        "tasks_with_kept_annotation",
        "images_rows",
        "birds_rows",
        "images_usable_true",
        "images_usable_false",
    ]
    values = [counts[key] for key in keys]
    bars = ax.bar(range(len(keys)), values, color="#2F6B8A")
    add_bar_headroom(ax, values, extra_ratio=0.15)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(
        ["raw tasks", "annotated tasks", "images", "birds", "usable images", "non-usable images"],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("count")
    ax.set_title("Current Extract Summary")
    annotate_bars(ax, bars, [format_count(value) for value in values], padding=4.0, fontsize=10)
    out["current_extract_summary"] = save_plot(fig, plots_dir / "current_extract_summary")
    plt.close(fig)

    stage_counts = current_payload["applicability_counts"]
    stage_order = [
        ("all_regions", "all regions"),
        ("isbird_yes", "is bird"),
        ("visibility_usable", "readable/occluded"),
        ("species_usable", "species usable"),
        ("behavior_labeled", "behavior labeled"),
        ("substrate_labeled", "substrate labeled"),
        ("stance_labeled", "stance labeled"),
    ]
    total_regions = max(stage_counts["all_regions"], 1)
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    labels = [label for _, label in stage_order]
    values = [stage_counts[key] for key, _ in stage_order]
    previous_values = [values[0], *values[:-1]]
    bars = ax.bar(range(len(labels)), values, color="#C66B3D")
    add_bar_headroom(ax, values, extra_ratio=0.26)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("birds / regions")
    ax.set_title("Applicability Funnel")
    funnel_texts = []
    for value, previous_value in zip(values, previous_values):
        total_pct = (value / total_regions) * 100 if total_regions else 0.0
        previous_pct = (value / previous_value) * 100 if previous_value else 0.0
        funnel_texts.append(f"{value:,}\n{total_pct:.1f}% total\n{previous_pct:.1f}% prev")
    annotate_bars(ax, bars, funnel_texts, padding=4.0, fontsize=9)
    out["current_null_counts"] = save_plot(fig, plots_dir / "current_null_counts")
    plt.close(fig)

    field_counts = current_payload["field_counts"]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.5))
    for ax, field in zip(axes.flat, LABEL_FIELDS):
        counts_map = {key: value for key, value in field_counts[field].items() if key != "<null>"}
        if not counts_map:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        labels = list(counts_map)
        values = [counts_map[label] for label in labels]
        bars = ax.bar(range(len(labels)), values, color="#478A6E")
        add_bar_headroom(ax, values, extra_ratio=0.22)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        total = sum(values)
        annotate_bars(ax, bars, [format_count_and_pct(value, total) for value in values], padding=3.0, fontsize=8)
        ax.set_title(f"{field} (n={total:,})")
    fig.suptitle("Current Label Distributions", y=0.995)
    out["current_label_distributions"] = save_plot(
        fig,
        plots_dir / "current_label_distributions",
        tight_rect=(0.0, 0.0, 1.0, 0.965),
    )
    plt.close(fig)
    return out


def render_comparison_plots(*, plots_dir: pathlib.Path, comparison_payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    plt = import_pyplot()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, str]] = {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["images added", "images removed", "images changed", "birds added", "birds removed", "birds changed"]
    values = [
        comparison_payload["images"]["added"],
        comparison_payload["images"]["removed"],
        comparison_payload["images"]["changed"],
        comparison_payload["birds"]["added"],
        comparison_payload["birds"]["removed"],
        comparison_payload["birds"]["changed"],
    ]
    ax.bar(range(len(labels)), values, color="#7A4EA3")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Version-to-Version Row Deltas")
    out["comparison_row_deltas"] = save_plot(fig, plots_dir / "comparison_row_deltas")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    field_change_counts = comparison_payload["field_change_counts"]
    labels = list(field_change_counts)
    ax.bar(range(len(labels)), [field_change_counts[label] for label in labels], color="#D18B47")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("changed birds")
    ax.set_title("Changed Bird Fields")
    out["comparison_field_changes"] = save_plot(fig, plots_dir / "comparison_field_changes")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for ax, field in zip(axes.flat, LABEL_FIELDS):
        deltas = {key: value for key, value in comparison_payload["label_delta_counts"][field].items() if value != 0}
        if not deltas:
            ax.text(0.5, 0.5, "No delta", ha="center", va="center")
            ax.set_axis_off()
            continue
        labels = list(deltas)
        values = [deltas[label] for label in labels]
        colors = ["#3D8A68" if value >= 0 else "#B4514D" for value in values]
        ax.bar(range(len(labels)), values, color=colors)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(field)
    fig.suptitle("Label Count Deltas vs Previous Extract", y=1.02)
    out["comparison_label_deltas"] = save_plot(fig, plots_dir / "comparison_label_deltas")
    plt.close(fig)
    return out


def embed_plot_block(title: str, plot_name: str, plot_files: dict[str, dict[str, str]]) -> list[str]:
    files = plot_files[plot_name]
    return [
        f"### {title}",
        "",
        f"![{title}](plots/{files['png']})",
        "",
        f"`SVG:` `plots/{files['svg']}`",
        "",
    ]


def write_extract_report_md(
    *,
    path: pathlib.Path,
    current_payload: dict[str, Any],
    comparison_payload: dict[str, Any],
    plot_files: dict[str, dict[str, str]],
) -> None:
    counts = current_payload["counts"]
    lines = [
        f"# Annotation Extract: {current_payload['annotation_version']}",
        "",
        f"- Schema version: `{current_payload['schema_version']}`",
        f"- Normalization policy: `{current_payload['normalization_policy_version']}`",
        f"- Source export: `{current_payload['export_json']}`",
        f"- Previous extract: `{comparison_payload['previous_annotation_version'] or 'none'}`",
        f"- Raw tasks: `{counts['tasks_total_raw']}`",
        f"- Annotated tasks kept: `{counts['tasks_with_kept_annotation']}`",
        f"- Annotated images: `{counts['images_rows']}`",
        f"- Bird rows: `{counts['birds_rows']}`",
        f"- Images with usable birds: `{counts['images_usable_true']}`",
        "",
    ]
    lines.extend(embed_plot_block("Current Extract Summary", "current_extract_summary", plot_files))
    lines.extend(embed_plot_block("Applicability Funnel", "current_null_counts", plot_files))
    lines.extend(embed_plot_block("Current Label Distributions", "current_label_distributions", plot_files))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report_md(
    *,
    path: pathlib.Path,
    annotation_version: str,
    comparison_payload: dict[str, Any],
    plot_files: dict[str, dict[str, str]],
) -> None:
    lines = [
        f"# Annotation Comparison: {annotation_version}",
        "",
        f"- Previous extract: `{comparison_payload['previous_annotation_version'] or 'none'}`",
        f"- Images added / removed / changed: `{comparison_payload['images']['added']}` / `{comparison_payload['images']['removed']}` / `{comparison_payload['images']['changed']}`",
        f"- Birds added / removed / changed: `{comparison_payload['birds']['added']}` / `{comparison_payload['birds']['removed']}` / `{comparison_payload['birds']['changed']}`",
        "",
    ]
    lines.extend(embed_plot_block("Version-to-Version Row Deltas", "comparison_row_deltas", plot_files))
    lines.extend(embed_plot_block("Changed Bird Fields", "comparison_field_changes", plot_files))
    lines.extend(embed_plot_block("Label Count Deltas", "comparison_label_deltas", plot_files))

    changed_birds = comparison_payload["bird_changes"]["changed"][:10]
    if changed_birds:
        lines.append("## Sample Changed Bird Rows")
        lines.append("")
        for item in changed_birds:
            lines.append(f"- `{item['bird_id']}` fields changed: `{sorted(item['changed_fields'])}`")
        lines.append("")

    changed_images = comparison_payload["image_changes"]["changed"][:10]
    if changed_images:
        lines.append("## Sample Changed Images")
        lines.append("")
        for item in changed_images:
            lines.append(f"- `{item['image_id']}` fields changed: `{sorted(item['changed_fields'])}`")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_export(
    *,
    export_json: pathlib.Path,
    annotation_version: str,
    data_home: pathlib.Path,
    species_slug: str,
    label_config: pathlib.Path = DEFAULT_LABEL_CONFIG,
    raw_metadata: dict[str, Any] | None = None,
) -> NormalizationResult:
    try:
        import pandas as pd  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required for annotation normalization.") from exc

    if not export_json.exists():
        raise FileNotFoundError(export_json)
    if not label_config.exists():
        raise FileNotFoundError(label_config)

    layout = ensure_layout(data_home, species_slug)
    out_dir = layout.labelstudio_normalized / annotation_version
    out_dir.mkdir(parents=True, exist_ok=False)
    payload = json.loads(export_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Label Studio export must be a list of tasks")

    schema = load_current_schema(label_config)
    stats = {
        "tasks_total_raw": len(payload),
        "tasks_with_any_annotations": 0,
        "tasks_with_cancelled_annotations": 0,
        "tasks_with_kept_annotation": 0,
        "tasks_skipped_without_kept_annotation": 0,
        "annotated_tasks_with_zero_regions": 0,
    }

    image_rows: list[dict[str, Any]] = []
    bird_rows: list[BirdRow] = []
    for task in payload:
        annotations = task.get("annotations") or []
        if annotations:
            stats["tasks_with_any_annotations"] += 1
        if any(isinstance(ann, dict) and ann.get("was_cancelled") for ann in annotations):
            stats["tasks_with_cancelled_annotations"] += 1
        image_row, birds = extract_task_rows(
            task=task,
            annotation_version=annotation_version,
            species_slug=species_slug,
            schema=schema,
        )
        if image_row is None:
            stats["tasks_skipped_without_kept_annotation"] += 1
            continue
        stats["tasks_with_kept_annotation"] += 1
        if not birds:
            stats["annotated_tasks_with_zero_regions"] += 1
        image_rows.append(image_row)
        bird_rows.extend(birds)

    images_df, birds_df = normalize_output_frames(image_rows, bird_rows)

    birds_out = out_dir / "birds.parquet"
    images_out = out_dir / "images_labels.parquet"
    write_deterministic_parquet(images_df, images_out)
    write_deterministic_parquet(birds_df, birds_out)

    current_payload = current_extract_payload(
        annotation_version=annotation_version,
        export_json=export_json,
        schema=schema,
        raw_metadata=raw_metadata,
        stats=stats,
        images_df=images_df,
        birds_df=birds_df,
    )

    previous_dir = find_previous_version_dir(layout.labelstudio_normalized, "ann", annotation_version)
    comparison_payload = compare_frames(images_df, birds_df, previous_dir)

    plots_dir = out_dir / "plots"
    current_plots = render_current_plots(plots_dir=plots_dir, current_payload=current_payload)
    comparison_plots = render_comparison_plots(plots_dir=plots_dir, comparison_payload=comparison_payload)
    plot_files = {**current_plots, **comparison_plots}

    manifest = {
        "annotation_version": annotation_version,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "species_slug": layout.species_slug,
        "species_root": str(layout.root),
        "source_export_json": str(export_json),
        "raw_export_metadata": raw_metadata,
        "previous_annotation_version": comparison_payload["previous_annotation_version"],
        "counts": current_payload["counts"],
        "outputs": {
            "birds_parquet": birds_out.name,
            "images_labels_parquet": images_out.name,
            "extract_report_json": "extract_report.json",
            "extract_report_md": "extract_report.md",
            "comparison_to_previous_json": "comparison_to_previous.json",
            "comparison_to_previous_md": "comparison_to_previous.md",
            "plots": {key: value for key, value in sorted(plot_files.items())},
        },
    }

    manifest_out = out_dir / "manifest.json"
    extract_report_json = out_dir / "extract_report.json"
    extract_report_md = out_dir / "extract_report.md"
    comparison_json = out_dir / "comparison_to_previous.json"
    comparison_md = out_dir / "comparison_to_previous.md"

    write_json(manifest_out, manifest)
    write_json(extract_report_json, current_payload)
    write_json(comparison_json, comparison_payload)
    write_extract_report_md(
        path=extract_report_md,
        current_payload=current_payload,
        comparison_payload=comparison_payload,
        plot_files=plot_files,
    )
    write_comparison_report_md(
        path=comparison_md,
        annotation_version=annotation_version,
        comparison_payload=comparison_payload,
        plot_files=plot_files,
    )

    return NormalizationResult(
        annotation_version=annotation_version,
        out_dir=out_dir,
        birds_out=birds_out,
        images_out=images_out,
        manifest_out=manifest_out,
        extract_report_json=extract_report_json,
        extract_report_md=extract_report_md,
        comparison_json=comparison_json,
        comparison_md=comparison_md,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_metadata_path = pathlib.Path(args.raw_metadata_json).expanduser().resolve() if args.raw_metadata_json else None
    result = normalize_export(
        export_json=pathlib.Path(args.export_json).expanduser().resolve(),
        annotation_version=args.annotation_version,
        data_home=pathlib.Path(args.data_home).expanduser().resolve(),
        species_slug=args.species_slug,
        label_config=pathlib.Path(args.label_config).expanduser().resolve(),
        raw_metadata=load_raw_metadata(raw_metadata_path),
    )
    print(f"annotation_version={result.annotation_version}")
    print(f"output_dir={result.out_dir}")
    print(f"birds_out={result.birds_out}")
    print(f"images_out={result.images_out}")
    print(f"manifest_out={result.manifest_out}")
    print(f"extract_report_json={result.extract_report_json}")
    print(f"extract_report_md={result.extract_report_md}")
    print(f"comparison_json={result.comparison_json}")
    print(f"comparison_md={result.comparison_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
