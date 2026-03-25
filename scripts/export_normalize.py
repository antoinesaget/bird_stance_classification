#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import urllib.parse
from dataclasses import dataclass
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout
from birdsys.training.attributes import (
    BEHAVIOR_TO_ID,
    STANCE_TO_ID,
    SUBSTRATE_TO_ID,
    normalize_behavior,
    normalize_choice,
    normalize_stance,
    normalize_substrate,
)


@dataclass
class BirdRow:
    annotation_version: str
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Label Studio export JSON into deterministic parquet")
    parser.add_argument("--export-json", required=True, help="Path to ann_vXXX.json export")
    parser.add_argument("--annotation-version", required=True, help="Annotation version, e.g. ann_v001")
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def image_id_from_task(task: dict[str, Any]) -> str:
    data = task.get("data", {})
    candidates = [data.get("image"), data.get("filepath"), data.get("path"), task.get("image")]
    for candidate in candidates:
        if not candidate:
            continue
        s = str(candidate)

        if "?d=" in s:
            parsed = urllib.parse.urlparse(s)
            query = urllib.parse.parse_qs(parsed.query)
            local = query.get("d", [""])[0]
            if local:
                return pathlib.Path(local).stem

        if s.startswith("http://") or s.startswith("https://"):
            path = urllib.parse.urlparse(s).path
            if path:
                return pathlib.Path(path).stem

        return pathlib.Path(s).stem

    raise ValueError(f"Cannot infer image_id from task: {task.get('id')}")


def image_path_from_task(task: dict[str, Any], data_root: pathlib.Path) -> str | None:
    def _resolve_local(decoded: str) -> str:
        decoded = decoded.strip()
        if not decoded:
            return ""
        if decoded.startswith("/data/birds_project/"):
            rel = decoded.removeprefix("/data/birds_project/").lstrip("/")
            return str((data_root / rel).resolve())
        if decoded.startswith("data/birds_project/"):
            rel = decoded.removeprefix("data/birds_project/").lstrip("/")
            return str((data_root / rel).resolve())
        if decoded.startswith("birds_project/"):
            rel = decoded.removeprefix("birds_project/").lstrip("/")
            return str((data_root / rel).resolve())
        if decoded.startswith("/"):
            return decoded
        return str((data_root / decoded).resolve())

    data = task.get("data", {})
    candidates = [data.get("image"), data.get("filepath"), data.get("path"), task.get("image")]
    for candidate in candidates:
        if not candidate:
            continue
        s = str(candidate)
        if "?d=" in s:
            parsed = urllib.parse.urlparse(s)
            query = urllib.parse.parse_qs(parsed.query)
            local = query.get("d", [""])[0]
            if local:
                resolved = _resolve_local(urllib.parse.unquote(local))
                if resolved:
                    return resolved
        if s.startswith("http://") or s.startswith("https://"):
            continue
        return _resolve_local(s)
    return None


def first_choice(result: dict[str, Any]) -> str | None:
    choices = (((result.get("value") or {}).get("choices")) or [])
    if not choices:
        return None
    return normalize_choice(str(choices[0]))


def bbox_from_region(region: dict[str, Any]) -> tuple[float, float, float, float]:
    value = region.get("value") or {}
    x = float(value.get("x", 0.0)) / 100.0
    y = float(value.get("y", 0.0)) / 100.0
    w = float(value.get("width", 0.0)) / 100.0
    h = float(value.get("height", 0.0)) / 100.0
    return clamp01(x), clamp01(y), clamp01(w), clamp01(h)


def pick_annotation(task: dict[str, Any]) -> dict[str, Any]:
    annotations = task.get("annotations") or []
    kept = [ann for ann in annotations if isinstance(ann, dict) and not ann.get("was_cancelled")]
    if not kept:
        return {"result": []}
    kept.sort(key=lambda ann: (str(ann.get("updated_at") or ""), str(ann.get("created_at") or ""), int(ann.get("id") or 0)))
    return kept[-1]


def extract_task_rows(
    task: dict[str, Any],
    annotation_version: str,
    migration_stats: dict[str, Any],
    data_root: pathlib.Path,
) -> tuple[dict[str, Any], list[BirdRow]]:
    image_id = image_id_from_task(task)
    data = task.get("data") or {}
    valid_isbird = {"yes", "no"}
    valid_readability = {"readable", "occluded", "unreadable"}
    valid_specie = {"correct", "incorrect", "unsure"}
    valid_behavior = set(BEHAVIOR_TO_ID)
    valid_substrate = set(SUBSTRATE_TO_ID)
    valid_stance = set(STANCE_TO_ID)

    annotation = pick_annotation(task)
    results: list[dict[str, Any]] = annotation.get("result") or []

    region_state: dict[str, dict[str, Any]] = {}

    for item in results:
        item_type = item.get("type")
        from_name = normalize_choice(item.get("from_name"))
        region_id = item.get("id")

        if item_type == "rectanglelabels":
            labels = [normalize_choice(v) for v in (item.get("value") or {}).get("rectanglelabels", [])]
            if "bird" not in labels:
                continue
            if not region_id:
                continue
            x, y, w, h = bbox_from_region(item)
            region_state[region_id] = {
                "bbox": (x, y, w, h),
                "isbird": None,
                "readability": None,
                "specie": None,
                "behavior": None,
                "substrate": None,
                "stance": None,
                "legacy_legs": None,
            }
            continue

        if item_type != "choices":
            continue

        parent_id = item.get("parentID") or item.get("parent_id")
        if not parent_id:
            candidate_id = item.get("id")
            if candidate_id in region_state:
                parent_id = candidate_id
        if not parent_id or parent_id not in region_state:
            continue

        choice = first_choice(item)
        if from_name in {"isbird", "readability", "specie", "behavior", "substrate", "stance", "legs"}:
            target_name = "stance" if from_name == "stance" else from_name
            if from_name == "legs":
                region_state[parent_id]["legacy_legs"] = choice
            region_state[parent_id][target_name] = choice

    birds: list[BirdRow] = []
    image_usable = False
    for i, (region_id, state) in enumerate(sorted(region_state.items(), key=lambda it: it[0])):
        isbird = normalize_choice(state.get("isbird"))
        if isbird is None:
            raise ValueError(f"Missing isbird in image {image_id}, region {region_id}")
        if isbird not in valid_isbird:
            raise ValueError(f"Invalid isbird '{isbird}' for image {image_id}, region {region_id}")
        if isbird == "yes":
            image_usable = True

        readability = normalize_choice(state.get("readability"))
        specie = normalize_choice(state.get("specie"))
        behavior_raw = state.get("behavior")
        substrate_raw = state.get("substrate")
        stance_raw = state.get("stance") or state.get("legacy_legs")

        behavior = normalize_behavior(behavior_raw)
        substrate = normalize_substrate(substrate_raw)
        stance = normalize_stance(stance_raw)

        if normalize_choice(substrate_raw) == "ground":
            migration_stats["mapped_ground_to_bare_ground"] += 1
        if normalize_choice(state.get("legacy_legs")) == "one":
            migration_stats["mapped_legacy_one_to_unipedal"] += 1
        if normalize_choice(state.get("legacy_legs")) == "two":
            migration_stats["mapped_legacy_two_to_bipedal"] += 1

        if isbird == "no":
            readability = None
            specie = None
            behavior = None
            substrate = None
            stance = None
        else:
            if readability is None:
                raise ValueError(f"Missing readability for image {image_id}, region {region_id}")
            if specie is None:
                raise ValueError(f"Missing specie for image {image_id}, region {region_id}")
            if readability not in valid_readability:
                raise ValueError(f"Invalid readability '{readability}' for image {image_id}")
            if specie not in valid_specie:
                raise ValueError(f"Invalid specie '{specie}' for image {image_id}")

            if readability == "unreadable" or specie == "incorrect":
                behavior = None
                substrate = None
                stance = None
            else:
                if behavior is None:
                    behavior = "unsure"
                if substrate is None:
                    substrate = "unsure"

                if behavior not in valid_behavior:
                    raise ValueError(f"Invalid behavior '{behavior}' for image {image_id}")
                if substrate not in valid_substrate:
                    raise ValueError(f"Invalid substrate '{substrate}' for image {image_id}")

                if behavior in {"resting", "backresting"} and substrate in {"bare_ground", "water", "unsure"}:
                    if stance is None:
                        stance = "unsure"
                else:
                    stance = None

                if stance is not None and stance not in valid_stance:
                    raise ValueError(f"Invalid stance '{stance}' for image {image_id}")

        x, y, w, h = state["bbox"]
        birds.append(
            BirdRow(
                annotation_version=annotation_version,
                image_id=image_id,
                bird_id=f"{image_id}:{i:03d}",
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
        "image_id": image_id,
        "image_usable": bool(image_usable),
        "filepath": image_path_from_task(task, data_root),
        "site_id": normalize_choice(data.get("site_id")),
    }
    return image_row, birds


def write_deterministic_parquet(df, out_path: pathlib.Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required. Install dependencies with `uv sync --python 3.11`.") from exc

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


def write_report(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required. Install dependencies with `uv sync --python 3.11`.") from exc

    export_json = pathlib.Path(args.export_json).expanduser().resolve()
    if not export_json.exists():
        raise FileNotFoundError(export_json)

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    layout = ensure_layout(data_root)
    out_dir = layout.labelstudio_normalized / args.annotation_version
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(export_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Label Studio export must be a list of tasks")

    migration_stats: dict[str, Any] = {
        "tasks_total": len(payload),
        "tasks_with_annotations": 0,
        "tasks_with_cancelled_annotations": 0,
        "mapped_ground_to_bare_ground": 0,
        "mapped_legacy_one_to_unipedal": 0,
        "mapped_legacy_two_to_bipedal": 0,
    }

    image_rows = []
    bird_rows: list[BirdRow] = []
    for task in payload:
        annotations = task.get("annotations") or []
        if annotations:
            migration_stats["tasks_with_annotations"] += 1
        if any(isinstance(ann, dict) and ann.get("was_cancelled") for ann in annotations):
            migration_stats["tasks_with_cancelled_annotations"] += 1

        image_row, birds = extract_task_rows(
            task=task,
            annotation_version=args.annotation_version,
            migration_stats=migration_stats,
            data_root=data_root,
        )
        image_rows.append(image_row)
        bird_rows.extend(birds)

    images_df = pd.DataFrame(image_rows).drop_duplicates(subset=["annotation_version", "image_id"])
    images_df = images_df.sort_values(["annotation_version", "image_id"]).reset_index(drop=True)

    bird_columns = [
        "annotation_version",
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
    ]
    birds_df = pd.DataFrame([vars(row) for row in bird_rows])
    if birds_df.empty:
        birds_df = pd.DataFrame(columns=bird_columns)
    birds_df = birds_df[bird_columns].sort_values(["annotation_version", "image_id", "bird_id"]).reset_index(drop=True)

    images_out = out_dir / "images_labels.parquet"
    birds_out = out_dir / "birds.parquet"
    report_out = out_dir / "migration_report.json"

    write_deterministic_parquet(images_df, images_out)
    write_deterministic_parquet(birds_df, birds_out)

    if not birds_df.empty:
        stance_relevant = (
            (birds_df["isbird"] == "yes")
            & birds_df["behavior"].isin(["resting", "backresting"])
            & birds_df["substrate"].isin(["bare_ground", "water", "unsure"])
        )
        field_counts = {}
        for column in ["isbird", "readability", "specie", "behavior", "substrate", "stance"]:
            vc = birds_df[column].fillna("<null>").value_counts().sort_index().to_dict()
            field_counts[column] = {str(k): int(v) for k, v in vc.items()}
        zero_support_labels = {
            "behavior": sorted(label for label in BEHAVIOR_TO_ID if int(field_counts["behavior"].get(label, 0)) <= 0),
            "substrate": sorted(label for label in SUBSTRATE_TO_ID if int(field_counts["substrate"].get(label, 0)) <= 0),
            "stance": sorted(label for label in STANCE_TO_ID if int(field_counts["stance"].get(label, 0)) <= 0),
        }
        mask_stats = {
            "total_birds": int(len(birds_df)),
            "isbird_yes_rows": int((birds_df["isbird"] == "yes").sum()),
            "isbird_no_rows": int((birds_df["isbird"] == "no").sum()),
            "behavior_null_rows": int(birds_df["behavior"].isna().sum()),
            "substrate_null_rows": int(birds_df["substrate"].isna().sum()),
            "stance_null_rows": int(birds_df["stance"].isna().sum()),
            "stance_relevant_rows": int(stance_relevant.sum()),
            "images_usable_true": int(images_df["image_usable"].astype(bool).sum()),
        }
    else:
        field_counts = {}
        zero_support_labels = {"behavior": sorted(BEHAVIOR_TO_ID), "substrate": sorted(SUBSTRATE_TO_ID), "stance": sorted(STANCE_TO_ID)}
        mask_stats = {
            "total_birds": 0,
            "isbird_yes_rows": 0,
            "isbird_no_rows": 0,
            "behavior_null_rows": 0,
            "substrate_null_rows": 0,
            "stance_null_rows": 0,
            "stance_relevant_rows": 0,
            "images_usable_true": 0,
        }

    report_payload = {
        "annotation_version": args.annotation_version,
        "export_json": str(export_json),
        "images_rows": int(len(images_df)),
        "birds_rows": int(len(birds_df)),
        "migration_stats": migration_stats,
        "mask_stats": mask_stats,
        "field_counts": field_counts,
        "zero_support_labels": zero_support_labels,
    }
    write_report(report_out, report_payload)

    print(f"tasks={len(payload)}")
    print(f"images_rows={len(images_df)}")
    print(f"birds_rows={len(birds_df)}")
    print(f"mask_stats={json.dumps(mask_stats, sort_keys=True)}")
    print(f"images_out={images_out}")
    print(f"birds_out={birds_out}")
    print(f"report_out={report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
