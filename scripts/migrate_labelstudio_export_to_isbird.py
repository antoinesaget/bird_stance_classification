#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Label Studio export from image_status schema to isbird per-region schema"
    )
    parser.add_argument("--input-json", required=True, help="Source Label Studio export JSON")
    parser.add_argument("--output-json", required=True, help="Migrated JSON output path")
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional migration report output path (default: <output-json>.report.json)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs")
    return parser.parse_args()


def _normalize_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _is_bird_rectangle(item: dict[str, Any]) -> bool:
    if _normalize_name(item.get("type")) != "rectanglelabels":
        return False
    labels = ((_normalize_name(v) for v in ((item.get("value") or {}).get("rectanglelabels") or [])))
    return "bird" in set(labels)


def _build_isbird_choice_from_rectangle(rect_item: dict[str, Any]) -> dict[str, Any]:
    rect_id = rect_item.get("id")
    value = rect_item.get("value") or {}
    choice_value: dict[str, Any] = {
        "choices": ["yes"],
    }
    for key in ("x", "y", "width", "height", "rotation"):
        if key in value:
            choice_value[key] = value[key]

    out: dict[str, Any] = {
        "id": rect_id,
        "from_name": "isbird",
        "to_name": rect_item.get("to_name", "image"),
        "type": "choices",
        "parentID": rect_id,
        "value": choice_value,
    }

    # Keep metadata fields when present to preserve editability in re-imported projects.
    for key in (
        "origin",
        "score",
        "image_rotation",
        "original_width",
        "original_height",
    ):
        if key in rect_item:
            out[key] = rect_item[key]
    return out


def migrate_result_items(
    result_items: list[dict[str, Any]],
    *,
    stats: dict[str, int],
    anomalies: list[str],
    owner_label: str,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []

    # Region id -> bool (already has explicit isbird)
    isbird_present: dict[str, bool] = {}
    bird_regions: list[tuple[str, dict[str, Any]]] = []

    for item in result_items:
        from_name = _normalize_name(item.get("from_name"))
        item_id = item.get("id")

        if from_name == "image_status":
            stats["result_items_removed_image_status"] += 1
            continue

        if from_name == "isbird":
            if item_id is not None:
                isbird_present[str(item_id)] = True

        if _is_bird_rectangle(item):
            if item_id is None:
                anomalies.append(f"{owner_label}: bird rectangle missing id")
            else:
                bird_regions.append((str(item_id), item))
                stats["bird_regions_seen"] += 1

        filtered.append(item)

    injected: list[dict[str, Any]] = []
    for region_id, rect_item in bird_regions:
        if isbird_present.get(region_id):
            stats["regions_already_with_isbird"] += 1
            continue
        injected.append(_build_isbird_choice_from_rectangle(rect_item))
        stats["isbird_injected"] += 1

    return filtered + injected


def migrate_payload(payload: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, int] = {
        "tasks_total": 0,
        "annotations_processed": 0,
        "predictions_processed": 0,
        "drafts_processed": 0,
        "bird_regions_seen": 0,
        "regions_already_with_isbird": 0,
        "isbird_injected": 0,
        "result_items_removed_image_status": 0,
    }
    anomalies: list[str] = []

    migrated: list[dict[str, Any]] = []
    for task in payload:
        stats["tasks_total"] += 1
        task_id = task.get("id", "unknown")
        task_copy = json.loads(json.dumps(task))

        annotations = task_copy.get("annotations") or []
        for idx, ann in enumerate(annotations):
            results = ann.get("result") or []
            if not isinstance(results, list):
                anomalies.append(f"task={task_id} annotation={idx}: result is not a list")
                continue
            ann["result"] = migrate_result_items(
                results,
                stats=stats,
                anomalies=anomalies,
                owner_label=f"task={task_id} annotation={idx}",
            )
            stats["annotations_processed"] += 1

        predictions = task_copy.get("predictions") or []
        for idx, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                continue
            results = pred.get("result") or []
            if not isinstance(results, list):
                anomalies.append(f"task={task_id} prediction={idx}: result is not a list")
                continue
            pred["result"] = migrate_result_items(
                results,
                stats=stats,
                anomalies=anomalies,
                owner_label=f"task={task_id} prediction={idx}",
            )
            stats["predictions_processed"] += 1

        drafts = task_copy.get("drafts") or []
        for idx, draft in enumerate(drafts):
            if not isinstance(draft, dict):
                continue
            results = draft.get("result") or []
            if not isinstance(results, list):
                anomalies.append(f"task={task_id} draft={idx}: result is not a list")
                continue
            draft["result"] = migrate_result_items(
                results,
                stats=stats,
                anomalies=anomalies,
                owner_label=f"task={task_id} draft={idx}",
            )
            stats["drafts_processed"] += 1

        migrated.append(task_copy)

    report = {
        "stats": stats,
        "anomalies": anomalies,
        "anomalies_count": len(anomalies),
    }
    return migrated, report


def main() -> int:
    args = parse_args()

    src = pathlib.Path(args.input_json).expanduser().resolve()
    dst = pathlib.Path(args.output_json).expanduser().resolve()
    report_path = (
        pathlib.Path(args.report_json).expanduser().resolve()
        if args.report_json
        else dst.with_suffix(dst.suffix + ".report.json")
    )

    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"{dst} already exists (use --overwrite)")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"{report_path} already exists (use --overwrite)")

    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Label Studio export payload must be a list")

    migrated, report = migrate_payload(payload)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(migrated, ensure_ascii=True) + "\n", encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"input={src}")
    print(f"output={dst}")
    print(f"report={report_path}")
    print(f"stats={json.dumps(report['stats'], sort_keys=True)}")
    print(f"anomalies_count={report['anomalies_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
