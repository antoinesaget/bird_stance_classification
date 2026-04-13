#!/usr/bin/env python3
"""Purpose: Backfill missing species/path metadata onto existing Label Studio tasks from import manifests."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import tempfile
import urllib.parse
from typing import Any

from birdsys.core import default_data_home, default_species_slug, ensure_layout, normalize_relative_path
from birdsys.labelstudio.export_snapshot import download_bytes, request_json, resolve_api_token


REQUIRED_META_FIELDS = ("species_slug", "compression_profile", "original_relative_path", "served_relative_path")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill task meta from stored Label Studio import manifests")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-token", required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--data-home", default=str(default_data_home()))
    parser.add_argument("--species-slug", default=default_species_slug())
    parser.add_argument("--apply", action="store_true", help="Actually PATCH matching tasks instead of only writing the report")
    return parser.parse_args(argv)


def parse_localfiles_relpath(image_value: str) -> str | None:
    if "?d=" not in image_value:
        return None
    parsed = urllib.parse.urlparse(image_value)
    query = urllib.parse.parse_qs(parsed.query)
    local = urllib.parse.unquote(query.get("d", [""])[0]).strip()
    if not local:
        return None
    return normalize_relative_path(local, field_name="task data.image local-files relpath")


def load_manifest_rows(import_root: pathlib.Path, *, species_slug: str) -> dict[str, dict[str, Any]]:
    by_image_id: dict[str, dict[str, Any]] = {}
    by_localfiles_relpath: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(import_root.glob("*.manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            image_id = str(row.get("image_id") or "").strip()
            if not image_id:
                continue
            original_relpath = normalize_relative_path(str(row.get("original_relative_path") or ""), field_name="original_relative_path")
            served_relpath = normalize_relative_path(str(row.get("served_relative_path") or ""), field_name="served_relative_path")
            compression_profile = str(row.get("compression_profile") or "").strip()
            if not compression_profile:
                parts = pathlib.Path(served_relpath).parts
                if len(parts) >= 3 and parts[0] == "labelstudio" and parts[1] == "images_compressed":
                    compression_profile = parts[2]
            if not compression_profile:
                raise ValueError(f"Manifest row for image_id={image_id!r} is missing compression_profile")
            localfiles_relpath = str(row.get("served_localfiles_relative_path") or "").strip()
            if not localfiles_relpath:
                localfiles_relpath = f"birds/{species_slug}/{served_relpath}"
            localfiles_relpath = normalize_relative_path(localfiles_relpath, field_name="served_localfiles_relative_path")
            record = {
                "image_id": image_id,
                "source_filename": str(row.get("source_filename") or pathlib.Path(original_relpath).name),
                "species_slug": species_slug,
                "compression_profile": compression_profile,
                "original_relative_path": original_relpath,
                "served_relative_path": served_relpath,
                "served_localfiles_relative_path": localfiles_relpath,
                "manifest_path": str(manifest_path),
            }
            by_image_id.setdefault(image_id, record)
            by_localfiles_relpath.setdefault(localfiles_relpath, record)
    return {"by_image_id": by_image_id, "by_localfiles_relpath": by_localfiles_relpath}


def task_has_required_meta(task: dict[str, Any]) -> bool:
    meta = task.get("meta") or {}
    return all(str(meta.get(field) or "").strip() for field in REQUIRED_META_FIELDS)


def match_manifest_row(task: dict[str, Any], *, manifest_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    data = task.get("data") or {}
    image_value = str(data.get("image") or "")
    localfiles_relpath = parse_localfiles_relpath(image_value) if image_value else None
    if localfiles_relpath:
        row = manifest_index["by_localfiles_relpath"].get(localfiles_relpath)
        if row is not None:
            return row
    image_id = str(data.get("image_id") or "").strip()
    if image_id:
        return manifest_index["by_image_id"].get(image_id)
    return None


def build_updated_task_payload(task: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    current_meta = dict(task.get("meta") or {})
    updated_meta = {
        **current_meta,
        "species_slug": row["species_slug"],
        "compression_profile": row["compression_profile"],
        "original_relative_path": row["original_relative_path"],
        "served_relative_path": row["served_relative_path"],
    }
    return {"meta": updated_meta}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layout = ensure_layout(pathlib.Path(args.data_home), args.species_slug)
    manifest_index = load_manifest_rows(layout.labelstudio_imports, species_slug=layout.species_slug)
    resolved_token = resolve_api_token(args.base_url, args.api_token)

    with tempfile.TemporaryDirectory(prefix="birdsys_backfill_") as tmpdir:
        export_path = pathlib.Path(tmpdir) / "project_export.json"
        query = urllib.parse.urlencode({"exportType": "JSON", "download_all_tasks": "true"})
        export_path.write_bytes(download_bytes(args.base_url, f"/api/projects/{args.project_id}/export?{query}", resolved_token))
        tasks = json.loads(export_path.read_text(encoding="utf-8"))

    if not isinstance(tasks, list):
        raise TypeError("Expected project export payload to be a list of tasks")

    matched = 0
    missing = 0
    patched = 0
    already_complete = 0
    planned_updates: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task.get("id")
        if task_has_required_meta(task):
            already_complete += 1
            continue
        row = match_manifest_row(task, manifest_index=manifest_index)
        if row is None:
            missing += 1
            planned_updates.append({"task_id": task_id, "status": "missing_manifest_match"})
            continue
        matched += 1
        payload = build_updated_task_payload(task, row)
        planned_updates.append({"task_id": task_id, "status": "matched", "payload": payload})
        if args.apply:
            request_json(args.base_url, f"/api/tasks/{task_id}", resolved_token, method="PATCH", payload=payload)
            patched += 1

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "species_slug": layout.species_slug,
        "project_id": args.project_id,
        "apply": bool(args.apply),
        "counts": {
            "tasks_total": len(tasks),
            "already_complete": already_complete,
            "matched": matched,
            "missing_manifest_match": missing,
            "patched": patched,
        },
        "updates": planned_updates,
    }
    report_path = layout.metadata / "task_metadata_backfill_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"species_slug={layout.species_slug}")
    print(f"report={report_path}")
    print(f"tasks_total={len(tasks)}")
    print(f"already_complete={already_complete}")
    print(f"matched={matched}")
    print(f"missing_manifest_match={missing}")
    print(f"patched={patched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
