#!/usr/bin/env python3
"""Purpose: Export and normalize one strict, versioned Label Studio annotation extract."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
from collections.abc import Sequence

from birdsys.core import default_data_home, default_species_slug, ensure_layout, next_version_dir
from birdsys.datasets.export_normalize import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_LABEL_CONFIG,
    NORMALIZATION_POLICY_VERSION,
    normalize_export,
)
from birdsys.labelstudio.export_snapshot import export_project_snapshot, request_json, resolve_api_token


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one versioned Label Studio annotation extract")
    parser.add_argument("--data-home", default=str(default_data_home()))
    parser.add_argument("--species-slug", default=default_species_slug())
    parser.add_argument("--annotation-version", default="", help="Optional explicit ann_vNNN override")
    parser.add_argument("--label-config", default=str(DEFAULT_LABEL_CONFIG))
    parser.add_argument("--base-url", default=os.getenv("LABEL_STUDIO_URL", ""))
    parser.add_argument("--api-token", default=os.getenv("LABEL_STUDIO_API_TOKEN", ""))
    parser.add_argument("--project-id", type=int, default=0)
    parser.add_argument("--title", default="")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--download-all-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the direct export endpoint so the raw export contains the whole project snapshot.",
    )
    parser.add_argument(
        "--export-json",
        default="",
        help="Optional local raw export JSON to version and normalize instead of calling the Label Studio API.",
    )
    return parser.parse_args(argv)


def load_dotenv(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_version_dir(parent: pathlib.Path, requested_version: str) -> pathlib.Path:
    if requested_version:
        path = parent / requested_version
        path.mkdir(parents=True, exist_ok=False)
        return path
    return next_version_dir(parent, "ann")


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(pathlib.Path(".env").resolve())
    args = parse_args(argv)

    data_home = pathlib.Path(args.data_home).expanduser().resolve()
    label_config = pathlib.Path(args.label_config).expanduser().resolve()
    if not label_config.exists():
        raise FileNotFoundError(label_config)

    layout = ensure_layout(data_home, args.species_slug)
    export_dir = resolve_version_dir(layout.labelstudio_exports, args.annotation_version)
    annotation_version = export_dir.name
    raw_export_path = export_dir / "project_export.json"
    raw_metadata_path = export_dir / "export_metadata.json"

    metadata: dict
    if args.export_json:
        source_export = pathlib.Path(args.export_json).expanduser().resolve()
        if not source_export.exists():
            raise FileNotFoundError(source_export)
        shutil.copy2(source_export, raw_export_path)
        metadata = {
            "mode": "local_file",
            "source_export_json": str(source_export),
            "copied_to": str(raw_export_path),
            "project_id": args.project_id or None,
            "base_url": args.base_url or None,
        }
    else:
        base_url = args.base_url or os.getenv("LABEL_STUDIO_URL", "")
        api_token = args.api_token or os.getenv("LABEL_STUDIO_API_TOKEN", "")
        if not base_url:
            raise ValueError("--base-url is required when --export-json is not used")
        if not api_token:
            raise ValueError("--api-token is required when --export-json is not used")
        if not args.project_id:
            raise ValueError("--project-id is required when --export-json is not used")

        resolved_token = resolve_api_token(base_url, api_token)
        project = request_json(base_url, f"/api/projects/{args.project_id}", resolved_token)
        metadata = export_project_snapshot(
            base_url=base_url,
            api_token=api_token,
            project_id=args.project_id,
            output=raw_export_path,
            metadata_out=None,
            title=args.title,
            download_all_tasks=bool(args.download_all_tasks),
            timeout_seconds=args.timeout_seconds,
        )
        if isinstance(project, dict):
            metadata["project_title"] = project.get("title")
            metadata["project_id"] = project.get("id", args.project_id)

    metadata.update(
        {
            "annotation_version": annotation_version,
            "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "label_config_path": str(label_config),
        }
    )
    write_json(raw_metadata_path, metadata)

    result = normalize_export(
        export_json=raw_export_path,
        annotation_version=annotation_version,
        data_home=data_home,
        species_slug=args.species_slug,
        label_config=label_config,
        raw_metadata=metadata,
    )

    print(f"annotation_version={annotation_version}")
    print(f"export_dir={export_dir}")
    print(f"raw_export_json={raw_export_path}")
    print(f"raw_metadata_json={raw_metadata_path}")
    print(f"normalized_dir={result.out_dir}")
    print(f"manifest_out={result.manifest_out}")
    print(f"extract_report_md={result.extract_report_md}")
    print(f"comparison_md={result.comparison_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
