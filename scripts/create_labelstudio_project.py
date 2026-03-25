#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from scripts.export_labelstudio_snapshot import request_json, resolve_api_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update a Label Studio project from repo XML and clone source tasks/ML backend settings.")
    parser.add_argument("--base-url", required=True, help="Label Studio base URL")
    parser.add_argument("--api-token", required=True, help="Label Studio API token or refresh token")
    parser.add_argument("--label-config", required=True, help="Path to label_config.xml")
    parser.add_argument("--source-project-id", required=True, type=int, help="Existing project to clone task data and defaults from")
    parser.add_argument("--title", required=True, help="Target project title")
    parser.add_argument("--ml-url", default="", help="Optional ML backend URL override")
    parser.add_argument("--ml-title", default="birdsys-schema-v2", help="Title for the project ML backend connection")
    parser.add_argument("--report-out", default="", help="Optional JSON report output path")
    return parser.parse_args()


def list_paginated(base_url: str, path: str, api_token: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    next_path = path
    while next_path:
        payload = request_json(base_url, next_path, api_token)
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            items.extend([item for item in payload["results"] if isinstance(item, dict)])
            next_value = payload.get("next")
            if next_value:
                if next_value.startswith(base_url):
                    next_path = next_value[len(base_url.rstrip("/")) :]
                else:
                    next_path = next_value
            else:
                next_path = ""
        elif isinstance(payload, list):
            items.extend([item for item in payload if isinstance(item, dict)])
            next_path = ""
        else:
            next_path = ""
    return items


def task_import_payload(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task in tasks:
        item: dict[str, Any] = {"data": dict(task.get("data") or {})}
        meta = task.get("meta") or {}
        if isinstance(meta, dict) and meta:
            item["meta"] = dict(meta)
        output.append(item)
    return output


def clone_project_defaults(source_project: dict[str, Any], *, title: str, label_config: str) -> dict[str, Any]:
    payload = {
        "title": title,
        "label_config": label_config,
        "description": source_project.get("description") or "",
        "expert_instruction": source_project.get("expert_instruction") or "",
        "show_instruction": bool(source_project.get("show_instruction", False)),
        "show_skip_button": bool(source_project.get("show_skip_button", True)),
        "enable_empty_annotation": bool(source_project.get("enable_empty_annotation", True)),
        "show_annotation_history": bool(source_project.get("show_annotation_history", False)),
        "color": source_project.get("color") or "#FFFFFF",
        "maximum_annotations": int(source_project.get("maximum_annotations") or 1),
        "show_collab_predictions": bool(source_project.get("show_collab_predictions", True)),
    }
    return payload


def write_report(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    resolved_token = resolve_api_token(args.base_url, args.api_token)
    label_config = Path(args.label_config).expanduser().resolve().read_text(encoding="utf-8")

    source_project = request_json(args.base_url, f"/api/projects/{args.source_project_id}", resolved_token)
    if not isinstance(source_project, dict):
        raise RuntimeError("Source project response was not an object")

    existing_projects = list_paginated(args.base_url, "/api/projects?page_size=100", resolved_token)
    target_project = next((proj for proj in existing_projects if proj.get("title") == args.title), None)
    project_payload = clone_project_defaults(source_project, title=args.title, label_config=label_config)

    created = False
    if target_project is None:
        target_project = request_json(args.base_url, "/api/projects", resolved_token, method="POST", payload=project_payload)
        created = True
    else:
        target_project = request_json(
            args.base_url,
            f"/api/projects/{target_project['id']}",
            resolved_token,
            method="PUT",
            payload={**target_project, **project_payload},
        )

    target_project_id = int(target_project["id"])
    target_tasks = request_json(args.base_url, f"/api/tasks?project={target_project_id}&page_size=1", resolved_token)
    target_task_total = int((target_tasks or {}).get("total") or 0) if isinstance(target_tasks, dict) else 0
    imported_tasks = 0
    if target_task_total == 0:
        source_tasks = list_paginated(args.base_url, f"/api/tasks?project={args.source_project_id}&page_size=100", resolved_token)
        payload = task_import_payload(source_tasks)
        request_json(
            args.base_url,
            f"/api/projects/{target_project_id}/import",
            resolved_token,
            method="POST",
            payload=payload,
        )
        imported_tasks = len(payload)

    ml_backends = list_paginated(args.base_url, f"/api/ml?project={args.source_project_id}&page_size=100", resolved_token)
    source_ml = ml_backends[0] if ml_backends else {}
    ml_url = args.ml_url or str(source_ml.get("url") or "")
    ml_payload = {
        "url": ml_url,
        "title": args.ml_title,
        "project": target_project_id,
        "auth_method": str(source_ml.get("auth_method") or "NONE"),
        "description": str(source_ml.get("description") or ""),
        "extra_params": source_ml.get("extra_params") or "",
        "timeout": float(source_ml.get("timeout") or 100.0),
        "auto_update": bool(source_ml.get("auto_update", True)),
        "is_interactive": bool(source_ml.get("is_interactive", False)),
    }
    target_ml_backends = list_paginated(args.base_url, f"/api/ml?project={target_project_id}&page_size=100", resolved_token)
    target_ml = next((item for item in target_ml_backends if item.get("title") == args.ml_title), None)
    if target_ml is None and target_ml_backends:
        target_ml = target_ml_backends[0]

    if target_ml is None:
        target_ml = request_json(args.base_url, "/api/ml", resolved_token, method="POST", payload=ml_payload)
    else:
        target_ml = request_json(
            args.base_url,
            f"/api/ml/{target_ml['id']}",
            resolved_token,
            method="PUT",
            payload={**target_ml, **ml_payload},
        )

    report = {
        "created": created,
        "source_project_id": args.source_project_id,
        "target_project_id": target_project_id,
        "target_project_title": args.title,
        "imported_tasks": imported_tasks,
        "ml_backend_id": target_ml.get("id"),
        "ml_backend_url": target_ml.get("url"),
    }
    write_report(args.report_out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
