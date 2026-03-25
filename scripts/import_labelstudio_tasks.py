#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.export_labelstudio_snapshot import request_json, resolve_api_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a JSON task bundle into a Label Studio project.")
    parser.add_argument("--base-url", required=True, help="Label Studio base URL")
    parser.add_argument("--api-token", required=True, help="Label Studio API token or refresh token")
    parser.add_argument("--project-id", required=True, type=int, help="Target Label Studio project ID")
    parser.add_argument("--tasks-json", required=True, help="Path to Label Studio tasks JSON")
    parser.add_argument("--report-out", default="", help="Optional JSON report output path")
    return parser.parse_args()


def write_report(path: str, payload: dict[str, object]) -> None:
    if not path:
        return
    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    resolved_token = resolve_api_token(args.base_url, args.api_token)
    tasks_path = Path(args.tasks_json).expanduser().resolve()
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise RuntimeError(f"Expected a JSON list of tasks in {tasks_path}")

    response = request_json(
        args.base_url,
        f"/api/projects/{args.project_id}/import",
        resolved_token,
        method="POST",
        payload=tasks,
    )

    report = {
        "project_id": args.project_id,
        "tasks_json": str(tasks_path),
        "tasks_submitted": len(tasks),
        "response": response,
    }
    write_report(args.report_out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
