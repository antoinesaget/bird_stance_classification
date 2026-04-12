#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path

from common import ROOT, TRUENAS_ENV_PATH, load_env_file, request_empty, request_json, resolve_api_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete predictions only for untouched tasks in project 7")
    parser.add_argument("--project-id", type=int, required=True)
    return parser.parse_args()


def task_pages(base_url: str, api_token: str, project_id: int, page_size: int = 100):
    page = 1
    while True:
        payload = request_json(base_url, f"/api/tasks?project={project_id}&page={page}&page_size={page_size}", api_token)
        tasks = payload.get("tasks") if isinstance(payload, dict) else None
        if not tasks:
            break
        for task in tasks:
            yield task
        total = int(payload.get("total") or 0)
        if page * page_size >= total:
            break
        page += 1


def main() -> int:
    args = parse_args()
    env = load_env_file(TRUENAS_ENV_PATH)
    base_url = env["LABEL_STUDIO_URL"]
    api_token = resolve_api_token(base_url, env["LABEL_STUDIO_API_TOKEN"])

    scanned_tasks = 0
    skipped_annotated_tasks = 0
    deleted_predictions = 0
    untouched_tasks_with_predictions = 0

    for task in task_pages(base_url, api_token, args.project_id):
        scanned_tasks += 1
        annotations = task.get("annotations") or []
        kept_annotations = [ann for ann in annotations if isinstance(ann, dict) and not ann.get("was_cancelled")]
        if kept_annotations:
            skipped_annotated_tasks += 1
            continue
        predictions = task.get("predictions") or []
        if not predictions:
            predictions = request_json(
                base_url,
                f"/api/predictions/?project={args.project_id}&task={task['id']}",
                api_token,
            )
        if predictions:
            untouched_tasks_with_predictions += 1
        for prediction in predictions:
            prediction_id = prediction.get("id")
            if prediction_id is None:
                continue
            request_empty(base_url, f"/api/predictions/{prediction_id}/", api_token)
            deleted_predictions += 1

    report = {
        "project_id": args.project_id,
        "scanned_tasks": scanned_tasks,
        "skipped_annotated_tasks": skipped_annotated_tasks,
        "untouched_tasks_with_predictions": untouched_tasks_with_predictions,
        "deleted_predictions": deleted_predictions,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
