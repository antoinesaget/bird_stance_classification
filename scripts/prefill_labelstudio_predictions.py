#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from scripts.export_labelstudio_snapshot import request_json, resolve_api_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-generate and store Label Studio predictions for a project.")
    parser.add_argument("--base-url", required=True, help="Label Studio base URL")
    parser.add_argument("--api-token", required=True, help="Label Studio API token or refresh token")
    parser.add_argument("--project-id", required=True, type=int, help="Target Label Studio project ID")
    parser.add_argument("--ml-backend-url", required=True, help="ML backend base URL, e.g. http://192.168.0.42:9090")
    parser.add_argument("--task-page-size", type=int, default=100, help="Task page size when scanning the project")
    parser.add_argument("--predict-batch-size", type=int, default=16, help="Tasks per /predict call")
    parser.add_argument("--import-batch-size", type=int, default=100, help="Predictions per Label Studio import call")
    parser.add_argument("--only-missing", action="store_true", help="Only process tasks with total_predictions == 0")
    parser.add_argument(
        "--store-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist explicit empty predictions for tasks where the backend returns no regions",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on processed tasks for dry runs")
    parser.add_argument("--report-out", default="", help="Optional JSON report output path")
    return parser.parse_args()


def ml_predict(ml_backend_url: str, tasks: list[dict]) -> list[dict]:
    request = urllib.request.Request(
        urllib.parse.urljoin(ml_backend_url.rstrip("/") + "/", "predict"),
        data=json.dumps({"tasks": tasks}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        predictions = payload.get("results")
    if not isinstance(predictions, list):
        return []
    return predictions


def ls_request_json(
    base_url: str,
    api_token: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict | list | None = None,
) -> dict | list:
    resolved_token = resolve_api_token(base_url, api_token)
    return request_json(base_url, path, resolved_token, method=method, payload=payload)


def sanitize_prediction(prediction: dict) -> dict:
    return {
        "task": prediction.get("task"),
        "model_version": prediction.get("model_version"),
        "score": float(prediction.get("score") or 0.0),
        "result": prediction.get("result") or [],
    }


def import_predictions(base_url: str, project_id: int, api_token: str, predictions: list[dict]) -> dict | list:
    return ls_request_json(
        base_url,
        api_token,
        f"/api/projects/{project_id}/import/predictions",
        method="POST",
        payload=predictions,
    )


def task_pages(base_url: str, project_id: int, api_token: str, page_size: int) -> list[dict]:
    page = 1
    while True:
        payload = ls_request_json(
            base_url,
            api_token,
            f"/api/tasks?project={project_id}&page={page}&page_size={page_size}",
        )
        tasks = payload.get("tasks") if isinstance(payload, dict) else None
        if not tasks:
            break
        for task in tasks:
            yield task
        total = int(payload.get("total") or 0) if isinstance(payload, dict) else 0
        if page * page_size >= total:
            break
        page += 1


def write_report(path: str, report: dict[str, object]) -> None:
    if not path:
        return
    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def main() -> int:
    args = parse_args()

    scanned_tasks = 0
    eligible_tasks = 0
    predicted_tasks = 0
    imported_predictions = 0
    skipped_existing = 0
    failed_predictions = 0
    empty_predictions = 0
    stored_empty_predictions = 0
    imported_batches = 0
    failures: list[dict[str, object]] = []

    pending_tasks: list[dict] = []

    def flush_pending() -> None:
        nonlocal predicted_tasks, imported_predictions, failed_predictions, empty_predictions, stored_empty_predictions, imported_batches
        if not pending_tasks:
            return
        predictions = ml_predict(args.ml_backend_url, pending_tasks.copy())
        task_by_id = {task.get("id"): task for task in pending_tasks}
        pending_tasks.clear()

        sanitized: list[dict] = []
        for prediction in predictions:
            task_id = prediction.get("task")
            result = prediction.get("result") or []
            if prediction.get("error"):
                failed_predictions += 1
                failures.append({"task": task_id, "error": prediction.get("error")})
                continue
            if task_id not in task_by_id:
                failed_predictions += 1
                failures.append({"task": task_id, "error": "unexpected_task_id"})
                continue
            if not result:
                empty_predictions += 1
                if not args.store_empty:
                    failures.append({"task": task_id, "error": "empty_result"})
                    continue
                stored_empty_predictions += 1
                sanitized.append(sanitize_prediction(prediction))
                continue
            sanitized.append(sanitize_prediction(prediction))

        predicted_tasks += len(sanitized)
        for batch in chunked(sanitized, args.import_batch_size):
            if not batch:
                continue
            response = import_predictions(args.base_url, args.project_id, args.api_token, batch)
            if isinstance(response, dict):
                imported_predictions += int(response.get("created") or len(batch))
            else:
                imported_predictions += len(batch)
            imported_batches += 1

    for task in task_pages(args.base_url, args.project_id, args.api_token, args.task_page_size):
        scanned_tasks += 1
        if args.only_missing and int(task.get("total_predictions") or 0) > 0:
            skipped_existing += 1
            continue

        eligible_tasks += 1
        pending_tasks.append(task)
        if args.limit and eligible_tasks >= args.limit:
            break
        if len(pending_tasks) >= args.predict_batch_size:
            flush_pending()

    flush_pending()

    report = {
        "project_id": args.project_id,
        "ml_backend_url": args.ml_backend_url,
        "task_page_size": args.task_page_size,
        "predict_batch_size": args.predict_batch_size,
        "import_batch_size": args.import_batch_size,
        "only_missing": args.only_missing,
        "limit": args.limit,
        "scanned_tasks": scanned_tasks,
        "eligible_tasks": eligible_tasks,
        "skipped_existing": skipped_existing,
        "predicted_tasks": predicted_tasks,
        "imported_predictions": imported_predictions,
        "imported_batches": imported_batches,
        "failed_predictions": failed_predictions,
        "empty_predictions": empty_predictions,
        "stored_empty_predictions": stored_empty_predictions,
        "failures": failures[:100],
    }
    write_report(args.report_out, report)
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
