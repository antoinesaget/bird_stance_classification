#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from birdsys.labelstudio.export_snapshot import auth_header, request_json, resolve_api_token


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        "--untouched-only",
        action="store_true",
        help="Only process tasks with no submitted annotations and no drafts",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace existing prediction rows after new predictions are imported successfully",
    )
    parser.add_argument(
        "--store-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist explicit empty predictions for tasks where the backend returns no regions",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on processed tasks for dry runs")
    parser.add_argument("--report-out", default="", help="Optional JSON report output path")
    return parser.parse_args(argv)


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


def ls_request(
    base_url: str,
    api_token: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict | list | None = None,
) -> tuple[int, dict | list | None]:
    resolved_token = resolve_api_token(base_url, api_token)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if resolved_token:
        headers["Authorization"] = auth_header(resolved_token)
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        data=body,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        if not raw:
            return response.status, None
        return response.status, json.loads(raw)


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


def task_is_untouched(task: dict[str, Any]) -> bool:
    if int(task.get("total_annotations") or 0) > 0:
        return False
    if bool(task.get("is_labeled")):
        return False
    return not task_has_drafts(task)


def task_has_drafts(task: dict[str, Any]) -> bool:
    drafts = task.get("drafts") or []
    return len(drafts) > 0


def get_task_detail(base_url: str, api_token: str, task_id: int) -> dict[str, Any]:
    payload = ls_request_json(base_url, api_token, f"/api/tasks/{task_id}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected task detail payload for task {task_id}")
    return payload


def prediction_ids_from_task_detail(task_detail: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for prediction in task_detail.get("predictions") or []:
        prediction_id = prediction.get("id")
        if prediction_id is None:
            continue
        out.append(int(prediction_id))
    return out


def delete_prediction(base_url: str, api_token: str, prediction_id: int) -> None:
    status, _ = ls_request(base_url, api_token, f"/api/predictions/{prediction_id}", method="DELETE")
    if status not in {200, 204}:
        raise RuntimeError(f"Unexpected delete status for prediction {prediction_id}: {status}")


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


def refresh_task_prediction_ids(
    *,
    base_url: str,
    api_token: str,
    task_id: int,
    old_prediction_ids: list[int],
) -> list[int]:
    detail = get_task_detail(base_url, api_token, task_id)
    old_ids = set(old_prediction_ids)
    return [prediction_id for prediction_id in prediction_ids_from_task_detail(detail) if prediction_id not in old_ids]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    scanned_tasks = 0
    eligible_tasks = 0
    predicted_tasks = 0
    imported_predictions = 0
    skipped_existing = 0
    skipped_annotated = 0
    skipped_drafts = 0
    backend_failures = 0
    import_failures = 0
    delete_failures = 0
    detail_failures = 0
    empty_predictions = 0
    stored_empty_predictions = 0
    imported_batches = 0
    tasks_successfully_replaced = 0
    old_prediction_rows_deleted = 0
    tasks_left_on_old_predictions = 0
    failures: list[dict[str, object]] = []

    pending_tasks: list[dict] = []

    def flush_pending() -> None:
        nonlocal predicted_tasks, imported_predictions, backend_failures, import_failures
        nonlocal delete_failures, detail_failures, empty_predictions, stored_empty_predictions, imported_batches
        nonlocal tasks_successfully_replaced, old_prediction_rows_deleted, tasks_left_on_old_predictions
        if not pending_tasks:
            return
        old_prediction_ids_by_task_id: dict[int, list[int]] = {}
        batch_tasks = pending_tasks.copy()
        pending_tasks.clear()

        for task in batch_tasks:
            task_id = int(task.get("id"))
            if not args.replace_existing:
                continue
            try:
                detail = get_task_detail(args.base_url, args.api_token, task_id)
            except Exception as exc:  # noqa: BLE001
                detail_failures += 1
                tasks_left_on_old_predictions += 1
                failures.append({"task": task_id, "error": f"detail_fetch_failed:{exc}"})
                continue
            old_prediction_ids_by_task_id[task_id] = prediction_ids_from_task_detail(detail)

        prediction_inputs = [
            task for task in batch_tasks if not args.replace_existing or int(task.get("id")) in old_prediction_ids_by_task_id
        ]
        if not prediction_inputs:
            return

        predictions = ml_predict(args.ml_backend_url, prediction_inputs)
        predictions_by_task_id: dict[int, dict[str, Any]] = {}
        for prediction in predictions:
            task_id_raw = prediction.get("task")
            if task_id_raw is None:
                backend_failures += 1
                failures.append({"task": None, "error": "missing_task_id"})
                continue
            task_id = int(task_id_raw)
            if task_id in predictions_by_task_id:
                backend_failures += 1
                tasks_left_on_old_predictions += int(args.replace_existing)
                failures.append({"task": task_id, "error": "duplicate_prediction"})
                continue
            predictions_by_task_id[task_id] = prediction

        sanitized: list[dict] = []
        sanitized_task_ids: list[int] = []
        for task in prediction_inputs:
            task_id = int(task.get("id"))
            prediction = predictions_by_task_id.get(task_id)
            if prediction is None:
                backend_failures += 1
                tasks_left_on_old_predictions += int(args.replace_existing)
                failures.append({"task": task_id, "error": "missing_prediction"})
                continue
            result = prediction.get("result") or []
            if prediction.get("error"):
                backend_failures += 1
                tasks_left_on_old_predictions += int(args.replace_existing)
                failures.append({"task": task_id, "error": prediction.get("error")})
                continue
            if not result:
                empty_predictions += 1
                if not args.store_empty:
                    backend_failures += 1
                    tasks_left_on_old_predictions += int(args.replace_existing)
                    failures.append({"task": task_id, "error": "empty_result"})
                    continue
                stored_empty_predictions += 1
                sanitized.append(sanitize_prediction(prediction))
                sanitized_task_ids.append(task_id)
                continue
            sanitized.append(sanitize_prediction(prediction))
            sanitized_task_ids.append(task_id)

        predicted_tasks += len(sanitized)
        for batch, batch_task_ids in zip(chunked(sanitized, args.import_batch_size), chunked(sanitized_task_ids, args.import_batch_size)):
            if not batch:
                continue
            try:
                response = import_predictions(args.base_url, args.project_id, args.api_token, batch)
            except Exception as exc:  # noqa: BLE001
                import_failures += len(batch_task_ids)
                tasks_left_on_old_predictions += len(batch_task_ids) if args.replace_existing else 0
                for task_id in batch_task_ids:
                    failures.append({"task": task_id, "error": f"import_failed:{exc}"})
                continue
            created_count = int(response.get("created") or len(batch)) if isinstance(response, dict) else len(batch)
            imported_predictions += created_count
            imported_batches += 1
            if not args.replace_existing:
                continue
            for task_id in batch_task_ids:
                old_prediction_ids = old_prediction_ids_by_task_id.get(task_id, [])
                try:
                    new_prediction_ids = refresh_task_prediction_ids(
                        base_url=args.base_url,
                        api_token=args.api_token,
                        task_id=task_id,
                        old_prediction_ids=old_prediction_ids,
                    )
                except Exception as exc:  # noqa: BLE001
                    detail_failures += 1
                    tasks_left_on_old_predictions += 1
                    failures.append({"task": task_id, "error": f"post_import_detail_failed:{exc}"})
                    continue
                if not new_prediction_ids:
                    import_failures += 1
                    tasks_left_on_old_predictions += 1
                    failures.append({"task": task_id, "error": "import_not_observed"})
                    continue
                delete_failed = False
                for prediction_id in old_prediction_ids:
                    try:
                        delete_prediction(args.base_url, args.api_token, prediction_id)
                    except Exception as exc:  # noqa: BLE001
                        delete_failures += 1
                        delete_failed = True
                        failures.append({"task": task_id, "prediction_id": prediction_id, "error": f"delete_failed:{exc}"})
                    else:
                        old_prediction_rows_deleted += 1
                if delete_failed:
                    tasks_left_on_old_predictions += 1
                    continue
                tasks_successfully_replaced += 1

    for task in task_pages(args.base_url, args.project_id, args.api_token, args.task_page_size):
        scanned_tasks += 1
        if args.untouched_only:
            if task_has_drafts(task):
                skipped_drafts += 1
                continue
            if not task_is_untouched(task):
                skipped_annotated += 1
                continue
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
        "untouched_only": args.untouched_only,
        "replace_existing": args.replace_existing,
        "limit": args.limit,
        "scanned_tasks": scanned_tasks,
        "eligible_tasks": eligible_tasks,
        "skipped_annotated": skipped_annotated,
        "skipped_drafts": skipped_drafts,
        "skipped_existing": skipped_existing,
        "predicted_tasks": predicted_tasks,
        "imported_predictions": imported_predictions,
        "imported_batches": imported_batches,
        "backend_failures": backend_failures,
        "import_failures": import_failures,
        "delete_failures": delete_failures,
        "detail_failures": detail_failures,
        "empty_predictions": empty_predictions,
        "stored_empty_predictions": stored_empty_predictions,
        "tasks_successfully_replaced": tasks_successfully_replaced,
        "old_prediction_rows_deleted": old_prediction_rows_deleted,
        "tasks_left_on_old_predictions": tasks_left_on_old_predictions,
        "failures": failures[:100],
    }
    write_report(args.report_out, report)
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
