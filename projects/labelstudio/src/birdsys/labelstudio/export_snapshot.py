#!/usr/bin/env python3
"""Purpose: Export project snapshots and related metadata from a Label Studio instance"""
from __future__ import annotations

import argparse
import base64
import json
import time
from collections.abc import Sequence
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and download a Label Studio export snapshot.")
    parser.add_argument("--base-url", required=True, help="Label Studio base URL, e.g. https://birds.ashs.live")
    parser.add_argument("--api-token", required=True, help="Label Studio API token")
    parser.add_argument("--project-id", required=True, type=int, help="Label Studio project ID")
    parser.add_argument("--output", required=True, help="Output JSON export path")
    parser.add_argument("--metadata-out", help="Optional metadata JSON sidecar path")
    parser.add_argument("--title", default="", help="Optional snapshot title")
    parser.add_argument("--download-all-tasks", action="store_true", help="Use the direct export endpoint to include unannotated tasks")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="How long to wait for a background snapshot")
    return parser.parse_args(argv)


def decode_jwt_payload(token: str) -> dict | None:
    if token.count(".") != 2:
        return None
    _, payload_b64, _ = token.split(".", 2)
    padding = "=" * (-len(payload_b64) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8"))
    except Exception:
        return None


def resolve_api_token(base_url: str, api_token: str) -> str:
    payload = decode_jwt_payload(api_token)
    if not payload or payload.get("token_type") != "refresh":
        return api_token
    response = request_json(
        base_url,
        "/api/token/refresh/",
        api_token="",
        method="POST",
        payload={"refresh": api_token},
        authorization="",
    )
    access_token = response.get("access")
    if not access_token:
        raise RuntimeError("Label Studio token refresh did not return an access token")
    return access_token


def auth_header(api_token: str) -> str:
    # Label Studio uses JWT access tokens with Bearer auth when JWT token auth is enabled.
    return f"Bearer {api_token}" if api_token.count(".") == 2 else f"Token {api_token}"


def request_json(
    base_url: str,
    path: str,
    api_token: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    authorization: str | None = None,
) -> dict | list:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if authorization is None:
        authorization = auth_header(api_token) if api_token else ""
    if authorization:
        headers["Authorization"] = authorization
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        data=body,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def download_bytes(base_url: str, path: str, api_token: str) -> bytes:
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        headers={"Authorization": auth_header(api_token)},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def write_metadata(metadata_out: str | None, metadata: dict) -> None:
    if not metadata_out:
        return
    metadata_path = Path(metadata_out).expanduser().resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def export_project_snapshot(
    *,
    base_url: str,
    api_token: str,
    project_id: int,
    output: str | Path,
    metadata_out: str | Path | None = None,
    title: str = "",
    download_all_tasks: bool = False,
    timeout_seconds: int = 300,
) -> dict:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = None if metadata_out is None else Path(metadata_out).expanduser().resolve()
    resolved_token = resolve_api_token(base_url, api_token)

    if download_all_tasks:
        query = urllib.parse.urlencode({"exportType": "JSON", "download_all_tasks": "true"})
        payload = download_bytes(base_url, f"/api/projects/{project_id}/export?{query}", resolved_token)
        output_path.write_bytes(payload)
        metadata = {
            "project_id": project_id,
            "mode": "direct",
            "base_url": base_url,
            "output": str(output_path),
        }
        write_metadata(str(metadata_path) if metadata_path is not None else None, metadata)
        return metadata

    create_payload = {}
    if title:
        create_payload["title"] = title
    snapshot = request_json(
        base_url,
        f"/api/projects/{project_id}/exports/",
        resolved_token,
        method="POST",
        payload=create_payload,
    )
    snapshot_id = snapshot.get("id")
    if snapshot_id is None:
        raise RuntimeError("Label Studio export creation returned no snapshot id")

    started = time.monotonic()
    current = request_json(base_url, f"/api/projects/{project_id}/exports/{snapshot_id}", resolved_token)
    while current.get("status") not in {"completed", "failed"}:
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for snapshot {snapshot_id}")
        time.sleep(5)
        current = request_json(base_url, f"/api/projects/{project_id}/exports/{snapshot_id}", resolved_token)

    if current.get("status") != "completed":
        raise RuntimeError(f"Snapshot {snapshot_id} finished with status {current.get('status')!r}")

    query = urllib.parse.urlencode({"exportType": "JSON"})
    payload = download_bytes(
        base_url,
        f"/api/projects/{project_id}/exports/{snapshot_id}/download?{query}",
        resolved_token,
    )
    output_path.write_bytes(payload)
    metadata = {
        "project_id": project_id,
        "mode": "snapshot",
        "snapshot_id": snapshot_id,
        "status": current.get("status"),
        "created_at": current.get("created_at"),
        "finished_at": current.get("finished_at"),
        "title": current.get("title") or title or None,
        "base_url": base_url,
        "output": str(output_path),
    }
    write_metadata(str(metadata_path) if metadata_path is not None else None, metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export_project_snapshot(
        base_url=args.base_url,
        api_token=args.api_token,
        project_id=args.project_id,
        output=args.output,
        metadata_out=args.metadata_out,
        title=args.title,
        download_all_tasks=args.download_all_tasks,
        timeout_seconds=args.timeout_seconds,
    )
    print(str(Path(args.output).expanduser().resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
