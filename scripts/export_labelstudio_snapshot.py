#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and download a Label Studio export snapshot.")
    parser.add_argument("--base-url", required=True, help="Label Studio base URL, e.g. https://birds.ashs.live")
    parser.add_argument("--api-token", required=True, help="Label Studio API token")
    parser.add_argument("--project-id", required=True, type=int, help="Label Studio project ID")
    parser.add_argument("--output", required=True, help="Output JSON export path")
    parser.add_argument("--metadata-out", help="Optional metadata JSON sidecar path")
    parser.add_argument("--title", default="", help="Optional snapshot title")
    parser.add_argument("--download-all-tasks", action="store_true", help="Use the direct export endpoint to include unannotated tasks")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="How long to wait for a background snapshot")
    return parser.parse_args()


def auth_header(api_token: str) -> str:
    # Label Studio uses JWT access tokens with Bearer auth when JWT token auth is enabled.
    return f"Bearer {api_token}" if api_token.count(".") == 2 else f"Token {api_token}"


def request_json(base_url: str, path: str, api_token: str, *, method: str = "GET", payload: dict | None = None) -> dict | list:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        data=body,
        method=method,
        headers={
            "Authorization": auth_header(api_token),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
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


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.download_all_tasks:
        query = urllib.parse.urlencode({"exportType": "JSON", "download_all_tasks": "true"})
        payload = download_bytes(args.base_url, f"/api/projects/{args.project_id}/export?{query}", args.api_token)
        output_path.write_bytes(payload)
        metadata = {
            "project_id": args.project_id,
            "mode": "direct",
            "base_url": args.base_url,
            "output": str(output_path),
        }
        write_metadata(args.metadata_out, metadata)
        print(str(output_path))
        return 0

    create_payload = {}
    if args.title:
        create_payload["title"] = args.title
    snapshot = request_json(
        args.base_url,
        f"/api/projects/{args.project_id}/exports/",
        args.api_token,
        method="POST",
        payload=create_payload,
    )
    snapshot_id = snapshot.get("id")
    if snapshot_id is None:
        raise RuntimeError("Label Studio export creation returned no snapshot id")

    started = time.monotonic()
    current = request_json(args.base_url, f"/api/projects/{args.project_id}/exports/{snapshot_id}", args.api_token)
    while current.get("status") not in {"completed", "failed"}:
        if time.monotonic() - started > args.timeout_seconds:
            raise TimeoutError(f"Timed out waiting for snapshot {snapshot_id}")
        time.sleep(5)
        current = request_json(args.base_url, f"/api/projects/{args.project_id}/exports/{snapshot_id}", args.api_token)

    if current.get("status") != "completed":
        raise RuntimeError(f"Snapshot {snapshot_id} finished with status {current.get('status')!r}")

    query = urllib.parse.urlencode({"exportType": "JSON"})
    payload = download_bytes(
        args.base_url,
        f"/api/projects/{args.project_id}/exports/{snapshot_id}/download?{query}",
        args.api_token,
    )
    output_path.write_bytes(payload)
    metadata = {
        "project_id": args.project_id,
        "mode": "snapshot",
        "snapshot_id": snapshot_id,
        "status": current.get("status"),
        "created_at": current.get("created_at"),
        "finished_at": current.get("finished_at"),
        "title": current.get("title") or args.title or None,
        "base_url": args.base_url,
        "output": str(output_path),
    }
    write_metadata(args.metadata_out, metadata)
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
