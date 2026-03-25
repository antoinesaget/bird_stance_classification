#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from label_studio_sdk.client import LabelStudio


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


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = LabelStudio(base_url=args.base_url, api_key=args.api_token)

    if args.download_all_tasks:
        chunks = client.projects.exports.download_sync(
            args.project_id,
            export_type="JSON",
            download_all_tasks=True,
        )
        with output_path.open("wb") as handle:
            for chunk in chunks:
                handle.write(chunk)

        metadata = {
            "project_id": args.project_id,
            "mode": "direct",
            "base_url": args.base_url,
            "output": str(output_path),
        }
    else:
        snapshot = client.projects.exports.create(args.project_id, title=args.title or None)
        if snapshot.id is None:
            raise RuntimeError("Label Studio export creation returned no snapshot id")

        started = time.monotonic()
        current = client.projects.exports.get(args.project_id, str(snapshot.id))
        while current.status not in {"completed", "failed"}:
            if time.monotonic() - started > args.timeout_seconds:
                raise TimeoutError(f"Timed out waiting for snapshot {snapshot.id}")
            time.sleep(5)
            current = client.projects.exports.get(args.project_id, str(snapshot.id))

        if current.status != "completed":
            raise RuntimeError(f"Snapshot {snapshot.id} finished with status {current.status!r}")

        with output_path.open("wb") as handle:
            for chunk in client.projects.exports.download(args.project_id, str(snapshot.id), export_type="JSON"):
                handle.write(chunk)

        metadata = {
            "project_id": args.project_id,
            "mode": "snapshot",
            "snapshot_id": snapshot.id,
            "status": current.status,
            "created_at": str(getattr(current, "created_at", None)),
            "finished_at": str(getattr(current, "finished_at", None)),
            "title": args.title or None,
            "base_url": args.base_url,
            "output": str(output_path),
        }

    if args.metadata_out:
        metadata_path = Path(args.metadata_out).expanduser().resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
