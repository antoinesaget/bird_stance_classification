#!/usr/bin/env python3
"""Purpose: Validate and export a sandbox candidate artifact into the backend-compatible shape"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from common import ROOT, extend_workspace_sys_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and export a selected run for the current backend")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def find_run(run_id: str) -> Path:
    matches = list((ROOT / "runs").glob(f"{run_id}*"))
    if not matches:
        raise FileNotFoundError(f"No run matches {run_id}")
    return sorted(matches)[0]


def main() -> int:
    args = parse_args()
    run_dir = find_run(args.run_id)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    out_dir = run_dir / "exports" / "current_backend"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_checkpoint = run_dir / "candidate" / "current_backend" / "checkpoint.pt"
    report_path = out_dir / "export_report.json"
    if not source_checkpoint.exists():
        report = {
            "status": "not_exportable_yet",
            "reason": "candidate checkpoint not present for this run",
            "run_id": summary["run_id"],
        }
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0

    payload = torch.load(source_checkpoint, map_location="cpu")
    required_keys = {"model_state", "backbone", "image_size", "schema_version", "label_maps", "supported_labels"}
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        report = {
            "status": "not_exportable_yet",
            "reason": f"checkpoint missing required keys: {missing}",
            "run_id": summary["run_id"],
        }
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0

    exported_checkpoint = out_dir / "checkpoint.pt"
    if source_checkpoint.resolve() != exported_checkpoint.resolve():
        shutil.copy2(source_checkpoint, exported_checkpoint)

    extend_workspace_sys_path()
    from birdsys.ml_backend.app.predictors.model_b_attributes import AttributePredictor

    predictor = AttributePredictor(checkpoint_path=exported_checkpoint)
    report = {
        "status": "ready" if predictor.model is not None else "not_exportable_yet",
        "run_id": summary["run_id"],
        "checkpoint": str(exported_checkpoint),
        "schema_version": payload.get("schema_version"),
        "supported_labels": payload.get("supported_labels"),
        "model_loaded": predictor.model is not None,
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0 if predictor.model is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
