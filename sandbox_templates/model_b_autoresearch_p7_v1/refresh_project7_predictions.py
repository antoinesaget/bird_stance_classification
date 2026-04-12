#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import IATS_ENV_PATH, PARENT_REPO_ROOT, TRUENAS_ENV_PATH, ROOT, load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a selected run and regenerate untouched project 7 predictions")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_iats = load_env_file(IATS_ENV_PATH)
    env_truenas = load_env_file(TRUENAS_ENV_PATH)

    export_cmd = [sys.executable, str(ROOT / "export_candidate.py"), "--run-id", args.run_id]
    subprocess.run(export_cmd, check=True, cwd=ROOT)

    matches = sorted((ROOT / "runs").glob(f"{args.run_id}*"))
    if not matches:
        raise FileNotFoundError(args.run_id)
    run_dir = matches[0]
    export_report = json.loads((run_dir / "exports" / "current_backend" / "export_report.json").read_text(encoding="utf-8"))
    if export_report.get("status") != "ready":
        raise RuntimeError(f"Run {args.run_id} is not exportable: {export_report}")

    promote_env = dict(**env_iats)
    promote_env.update(
        {
            "MODEL_B_SOURCE": export_report["checkpoint"],
            "PROMOTION_LABEL": f"autoresearch_{args.run_id}",
            "PROMOTION_NOTES": f"Promoted from sandbox run {args.run_id}",
        }
    )
    subprocess.run(
        ["bash", str(PARENT_REPO_ROOT / "scripts" / "ops" / "iats_deploy_model_b_remote.sh")],
        check=True,
        cwd=PARENT_REPO_ROOT,
        env={**os.environ, **promote_env},
    )

    subprocess.run([sys.executable, str(ROOT / "clear_unannotated_project7_predictions.py"), "--project-id", str(args.project_id)], check=True, cwd=ROOT)

    report_out = run_dir / "refresh_project7_predictions_report.json"
    subprocess.run(
        [
            str(PARENT_REPO_ROOT / ".venv" / "bin" / "python"),
            str(PARENT_REPO_ROOT / "scripts" / "prefill_labelstudio_predictions.py"),
            "--base-url",
            env_truenas["LABEL_STUDIO_URL"],
            "--api-token",
            env_truenas["LABEL_STUDIO_API_TOKEN"],
            "--project-id",
            str(args.project_id),
            "--ml-backend-url",
            f"http://127.0.0.1:{env_iats.get('ML_BACKEND_PORT', '9090')}",
            "--only-missing",
            "--report-out",
            str(report_out),
        ],
        check=True,
        cwd=PARENT_REPO_ROOT,
    )
    print(json.dumps({"status": "ok", "run_id": args.run_id, "project_id": args.project_id, "report": str(report_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
