"""Purpose: Expose Label Studio export, import, and prediction commands through the umbrella CLI"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import build_localfiles_batch, create_project, export_snapshot, import_tasks, prefill_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="birdsys labelstudio",
        description="Label Studio subproject commands",
    )
    parser.add_argument(
        "command",
        choices=[
            "export-snapshot",
            "create-project",
            "import-tasks",
            "prefill-predictions",
            "build-localfiles-batch",
        ],
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if args.command == "export-snapshot":
        return export_snapshot.main(forwarded)
    if args.command == "create-project":
        return create_project.main(forwarded)
    if args.command == "import-tasks":
        return import_tasks.main(forwarded)
    if args.command == "prefill-predictions":
        return prefill_predictions.main(forwarded)
    return build_localfiles_batch.main(forwarded)
