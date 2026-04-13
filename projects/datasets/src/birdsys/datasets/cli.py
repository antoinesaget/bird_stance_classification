from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import build_dataset, export_normalize, make_crops


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="birdsys datasets",
        description="Dataset subproject commands",
    )
    parser.add_argument(
        "command",
        choices=["normalize-export", "make-crops", "build-dataset"],
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if args.command == "normalize-export":
        return export_normalize.main(forwarded)
    if args.command == "make-crops":
        return make_crops.main(forwarded)
    return build_dataset.main(forwarded)
