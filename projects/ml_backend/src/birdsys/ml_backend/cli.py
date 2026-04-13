"""Purpose: Expose backend-only operational commands through the umbrella CLI"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import promote_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="birdsys backend",
        description="ML backend subproject commands",
    )
    parser.add_argument("command", choices=["promote-model"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return promote_model.main(forwarded)
