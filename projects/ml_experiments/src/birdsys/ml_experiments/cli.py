"""Purpose: Expose training, evaluation, and sandbox commands through the umbrella CLI"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import create_autoresearch_sandbox, evaluate_model_b_checkpoint, train_attributes, train_attributes_cv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="birdsys experiment",
        description="ML experiments subproject commands",
    )
    parser.add_argument(
        "command",
        choices=[
            "train-attributes",
            "train-attributes-cv",
            "evaluate-model-b",
            "create-autoresearch-sandbox",
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

    if args.command == "train-attributes":
        return train_attributes.main(forwarded)
    if args.command == "train-attributes-cv":
        return train_attributes_cv.main(forwarded)
    if args.command == "evaluate-model-b":
        return evaluate_model_b_checkpoint.main(forwarded)
    return create_autoresearch_sandbox.main(forwarded)
