from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="birdsys",
        description="BirdSys monorepo umbrella CLI",
    )
    parser.add_argument(
        "project",
        choices=["labelstudio", "datasets", "backend", "experiment"],
        help="Subproject command group to run",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected subproject CLI",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if args.project == "labelstudio":
        from birdsys.labelstudio.cli import main as target
    elif args.project == "datasets":
        from birdsys.datasets.cli import main as target
    elif args.project == "backend":
        from birdsys.ml_backend.cli import main as target
    else:
        from birdsys.ml_experiments.cli import main as target
    return target(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
