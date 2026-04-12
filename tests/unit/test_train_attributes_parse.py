from __future__ import annotations

import sys

from scripts.train_attributes import parse_args


def test_train_attributes_accepts_eval_split_none(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_attributes.py",
            "--dataset-dir",
            "/tmp/ds",
            "--eval-split",
            "none",
        ],
    )
    args = parse_args()
    assert args.eval_split == "none"

