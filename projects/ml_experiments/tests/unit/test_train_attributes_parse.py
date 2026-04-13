"""Purpose: Verify argument parsing and config loading for Model B training commands"""
from __future__ import annotations

from birdsys.ml_experiments.train_attributes import parse_args


def test_train_attributes_accepts_eval_split_none(monkeypatch) -> None:
    args = parse_args(
        [
            "--dataset-dir",
            "/tmp/ds",
            "--eval-split",
            "none",
        ]
    )
    assert args.eval_split == "none"
