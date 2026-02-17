from __future__ import annotations

import pathlib
import subprocess
import sys

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_export_normalize_masks_and_defaults(tmp_path: pathlib.Path) -> None:
    export_path = REPO_ROOT / "tests" / "fixtures" / "ann_v001_sample.json"
    data_root = tmp_path / "data"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_normalize.py"),
        "--export-json",
        str(export_path),
        "--annotation-version",
        "ann_v001",
        "--data-root",
        str(data_root),
    ]
    subprocess.run(cmd, check=True)

    birds_path = data_root / "labelstudio" / "normalized" / "ann_v001" / "birds.parquet"
    images_path = data_root / "labelstudio" / "normalized" / "ann_v001" / "images_labels.parquet"

    birds = pd.read_parquet(birds_path)
    images = pd.read_parquet(images_path)

    assert len(images) == 2
    assert len(birds) == 2

    standing = birds[birds["image_id"] == "00001"].iloc[0]
    unreadable = birds[birds["image_id"] == "00002"].iloc[0]

    assert standing["readability"] == "readable"
    assert standing["activity"] == "standing"
    assert standing["legs"] == "unsure"  # default for standing when missing
    assert standing["resting_back"] == "no"  # default for standing when missing

    assert unreadable["readability"] == "unreadable"
    assert pd.isna(unreadable["activity"])
    assert pd.isna(unreadable["support"])
    assert pd.isna(unreadable["legs"])
    assert pd.isna(unreadable["resting_back"])
