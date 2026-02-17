from __future__ import annotations

import json
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


def test_export_normalize_supports_region_id_linked_choices(tmp_path: pathlib.Path) -> None:
    export_path = tmp_path / "ann_v002_region_id_style.json"
    data_root = tmp_path / "data"

    export_payload = [
        {
            "id": 1,
            "data": {"image": "/data/birds_project/raw_images/scolop2/00003_651092721.jpg"},
            "annotations": [
                {
                    "result": [
                        {
                            "id": "r_1_000",
                            "from_name": "bird_bbox",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "rectanglelabels": ["Bird"],
                            },
                        },
                        {
                            "id": "r_1_000",
                            "from_name": "readability",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "choices": ["readable"],
                            },
                        },
                        {
                            "id": "r_1_000",
                            "from_name": "activity",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "choices": ["standing"],
                            },
                        },
                        {
                            "id": "r_1_000",
                            "from_name": "support",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "choices": ["ground"],
                            },
                        },
                        {
                            "id": "r_1_000",
                            "from_name": "legs",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "choices": ["two"],
                            },
                        },
                        {
                            "id": "r_1_000",
                            "from_name": "resting_back",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "choices": ["no"],
                            },
                        },
                        {
                            "id": "img_1_image_status",
                            "from_name": "image_status",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["has_usable_birds"]},
                        },
                    ]
                }
            ],
        }
    ]
    export_path.write_text(json.dumps(export_payload), encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_normalize.py"),
        "--export-json",
        str(export_path),
        "--annotation-version",
        "ann_v002",
        "--data-root",
        str(data_root),
    ]
    subprocess.run(cmd, check=True)

    birds_path = data_root / "labelstudio" / "normalized" / "ann_v002" / "birds.parquet"
    birds = pd.read_parquet(birds_path)
    assert len(birds) == 1
    row = birds.iloc[0]
    assert row["readability"] == "readable"
    assert row["activity"] == "standing"
    assert row["support"] == "ground"
    assert row["legs"] == "two"
    assert row["resting_back"] == "no"
