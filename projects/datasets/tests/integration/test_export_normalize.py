"""Purpose: Verify that export normalization produces the expected parquet outputs and labels."""
from __future__ import annotations

import json
import pathlib

import pandas as pd
from birdsys.datasets import export_normalize as mod


def test_export_normalize_masks_and_defaults(tmp_path: pathlib.Path) -> None:
    export_path = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ann_v001_sample.json"
    data_root = tmp_path / "data"

    assert mod.main(
        [
        "--export-json",
        str(export_path),
        "--annotation-version",
        "ann_v001",
        "--data-root",
        str(data_root),
        ]
    ) == 0

    birds_path = data_root / "labelstudio" / "normalized" / "ann_v001" / "birds.parquet"
    images_path = data_root / "labelstudio" / "normalized" / "ann_v001" / "images_labels.parquet"

    birds = pd.read_parquet(birds_path)
    images = pd.read_parquet(images_path)

    assert len(images) == 2
    assert len(birds) == 2

    standing = birds[birds["image_id"] == "00001"].iloc[0]
    non_bird = birds[birds["image_id"] == "00002"].iloc[0]

    assert standing["isbird"] == "yes"
    assert standing["readability"] == "readable"
    assert standing["specie"] == "correct"
    assert standing["behavior"] == "resting"
    assert standing["substrate"] == "bare_ground"
    assert standing["stance"] == "unsure"  # default for resting+ground when missing
    assert bool(images[images["image_id"] == "00001"].iloc[0]["image_usable"]) is True

    assert non_bird["isbird"] == "no"
    assert pd.isna(non_bird["readability"])
    assert pd.isna(non_bird["specie"])
    assert pd.isna(non_bird["behavior"])
    assert pd.isna(non_bird["substrate"])
    assert pd.isna(non_bird["stance"])
    assert bool(images[images["image_id"] == "00002"].iloc[0]["image_usable"]) is False


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
                            "from_name": "isbird",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "choices": ["yes"],
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
                            "from_name": "specie",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "choices": ["correct"],
                            },
                        },
                        {
                            "id": "r_1_000",
                            "from_name": "behavior",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "choices": ["bathing"],
                            },
                        },
                        {
                            "id": "r_1_000",
                            "from_name": "substrate",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "choices": ["vegetation"],
                            },
                        },
                        {
                            "id": "r_1_000",
                            "from_name": "stance",
                            "to_name": "image",
                            "type": "choices",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rotation": 0,
                                "choices": ["sitting"],
                            },
                        },
                    ]
                }
            ],
        }
    ]
    export_path.write_text(json.dumps(export_payload), encoding="utf-8")

    assert mod.main(
        [
        "--export-json",
        str(export_path),
        "--annotation-version",
        "ann_v002",
        "--data-root",
        str(data_root),
        ]
    ) == 0

    birds_path = data_root / "labelstudio" / "normalized" / "ann_v002" / "birds.parquet"
    birds = pd.read_parquet(birds_path)
    assert len(birds) == 1
    row = birds.iloc[0]
    assert row["isbird"] == "yes"
    assert row["readability"] == "readable"
    assert row["specie"] == "correct"
    assert row["behavior"] == "bathing"
    assert row["substrate"] == "vegetation"
    assert pd.isna(row["stance"])


def test_export_normalize_masks_on_specie_incorrect(tmp_path: pathlib.Path) -> None:
    export_path = tmp_path / "ann_v003_specie_incorrect.json"
    data_root = tmp_path / "data"

    payload = [
        {
            "id": 7,
            "data": {"image": "/data/birds_project/raw_images/scolop2/00007_651092999.jpg"},
            "annotations": [
                {
                    "result": [
                        {
                            "id": "r_7_000",
                            "from_name": "bird_bbox",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": 15.0,
                                "y": 25.0,
                                "width": 20.0,
                                "height": 20.0,
                                "rectanglelabels": ["Bird"],
                            },
                        },
                        {
                            "id": "r_7_000",
                            "from_name": "isbird",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["yes"]},
                        },
                        {
                            "id": "r_7_000",
                            "from_name": "readability",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["readable"]},
                        },
                        {
                            "id": "r_7_000",
                            "from_name": "specie",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["incorrect"]},
                        },
                        {
                            "id": "r_7_000",
                            "from_name": "behavior",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["resting"]},
                        },
                        {
                            "id": "r_7_000",
                            "from_name": "substrate",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["ground"]},
                        },
                        {
                            "id": "r_7_000",
                            "from_name": "legs",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["one"]},
                        },
                    ]
                }
            ],
        }
    ]
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    assert mod.main(
        [
        "--export-json",
        str(export_path),
        "--annotation-version",
        "ann_v003",
        "--data-root",
        str(data_root),
        ]
    ) == 0

    birds = pd.read_parquet(data_root / "labelstudio" / "normalized" / "ann_v003" / "birds.parquet")
    assert len(birds) == 1
    row = birds.iloc[0]
    assert row["isbird"] == "yes"
    assert row["readability"] == "readable"
    assert row["specie"] == "incorrect"
    assert pd.isna(row["behavior"])
    assert pd.isna(row["substrate"])
    assert pd.isna(row["stance"])


def test_export_normalize_image_usable_false_without_bboxes(tmp_path: pathlib.Path) -> None:
    export_path = tmp_path / "ann_v004_no_bbox.json"
    data_root = tmp_path / "data"

    payload = [
        {
            "id": 1,
            "data": {"image": "/data/birds_project/raw_images/scolop2/00010_651092999.jpg"},
            "annotations": [{"result": []}],
        }
    ]
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    assert mod.main(
        [
        "--export-json",
        str(export_path),
        "--annotation-version",
        "ann_v004",
        "--data-root",
        str(data_root),
        ]
    ) == 0

    images = pd.read_parquet(data_root / "labelstudio" / "normalized" / "ann_v004" / "images_labels.parquet")
    birds = pd.read_parquet(data_root / "labelstudio" / "normalized" / "ann_v004" / "birds.parquet")
    report = json.loads((data_root / "labelstudio" / "normalized" / "ann_v004" / "migration_report.json").read_text(encoding="utf-8"))
    assert len(images) == 1
    assert len(birds) == 0
    assert bool(images.iloc[0]["image_usable"]) is False
    assert report["mask_stats"]["total_birds"] == 0
