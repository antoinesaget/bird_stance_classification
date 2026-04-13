from __future__ import annotations

import json
import pathlib

from birdsys.labelstudio import extract_annotations as mod


def test_extract_annotations_local_export_mode(tmp_path: pathlib.Path) -> None:
    export_payload = [
        {
            "id": 7,
            "data": {"image": "/data/birds_project/raw_images/project7/img001.jpg"},
            "annotations": [
                {
                    "id": 70,
                    "updated_at": "2026-04-13T12:00:00Z",
                    "was_cancelled": False,
                    "result": [
                        {
                            "id": "r1",
                            "from_name": "bird_bbox",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rectanglelabels": ["Bird"],
                            },
                        },
                        {
                            "id": "r1",
                            "from_name": "isbird",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["yes"]},
                        },
                        {
                            "id": "r1",
                            "from_name": "readability",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["readable"]},
                        },
                        {
                            "id": "r1",
                            "from_name": "specie",
                            "to_name": "image",
                            "type": "choices",
                            "value": {"choices": ["correct"]},
                        },
                    ],
                }
            ],
        }
    ]
    export_path = tmp_path / "raw_export.json"
    export_path.write_text(json.dumps(export_payload, indent=2) + "\n", encoding="utf-8")

    assert (
        mod.main(
            [
                "--export-json",
                str(export_path),
                "--annotation-version",
                "ann_v001",
                "--data-root",
                str(tmp_path / "data"),
            ]
        )
        == 0
    )

    export_dir = tmp_path / "data" / "labelstudio" / "exports" / "ann_v001"
    normalized_dir = tmp_path / "data" / "labelstudio" / "normalized" / "ann_v001"
    assert (export_dir / "project_export.json").exists()
    assert (export_dir / "export_metadata.json").exists()
    assert (normalized_dir / "birds.parquet").exists()
    assert (normalized_dir / "images_labels.parquet").exists()
    assert (normalized_dir / "manifest.json").exists()
    assert (normalized_dir / "extract_report.md").exists()
    assert (normalized_dir / "comparison_to_previous.md").exists()
