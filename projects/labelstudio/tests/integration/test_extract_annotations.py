from __future__ import annotations

import json
import pathlib

from birdsys.labelstudio import extract_annotations as mod


def test_extract_annotations_local_export_mode(tmp_path: pathlib.Path) -> None:
    export_payload = [
        {
            "id": 7,
            "data": {
                "image": "/data/local-files/?d=birds/black_winged_stilt/labelstudio/images_compressed/q60/img001.jpg",
                "image_id": "img001",
                "source_filename": "img001.jpg",
            },
            "meta": {
                "species_slug": "black_winged_stilt",
                "compression_profile": "q60",
                "original_relative_path": "originals/project7/img001.jpg",
                "served_relative_path": "labelstudio/images_compressed/q60/img001.jpg",
            },
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
                "--data-home",
                str(tmp_path / "birds"),
                "--species-slug",
                "black_winged_stilt",
            ]
        )
        == 0
    )

    species_root = tmp_path / "birds" / "black_winged_stilt"
    export_dir = species_root / "labelstudio" / "exports" / "ann_v001"
    normalized_dir = species_root / "labelstudio" / "normalized" / "ann_v001"
    assert (export_dir / "project_export.json").exists()
    assert (export_dir / "export_metadata.json").exists()
    assert (normalized_dir / "birds.parquet").exists()
    assert (normalized_dir / "images_labels.parquet").exists()
    assert (normalized_dir / "manifest.json").exists()
    assert (normalized_dir / "extract_report.md").exists()
    assert (normalized_dir / "comparison_to_previous.md").exists()
