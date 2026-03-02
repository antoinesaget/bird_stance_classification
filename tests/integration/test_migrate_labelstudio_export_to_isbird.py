from __future__ import annotations

import json
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_migration_removes_image_status_and_injects_isbird(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "old_export.json"
    dst = tmp_path / "new_export.json"

    payload = [
        {
            "id": 11,
            "data": {"image": "/data/birds_project/raw_images/v1/abc.jpg"},
            "annotations": [
                {
                    "result": [
                        {
                            "id": "r_11_000",
                            "type": "rectanglelabels",
                            "from_name": "bird_bbox",
                            "to_name": "image",
                            "value": {
                                "x": 1,
                                "y": 2,
                                "width": 3,
                                "height": 4,
                                "rectanglelabels": ["Bird"],
                            },
                        },
                        {
                            "id": "r_11_000",
                            "type": "choices",
                            "from_name": "readability",
                            "to_name": "image",
                            "value": {"choices": ["readable"]},
                        },
                        {
                            "id": "img_11_image_status",
                            "type": "choices",
                            "from_name": "image_status",
                            "to_name": "image",
                            "value": {"choices": ["has_usable_birds"]},
                        },
                    ]
                }
            ],
        }
    ]
    src.write_text(json.dumps(payload), encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "migrate_labelstudio_export_to_isbird.py"),
        "--input-json",
        str(src),
        "--output-json",
        str(dst),
    ]
    subprocess.run(cmd, check=True)

    out = json.loads(dst.read_text(encoding="utf-8"))
    result = out[0]["annotations"][0]["result"]

    assert not any(item.get("from_name") == "image_status" for item in result)
    isbird_items = [item for item in result if item.get("from_name") == "isbird"]
    assert len(isbird_items) == 1
    assert isbird_items[0]["value"]["choices"] == ["yes"]
