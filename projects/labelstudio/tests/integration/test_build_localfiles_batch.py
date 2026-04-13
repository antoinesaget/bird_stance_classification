"""Purpose: Verify that local-files batch generation produces the expected import artifacts"""
from __future__ import annotations

import csv
import json
import pathlib
import random

from PIL import Image
from birdsys.labelstudio import build_localfiles_batch as mod


def test_build_labelstudio_localfiles_batch_flat_source(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "lines_project"
    source_root = data_root / "raw_images"
    source_root.mkdir(parents=True, exist_ok=True)

    for name, color in [
        ("a001.jpg", (10, 20, 30)),
        ("a002.jpeg", (200, 30, 90)),
        ("a003.png", (40, 180, 90)),
        ("a004.jpg", (120, 80, 60)),
    ]:
        Image.new("RGB", (128, 96), color=color).save(source_root / name)

    assert mod.main(
        [
        "--data-root",
        str(data_root),
        "--source-relative-root",
        "raw_images",
        "--mirror-relative-root",
        "labelstudio/images_compressed/lines_bw_stilts_q60",
        "--import-relative-root",
        "labelstudio/imports",
        "--batch-name",
        "lines_bw_stilts_3_seed_7_q60",
        "--sample-size",
        "3",
        "--sample-mode",
        "random",
        "--seed",
        "7",
        "--jpeg-quality",
        "60",
        "--dataset-name",
        "lines_project",
        ]
    ) == 0

    import_root = data_root / "labelstudio" / "imports"
    mirror_root = data_root / "labelstudio" / "images_compressed" / "lines_bw_stilts_q60"
    tasks_path = import_root / "lines_bw_stilts_3_seed_7_q60.tasks.json"
    manifest_csv_path = import_root / "lines_bw_stilts_3_seed_7_q60.manifest.csv"
    manifest_json_path = import_root / "lines_bw_stilts_3_seed_7_q60.manifest.json"
    summary_path = import_root / "lines_bw_stilts_3_seed_7_q60.summary.json"

    assert tasks_path.exists()
    assert manifest_csv_path.exists()
    assert manifest_json_path.exists()
    assert summary_path.exists()

    source_images = sorted(source_root.iterdir())
    expected = random.Random(7).sample(source_images, 3)
    expected_names = [path.name for path in expected]

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    assert len(tasks) == 3
    actual_names = [task["data"]["source_filename"] for task in tasks]
    assert actual_names == expected_names
    for idx, task in enumerate(tasks):
        assert task["meta"]["sample_index"] == idx
        assert task["meta"]["sample_seed"] == 7
        assert task["meta"]["batch_name"] == "lines_bw_stilts_3_seed_7_q60"
        assert task["data"]["image"].startswith(
            "/data/local-files/?d=lines_project/labelstudio/images_compressed/lines_bw_stilts_q60/"
        )

    with manifest_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert [row["source_filename"] for row in rows] == expected_names

    manifest = json.loads(manifest_json_path.read_text(encoding="utf-8"))
    assert len(manifest) == 3
    for row in manifest:
        served_path = pathlib.Path(row["served_absolute_path"])
        assert served_path.exists()
        assert served_path.parent == mirror_root
        with Image.open(served_path) as image:
            assert image.size == (128, 96)
        assert row["jpeg_quality"] == 60
        assert int(row["original_bytes"]) > 0
        assert int(row["served_bytes"]) > 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["images_available"] == 4
    assert summary["images_selected"] == 3
    assert summary["jpeg_quality"] == 60
    assert summary["size_ratio"] > 0
