from __future__ import annotations

import pathlib
import subprocess
import sys

import pandas as pd
from PIL import Image


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_build_annotation_image_mirror(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    site_dir = data_root / "raw_images" / "siteA"
    site_dir.mkdir(parents=True, exist_ok=True)

    img1 = site_dir / "a001.png"
    img2 = site_dir / "a002.jpg"
    Image.new("RGB", (120, 80), color=(10, 20, 30)).save(img1)
    Image.new("RGB", (64, 64), color=(200, 30, 90)).save(img2)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_annotation_image_mirror.py"),
        "--data-root",
        str(data_root),
        "--site-id",
        "siteA",
        "--quality",
        "60",
        "--max-images",
        "1000",
    ]
    subprocess.run(cmd, check=True)

    out_dir = data_root / "labelstudio" / "images_compressed" / "q60" / "siteA"
    out_a = out_dir / "a001.jpg"
    out_b = out_dir / "a002.jpg"
    assert out_a.exists()
    assert out_b.exists()

    with Image.open(out_a) as im_a:
        assert im_a.size == (120, 80)
    with Image.open(out_b) as im_b:
        assert im_b.size == (64, 64)

    manifest = pd.read_parquet(out_dir / "manifest.parquet")
    assert len(manifest) == 2
    assert set(manifest["image_id"].tolist()) == {"a001", "a002"}
