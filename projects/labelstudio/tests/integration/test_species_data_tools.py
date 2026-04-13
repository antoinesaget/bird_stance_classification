from __future__ import annotations

import json
import pathlib

from PIL import Image

from birdsys.core import ensure_layout
from birdsys.labelstudio import build_localfiles_batch as batch_mod
from birdsys.labelstudio import sync_species_data as sync_mod


SPECIES_SLUG = "black_winged_stilt"


def make_image(path: pathlib.Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 90), color=color).save(path, format="JPEG", quality=95)


def test_build_localfiles_batch_emits_species_aware_metadata(tmp_path: pathlib.Path) -> None:
    data_home = tmp_path / "birds"
    layout = ensure_layout(data_home, SPECIES_SLUG)
    make_image(layout.originals / "batch/img001.jpg", color=(10, 20, 30))
    make_image(layout.originals / "batch/img002.jpg", color=(40, 50, 60))

    assert batch_mod.main(
        [
            "--data-home",
            str(data_home),
            "--species-slug",
            SPECIES_SLUG,
            "--source-relative-root",
            "originals/batch",
            "--batch-name",
            "batch001",
            "--sample-mode",
            "first",
            "--sample-size",
            "2",
        ]
    ) == 0

    manifest = json.loads((layout.labelstudio_imports / "batch001.manifest.json").read_text(encoding="utf-8"))
    tasks = json.loads((layout.labelstudio_imports / "batch001.tasks.json").read_text(encoding="utf-8"))
    assert manifest[0]["served_url"].startswith("/data/local-files/?d=birds/black_winged_stilt/")
    assert manifest[0]["compression_profile"] == "q60"
    assert tasks[0]["meta"]["species_slug"] == SPECIES_SLUG
    assert tasks[0]["meta"]["original_relative_path"].startswith("originals/")
    assert tasks[0]["meta"]["served_relative_path"].startswith("labelstudio/images_compressed/q60/")


def test_sync_species_data_local_mode(tmp_path: pathlib.Path) -> None:
    source_home = tmp_path / "source_birds"
    dest_home = tmp_path / "dest_birds"
    source_layout = ensure_layout(source_home, SPECIES_SLUG)
    make_image(source_layout.originals / "project7/img001.jpg", color=(1, 2, 3))
    make_image(source_layout.labelstudio_images_compressed / "q60/img001.jpg", color=(4, 5, 6))
    source_layout.labelstudio_imports.mkdir(parents=True, exist_ok=True)
    (source_layout.labelstudio_imports / "batch001.manifest.json").write_text("[]\n", encoding="utf-8")

    assert sync_mod.main(
        [
            "--species-slug",
            SPECIES_SLUG,
            "--source-host",
            "local",
            "--source-data-home",
            str(source_home),
            "--dest-data-home",
            str(dest_home),
        ]
    ) == 0

    dest_layout = ensure_layout(dest_home, SPECIES_SLUG)
    assert (dest_layout.originals / "project7/img001.jpg").exists()
    assert (dest_layout.labelstudio_images_compressed / "q60/img001.jpg").exists()
    manifest = json.loads((dest_layout.metadata / "sync_manifest.json").read_text(encoding="utf-8"))
    assert manifest["species_slug"] == SPECIES_SLUG
    assert "q60" in manifest["compression_profiles"]
    assert manifest["import_manifests_present"] is True
