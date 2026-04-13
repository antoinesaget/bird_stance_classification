from __future__ import annotations

import json
import pathlib

import pandas as pd
from PIL import Image

from birdsys.core import ensure_layout
from birdsys.datasets import build_dataset as dataset_mod
from birdsys.datasets import build_split as split_mod
from birdsys.datasets import make_crops as crops_mod


def make_image(path: pathlib.Path, *, width: int = 100, height: int = 80, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=color).save(path, format="JPEG", quality=95)
    return str(path)


def image_spec(idx: int, path: str) -> dict:
    readability = ["readable", "occluded", "readable", "readable", "occluded"][idx % 5]
    specie = ["correct", "correct", "unsure", "correct", "correct"][idx % 5]
    behavior = ["resting", "backresting", "moving", "foraging", "resting"][idx % 5]
    substrate = ["bare_ground", "bare_ground", "water", "vegetation", "water"][idx % 5]
    stance = "bipedal" if behavior in {"resting", "backresting"} and substrate in {"bare_ground", "water"} else None
    return {
        "image_id": f"img{idx:03d}",
        "task_id": f"task_{idx:03d}",
        "annotation_id": f"ann_{idx:03d}",
        "site_id": f"site_{idx % 2}",
        "filepath": path,
        "bird": {
            "bird_id": f"task:{idx:03d}:region:r1",
            "region_id": "r1",
            "bbox_x": 0.25,
            "bbox_y": 0.20,
            "bbox_w": 0.30,
            "bbox_h": 0.40,
            "isbird": "yes",
            "readability": readability,
            "specie": specie,
            "behavior": behavior,
            "substrate": substrate,
            "stance": stance,
        },
    }


def write_annotation_version(*, data_root: pathlib.Path, annotation_version: str, specs: list[dict]) -> pathlib.Path:
    layout = ensure_layout(data_root)
    ann_dir = layout.labelstudio_normalized / annotation_version
    ann_dir.mkdir(parents=True, exist_ok=True)

    birds_rows = []
    image_rows = []
    for spec in specs:
        image_rows.append(
            {
                "annotation_version": annotation_version,
                "task_id": spec["task_id"],
                "annotation_id": spec["annotation_id"],
                "image_id": spec["image_id"],
                "image_usable": True,
                "filepath": spec["filepath"],
                "site_id": spec["site_id"],
            }
        )
        bird = spec["bird"]
        birds_rows.append(
            {
                "annotation_version": annotation_version,
                "task_id": spec["task_id"],
                "annotation_id": spec["annotation_id"],
                "region_id": bird["region_id"],
                "image_id": spec["image_id"],
                "bird_id": bird["bird_id"],
                "bbox_x": bird["bbox_x"],
                "bbox_y": bird["bbox_y"],
                "bbox_w": bird["bbox_w"],
                "bbox_h": bird["bbox_h"],
                "bbox_x_px": None,
                "bbox_y_px": None,
                "bbox_w_px": None,
                "bbox_h_px": None,
                "isbird": bird["isbird"],
                "readability": bird["readability"],
                "specie": bird["specie"],
                "behavior": bird["behavior"],
                "substrate": bird["substrate"],
                "stance": bird["stance"],
            }
        )

    pd.DataFrame(birds_rows).to_parquet(ann_dir / "birds.parquet", index=False)
    pd.DataFrame(image_rows).to_parquet(ann_dir / "images_labels.parquet", index=False)
    return ann_dir


def build_specs(tmp_path: pathlib.Path, count: int) -> list[dict]:
    specs = []
    for idx in range(count):
        image_path = tmp_path / "images" / f"img{idx:03d}.jpg"
        filepath = make_image(image_path, color=((idx * 30) % 255, (idx * 70) % 255, (idx * 110) % 255))
        specs.append(image_spec(idx, filepath))
    return specs


def test_split_builder_preserves_previous_test_and_tops_up(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    specs_v1 = build_specs(tmp_path, 6)
    specs_v2 = build_specs(tmp_path, 10)
    write_annotation_version(data_root=data_root, annotation_version="ann_v001", specs=specs_v1)
    write_annotation_version(data_root=data_root, annotation_version="ann_v002", specs=specs_v2)

    assert split_mod.main(["--annotation-version", "ann_v001", "--data-root", str(data_root)]) == 0
    assert split_mod.main(["--annotation-version", "ann_v002", "--data-root", str(data_root)]) == 0

    layout = ensure_layout(data_root)
    split_v1 = layout.derived_splits / "split_v001"
    split_v2 = layout.derived_splits / "split_v002"
    test_v1 = set(pd.read_parquet(split_v1 / "test_groups.parquet")["image_id"].astype(str).tolist())
    test_v2 = set(pd.read_parquet(split_v2 / "test_groups.parquet")["image_id"].astype(str).tolist())
    folds_v2 = pd.read_parquet(split_v2 / "fold_assignments.parquet")
    manifest_v2 = json.loads((split_v2 / "split_manifest.json").read_text(encoding="utf-8"))

    assert test_v1
    assert test_v1.issubset(test_v2)
    assert manifest_v2["test_membership"]["removed"] == 0
    assert set(folds_v2["image_id"].astype(str)).isdisjoint(test_v2)
    assert len(set(folds_v2["fold_id"].astype(int).tolist())) == 5
    assert (split_v2 / "split_report.md").exists()


def test_crop_specs_are_isolated(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    specs = build_specs(tmp_path, 4)
    write_annotation_version(data_root=data_root, annotation_version="ann_v001", specs=specs)

    assert crops_mod.main(["--annotation-version", "ann_v001", "--crop-spec-id", "crop_a", "--data-root", str(data_root), "--margin", "1.0"]) == 0
    assert crops_mod.main(["--annotation-version", "ann_v001", "--crop-spec-id", "crop_b", "--data-root", str(data_root), "--margin", "1.5"]) == 0

    layout = ensure_layout(data_root)
    crop_a = pd.read_parquet(layout.derived_crops / "ann_v001" / "crop_a" / "_crops.parquet")
    crop_b = pd.read_parquet(layout.derived_crops / "ann_v001" / "crop_b" / "_crops.parquet")
    merged = crop_a.merge(crop_b, on="bird_id", suffixes=("_a", "_b"))

    assert (layout.derived_crops / "ann_v001" / "crop_a" / "crop_manifest.json").exists()
    assert (layout.derived_crops / "ann_v001" / "crop_b" / "crop_manifest.json").exists()
    assert any(int(row["crop_width_b"]) > int(row["crop_width_a"]) for _, row in merged.iterrows())


def test_dataset_builder_uses_split_and_crop_artifacts(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    specs = build_specs(tmp_path, 10)
    write_annotation_version(data_root=data_root, annotation_version="ann_v001", specs=specs)

    assert split_mod.main(["--annotation-version", "ann_v001", "--data-root", str(data_root)]) == 0
    assert crops_mod.main(["--annotation-version", "ann_v001", "--crop-spec-id", "crop_a", "--data-root", str(data_root), "--margin", "1.0"]) == 0
    assert dataset_mod.main(
        [
            "--annotation-version",
            "ann_v001",
            "--split-version",
            "split_v001",
            "--crop-spec-id",
            "crop_a",
            "--data-root",
            str(data_root),
            "--dataset-version",
            "ds_v001",
        ]
    ) == 0

    assert crops_mod.main(["--annotation-version", "ann_v001", "--crop-spec-id", "crop_b", "--data-root", str(data_root), "--margin", "1.35"]) == 0
    assert dataset_mod.main(
        [
            "--annotation-version",
            "ann_v001",
            "--split-version",
            "split_v001",
            "--crop-spec-id",
            "crop_b",
            "--data-root",
            str(data_root),
            "--dataset-version",
            "ds_v002",
        ]
    ) == 0

    layout = ensure_layout(data_root)
    ds_v1 = layout.derived_datasets / "ds_v001"
    ds_v2 = layout.derived_datasets / "ds_v002"
    manifest_v1 = json.loads((ds_v1 / "manifest.json").read_text(encoding="utf-8"))
    manifest_v2 = json.loads((ds_v2 / "manifest.json").read_text(encoding="utf-8"))
    train_pool_df = pd.read_parquet(ds_v2 / "train_pool.parquet")
    test_df = pd.read_parquet(ds_v2 / "test.parquet")
    all_df = pd.read_parquet(ds_v2 / "all_data.parquet")

    assert manifest_v1["crop_spec_id"] == "crop_a"
    assert manifest_v2["crop_spec_id"] == "crop_b"
    assert manifest_v2["comparison_to_previous"] is not None
    assert manifest_v2["comparison_to_previous"]["delta_counts"]["rows_total"] == 0.0
    assert set(test_df["image_id"].astype(str)).isdisjoint(set(train_pool_df["image_id"].astype(str)))
    assert set(all_df["split_role"].astype(str).unique().tolist()) == {"test", "train_pool"}
    assert (ds_v2 / "fold_assignments.parquet").exists()
    assert (ds_v2 / "dataset_report.md").exists()
    assert (ds_v2 / "plots" / "crop_spec_comparison.png").exists()
