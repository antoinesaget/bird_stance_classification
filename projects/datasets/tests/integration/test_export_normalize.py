from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from birdsys.datasets import export_normalize as mod


SPECIES_SLUG = "black_winged_stilt"


def write_export(path: pathlib.Path, payload: list[dict]) -> pathlib.Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def task_payload(
    *,
    task_id: int,
    image_name: str,
    annotation_id: int | None,
    result: list[dict],
    updated_at: str = "2026-04-13T12:00:00Z",
    cancelled: bool = False,
) -> dict:
    annotations = []
    if annotation_id is not None:
        annotations.append(
            {
                "id": annotation_id,
                "updated_at": updated_at,
                "was_cancelled": cancelled,
                "result": result,
            }
        )
    compressed_relpath = f"labelstudio/images_compressed/q60/{image_name}.jpg"
    return {
        "id": task_id,
        "data": {
            "image": f"/data/local-files/?d=birds/{SPECIES_SLUG}/{compressed_relpath}",
            "image_id": image_name,
            "source_filename": f"{image_name}.jpg",
        },
        "meta": {
            "species_slug": SPECIES_SLUG,
            "compression_profile": "q60",
            "original_relative_path": f"originals/project7/{image_name}.jpg",
            "served_relative_path": compressed_relpath,
        },
        "annotations": annotations,
    }


def region_result(*, region_id: str, isbird: str, readability: str | None, specie: str | None, behavior: str | None, substrate: str | None, stance: str | None) -> list[dict]:
    items = [
        {
            "id": region_id,
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
            "id": region_id,
            "from_name": "isbird",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": [isbird]},
        },
    ]
    if readability is not None:
        items.append({"id": region_id, "from_name": "readability", "to_name": "image", "type": "choices", "value": {"choices": [readability]}})
    if specie is not None:
        items.append({"id": region_id, "from_name": "specie", "to_name": "image", "type": "choices", "value": {"choices": [specie]}})
    if behavior is not None:
        items.append({"id": region_id, "from_name": "behavior", "to_name": "image", "type": "choices", "value": {"choices": [behavior]}})
    if substrate is not None:
        items.append({"id": region_id, "from_name": "substrate", "to_name": "image", "type": "choices", "value": {"choices": [substrate]}})
    if stance is not None:
        items.append({"id": region_id, "from_name": "stance", "to_name": "image", "type": "choices", "value": {"choices": [stance]}})
    return items


def test_export_normalize_strict_current_schema_and_reports(tmp_path: pathlib.Path) -> None:
    export_path = write_export(
        tmp_path / "ann_v001.json",
        [
            {
                **task_payload(task_id=101, image_name="img001", annotation_id=None, result=[]),
                "annotations": [
                    {
                        "id": 10,
                        "updated_at": "2026-04-12T10:00:00Z",
                        "was_cancelled": False,
                        "result": region_result(
                            region_id="r1",
                            isbird="yes",
                            readability="readable",
                            specie="correct",
                            behavior="moving",
                            substrate="water",
                            stance=None,
                        ),
                    },
                    {
                        "id": 11,
                        "updated_at": "2026-04-13T10:00:00Z",
                        "was_cancelled": False,
                        "result": region_result(
                            region_id="r1",
                            isbird="yes",
                            readability="readable",
                            specie="correct",
                            behavior="resting",
                            substrate="bare_ground",
                            stance=None,
                        ),
                    },
                ],
            },
            task_payload(task_id=202, image_name="img002", annotation_id=20, result=[]),
            task_payload(task_id=303, image_name="img003", annotation_id=None, result=[]),
        ],
    )
    result = mod.normalize_export(
        export_json=export_path,
        annotation_version="ann_v001",
        data_home=tmp_path / "birds",
        species_slug=SPECIES_SLUG,
    )

    birds = pd.read_parquet(result.birds_out)
    images = pd.read_parquet(result.images_out)
    manifest = json.loads(result.manifest_out.read_text(encoding="utf-8"))
    extract_report = json.loads(result.extract_report_json.read_text(encoding="utf-8"))

    assert len(images) == 2
    assert len(birds) == 1
    assert manifest["previous_annotation_version"] is None
    assert manifest["species_slug"] == SPECIES_SLUG
    assert extract_report["counts"]["tasks_total_raw"] == 3
    assert extract_report["counts"]["tasks_with_kept_annotation"] == 2
    assert extract_report["counts"]["tasks_skipped_without_kept_annotation"] == 1

    image_row = images.iloc[0]
    assert image_row["original_relpath"] == "originals/project7/img001.jpg"
    assert image_row["compressed_relpath"] == "labelstudio/images_compressed/q60/img001.jpg"
    assert image_row["labelstudio_localfiles_relpath"] == "birds/black_winged_stilt/labelstudio/images_compressed/q60/img001.jpg"

    row = birds.iloc[0]
    assert row["task_id"] == "101"
    assert row["annotation_id"] == "11"
    assert row["region_id"] == "r1"
    assert row["bird_id"] == "task:101:region:r1"
    assert row["behavior"] == "resting"
    assert row["substrate"] == "bare_ground"
    assert row["stance"] == "unsure"

    plots_dir = result.out_dir / "plots"
    for name in (
        "current_extract_summary.png",
        "current_null_counts.png",
        "current_label_distributions.png",
        "comparison_row_deltas.png",
        "comparison_field_changes.png",
        "comparison_label_deltas.png",
    ):
        assert (plots_dir / name).exists(), name


def test_export_normalize_rejects_missing_required_task_metadata(tmp_path: pathlib.Path) -> None:
    export_path = write_export(
        tmp_path / "bad.json",
        [
            {
                "id": 7,
                "data": {"image": "/data/local-files/?d=birds/black_winged_stilt/labelstudio/images_compressed/q60/legacy001.jpg"},
                "annotations": [
                    {
                        "id": 1,
                        "updated_at": "2026-04-13T12:00:00Z",
                        "was_cancelled": False,
                        "result": region_result(
                            region_id="r1",
                            isbird="yes",
                            readability="readable",
                            specie="correct",
                            behavior="resting",
                            substrate="water",
                            stance=None,
                        ),
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="meta\\.species_slug"):
        mod.normalize_export(
            export_json=export_path,
            annotation_version="ann_v001",
            data_home=tmp_path / "birds",
            species_slug=SPECIES_SLUG,
        )


def test_export_normalize_rejects_legacy_value(tmp_path: pathlib.Path) -> None:
    export_path = write_export(
        tmp_path / "legacy.json",
        [
            task_payload(
                task_id=7,
                image_name="legacy001",
                annotation_id=1,
                result=region_result(
                    region_id="legacy-region",
                    isbird="yes",
                    readability="readable",
                    specie="correct",
                    behavior="resting",
                    substrate="ground",
                    stance="one",
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="not in the current schema"):
        mod.normalize_export(
            export_json=export_path,
            annotation_version="ann_v001",
            data_home=tmp_path / "birds",
            species_slug=SPECIES_SLUG,
        )


def test_export_normalize_compares_to_previous_extract(tmp_path: pathlib.Path) -> None:
    data_home = tmp_path / "birds"
    write_export(
        tmp_path / "ann_v001.json",
        [
            task_payload(
                task_id=101,
                image_name="img001",
                annotation_id=11,
                result=region_result(
                    region_id="r1",
                    isbird="yes",
                    readability="readable",
                    specie="correct",
                    behavior="resting",
                    substrate="bare_ground",
                    stance=None,
                ),
            )
        ],
    )
    mod.normalize_export(
        export_json=tmp_path / "ann_v001.json",
        annotation_version="ann_v001",
        data_home=data_home,
        species_slug=SPECIES_SLUG,
    )

    write_export(
        tmp_path / "ann_v002.json",
        [
            task_payload(
                task_id=101,
                image_name="img001",
                annotation_id=12,
                result=region_result(
                    region_id="r1",
                    isbird="yes",
                    readability="readable",
                    specie="correct",
                    behavior="moving",
                    substrate="water",
                    stance=None,
                ),
            ),
            task_payload(
                task_id=202,
                image_name="img002",
                annotation_id=22,
                result=region_result(
                    region_id="r2",
                    isbird="yes",
                    readability="occluded",
                    specie="unsure",
                    behavior="foraging",
                    substrate="water",
                    stance=None,
                ),
            ),
        ],
    )
    result = mod.normalize_export(
        export_json=tmp_path / "ann_v002.json",
        annotation_version="ann_v002",
        data_home=data_home,
        species_slug=SPECIES_SLUG,
    )
    comparison = json.loads(result.comparison_json.read_text(encoding="utf-8"))

    assert comparison["previous_annotation_version"] == "ann_v001"
    assert comparison["images"]["added"] == 1
    assert comparison["birds"]["added"] == 1
    assert comparison["birds"]["changed"] == 1
    assert comparison["field_change_counts"]["behavior"] == 1
    assert comparison["field_change_counts"]["substrate"] == 1
