from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from birdsys.training.model_b_artifacts import HEADS, load_model_b_artifact


class DummyModel:
    def __init__(self, backbone_name: str, pretrained: bool, heads: list[str] | None = None) -> None:
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.heads = heads or HEADS[:]
        self.loaded_state = None
        self.loaded_device = None

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.loaded_state = state

    def eval(self) -> DummyModel:
        return self

    def to(self, device: torch.device | str) -> DummyModel:
        self.loaded_device = str(device)
        return self


def _write_checkpoint(path: Path, *, active_heads: list[str] | None, backbone: str, image_size: int) -> None:
    torch.save(
        {
            "model_state": {},
            "backbone": backbone,
            "image_size": image_size,
            "schema_version": "annotation_schema_v2",
            "active_heads": active_heads,
            "label_maps": {},
            "supported_labels": {},
            "train_label_counts": {},
        },
        path,
    )


def test_load_model_b_artifact_accepts_legacy_checkpoint_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("birdsys.training.model_b_artifacts.MultiHeadAttributeModel", DummyModel)
    artifact_dir = tmp_path / "legacy"
    artifact_dir.mkdir()
    _write_checkpoint(
        artifact_dir / "checkpoint.pt",
        active_heads=None,
        backbone="convnextv2_tiny.fcmae_ft_in1k",
        image_size=384,
    )

    artifact = load_model_b_artifact(artifact_dir, device="cpu")

    assert artifact.mode == "legacy_single_checkpoint"
    assert artifact.source_path == artifact_dir.resolve()
    assert len(artifact.members) == 1
    assert artifact.members[0].checkpoint_heads == tuple(HEADS)
    assert artifact.head_to_member == {head: "model_b" for head in HEADS}


def test_load_model_b_artifact_accepts_specialist_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("birdsys.training.model_b_artifacts.MultiHeadAttributeModel", DummyModel)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_checkpoint(
        bundle_dir / "r4.pt",
        active_heads=["readability", "substrate"],
        backbone="vit_small_patch16_dinov3.lvd1689m",
        image_size=256,
    )
    _write_checkpoint(
        bundle_dir / "r5.pt",
        active_heads=["specie", "stance"],
        backbone="vit_base_patch16_dinov3.lvd1689m",
        image_size=256,
    )
    _write_checkpoint(
        bundle_dir / "r3.pt",
        active_heads=["behavior"],
        backbone="convnext_tiny.dinov3_lvd1689m",
        image_size=224,
    )
    manifest = {
        "artifact_type": "model_b_specialist_bundle",
        "schema_version": "annotation_schema_v2",
        "members": [
            {
                "name": "r4_context",
                "checkpoint": "r4.pt",
                "heads": ["readability", "substrate"],
                "backbone": "vit_small_patch16_dinov3.lvd1689m",
                "image_size": 256,
            },
            {
                "name": "r5_identity_stance",
                "checkpoint": "r5.pt",
                "heads": ["specie", "stance"],
                "backbone": "vit_base_patch16_dinov3.lvd1689m",
                "image_size": 256,
            },
            {
                "name": "r3_behavior",
                "checkpoint": "r3.pt",
                "heads": ["behavior"],
                "backbone": "convnext_tiny.dinov3_lvd1689m",
                "image_size": 224,
            },
        ],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    artifact = load_model_b_artifact(bundle_dir, device="cpu")

    assert artifact.mode == "specialist_bundle"
    assert set(artifact.head_to_member) == set(HEADS)
    assert artifact.head_to_member["behavior"] == "r3_behavior"
    assert artifact.head_to_member["readability"] == "r4_context"
    assert artifact.head_to_member["stance"] == "r5_identity_stance"


def test_load_model_b_artifact_rejects_incomplete_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("birdsys.training.model_b_artifacts.MultiHeadAttributeModel", DummyModel)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _write_checkpoint(
        bundle_dir / "r4.pt",
        active_heads=["readability", "substrate"],
        backbone="vit_small_patch16_dinov3.lvd1689m",
        image_size=256,
    )
    manifest = {
        "artifact_type": "model_b_specialist_bundle",
        "schema_version": "annotation_schema_v2",
        "members": [
            {
                "name": "r4_context",
                "checkpoint": "r4.pt",
                "heads": ["readability", "substrate"],
                "backbone": "vit_small_patch16_dinov3.lvd1689m",
                "image_size": 256,
            }
        ],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cover all heads"):
        load_model_b_artifact(bundle_dir, device="cpu")
