"""Purpose: Verify Model B predictor loading and inference edge cases"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from birdsys.core.attributes import (
    BEHAVIOR_TO_ID,
    READABILITY_TO_ID,
    SPECIE_TO_ID,
    STANCE_TO_ID,
    SUBSTRATE_TO_ID,
)
from birdsys.core.model_b_artifacts import DEFAULT_LABELS, HEADS, LoadedModelBArtifact, LoadedModelBMember
from birdsys.ml_backend.app.predictors.model_a_yolo import Detection
from birdsys.ml_backend.app.predictors.model_b_attributes import AttributePredictor


class DummyMemberModel:
    def __init__(self, logits_by_head: dict[str, torch.Tensor]) -> None:
        self.logits_by_head = logits_by_head

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = int(x.shape[0])
        return {
            head: logits.unsqueeze(0).repeat(batch_size, 1)
            for head, logits in self.logits_by_head.items()
        }


def make_logits(size: int, hot_idx: int) -> torch.Tensor:
    values = torch.full((size,), -5.0, dtype=torch.float32)
    values[hot_idx] = 5.0
    return values


def build_member(
    *,
    name: str,
    inference_heads: list[str],
    logits_by_head: dict[str, torch.Tensor],
) -> LoadedModelBMember:
    return LoadedModelBMember(
        name=name,
        source_path=Path(f"/tmp/{name}.pt"),
        model=DummyMemberModel(logits_by_head),
        backbone="dummy",
        image_size=224,
        schema_version="annotation_schema_v2",
        checkpoint_heads=tuple(inference_heads),
        inference_heads=tuple(inference_heads),
        id_to_label={head: DEFAULT_LABELS[head][:] for head in HEADS},
        supported_labels={head: set(DEFAULT_LABELS[head]) for head in HEADS},
        train_label_counts={head: {} for head in HEADS},
    )


def test_attribute_predictor_stitches_specialist_bundle(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bundle"
    artifact_dir.mkdir()
    image_path = tmp_path / "crop.jpg"
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(image_path)

    r4 = build_member(
        name="r4_context",
        inference_heads=["readability", "substrate"],
        logits_by_head={
            "readability": make_logits(3, READABILITY_TO_ID["readable"]),
            "substrate": make_logits(5, SUBSTRATE_TO_ID["bare_ground"]),
        },
    )
    r5 = build_member(
        name="r5_identity_stance",
        inference_heads=["specie", "stance"],
        logits_by_head={
            "specie": make_logits(3, SPECIE_TO_ID["correct"]),
            "stance": make_logits(4, STANCE_TO_ID["bipedal"]),
        },
    )
    r3 = build_member(
        name="r3_behavior",
        inference_heads=["behavior"],
        logits_by_head={
            "behavior": make_logits(12, BEHAVIOR_TO_ID["resting"]),
        },
    )
    artifact = LoadedModelBArtifact(
        mode="specialist_bundle",
        source_path=artifact_dir,
        schema_version="annotation_schema_v2",
        members=[r4, r5, r3],
        id_to_label={head: DEFAULT_LABELS[head][:] for head in HEADS},
        supported_labels={head: set(DEFAULT_LABELS[head]) for head in HEADS},
        train_label_counts={head: {} for head in HEADS},
        head_to_member={
            "readability": "r4_context",
            "substrate": "r4_context",
            "specie": "r5_identity_stance",
            "stance": "r5_identity_stance",
            "behavior": "r3_behavior",
        },
    )

    monkeypatch.setattr(
        "birdsys.ml_backend.app.predictors.model_b_attributes.load_model_b_artifact",
        lambda checkpoint_path, device: artifact,
    )

    predictor = AttributePredictor(checkpoint_path=artifact_dir)
    predictions = predictor.predict(
        [Detection(x=0.0, y=0.0, w=1.0, h=1.0, score=0.9)],
        image_path=image_path,
    )

    assert predictor.mode == "specialist_bundle"
    assert len(predictor.members) == 3
    assert len(predictions) == 1
    assert predictions[0].readability == "readable"
    assert predictions[0].specie == "correct"
    assert predictions[0].behavior == "resting"
    assert predictions[0].substrate == "bare_ground"
    assert predictions[0].stance == "bipedal"
