"""Purpose: Verify offline Model B evaluation metrics, guards, and confusion accounting"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from birdsys.core.model_b_artifacts import DEFAULT_LABELS, HEADS, LoadedModelBArtifact, LoadedModelBMember
from birdsys.ml_experiments import model_b_evaluation as eval_mod


class FakeMemberModel:
    def __init__(self, logits: dict[str, torch.Tensor]) -> None:
        self.logits = logits

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = int(x.shape[0])
        return {head: values[:batch] for head, values in self.logits.items()}


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color="white").save(path)


def _empty_counts() -> dict[str, dict[str, int]]:
    return {head: {label: 0 for label in DEFAULT_LABELS[head]} for head in HEADS}


def test_evaluate_checkpoint_on_frame_tracks_coercions_and_suppressions(tmp_path: Path, monkeypatch) -> None:
    crop1 = tmp_path / "crop1.jpg"
    crop2 = tmp_path / "crop2.jpg"
    _write_image(crop1)
    _write_image(crop2)
    frame = pd.DataFrame(
        [
            {
                "crop_path": str(crop1),
                "isbird": "yes",
                "readability": "readable",
                "specie": "correct",
                "behavior": "resting",
                "substrate": "bare_ground",
                "stance": "bipedal",
            },
            {
                "crop_path": str(crop2),
                "isbird": "yes",
                "readability": "readable",
                "specie": "correct",
                "behavior": "foraging",
                "substrate": "water",
                "stance": "unipedal",
            },
        ]
    )
    logits = {
        "readability": torch.tensor([[8.0, 0.0, 0.0], [0.0, 0.0, 8.0]]),
        "specie": torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        "behavior": torch.tensor(
            [
                [0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "substrate": torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0, 0.0]]),
        "stance": torch.tensor([[0.0, 0.0, 8.0, 0.0], [8.0, 0.0, 0.0, 0.0]]),
    }
    member = LoadedModelBMember(
        name="model_b",
        source_path=tmp_path / "checkpoint.pt",
        model=FakeMemberModel(logits),
        backbone="dummy",
        image_size=8,
        schema_version="annotation_schema_v2",
        checkpoint_heads=tuple(HEADS),
        inference_heads=tuple(HEADS),
        id_to_label={head: DEFAULT_LABELS[head][:] for head in HEADS},
        supported_labels={
            "readability": {"readable", "occluded", "unreadable"},
            "specie": {"correct", "incorrect", "unsure"},
            "behavior": {"resting", "foraging", "unsure"},
            "substrate": {"bare_ground", "water", "unsure"},
            "stance": {"unipedal", "bipedal", "unsure"},
        },
        train_label_counts=_empty_counts(),
    )
    artifact = LoadedModelBArtifact(
        mode="legacy_single_checkpoint",
        source_path=tmp_path / "served",
        schema_version="annotation_schema_v2",
        members=[member],
        id_to_label={head: DEFAULT_LABELS[head][:] for head in HEADS},
        supported_labels=member.supported_labels,
        train_label_counts=_empty_counts(),
        head_to_member={head: "model_b" for head in HEADS},
    )

    monkeypatch.setattr(eval_mod, "load_model_b_artifact", lambda checkpoint_path, device: artifact)

    result = eval_mod.evaluate_checkpoint_on_frame(
        checkpoint_path=tmp_path / "served",
        frame=frame,
        batch_size=2,
        device="cpu",
    )

    assert result.artifact_mode == "legacy_single_checkpoint"
    assert result.metrics["readability_accuracy"] == 0.5
    assert result.diagnostics.unsupported_label_coercions["stance"]["sitting->unsure"] == 1
    assert result.diagnostics.unreadable_or_incorrect_suppressions["behavior"] == 1
    assert isinstance(result.prediction_frame, pd.DataFrame)
    assert len(result.prediction_frame) == 2
