from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

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


def _artifact(tmp_path: Path, *, score_shift: float) -> LoadedModelBArtifact:
    logits = {
        "readability": torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        "specie": torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        "behavior": torch.tensor(
            [
                [8.0 + score_shift, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 8.0 + score_shift, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "substrate": torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 8.0 + score_shift, 0.0, 0.0]]),
        "stance": torch.tensor([[0.0, 8.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]),
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
        supported_labels={head: set(DEFAULT_LABELS[head]) for head in HEADS},
        train_label_counts=_empty_counts(),
    )
    return LoadedModelBArtifact(
        mode="legacy_single_checkpoint",
        source_path=tmp_path / "served",
        schema_version="annotation_schema_v2",
        members=[member],
        id_to_label={head: DEFAULT_LABELS[head][:] for head in HEADS},
        supported_labels={head: set(DEFAULT_LABELS[head]) for head in HEADS},
        train_label_counts=_empty_counts(),
        head_to_member={head: "model_b" for head in HEADS},
    )


def test_evaluate_checkpoint_on_frame_tracks_comparison_and_predictions(tmp_path: Path, monkeypatch) -> None:
    crop1 = tmp_path / "crop1.jpg"
    crop2 = tmp_path / "crop2.jpg"
    _write_image(crop1)
    _write_image(crop2)
    frame = pd.DataFrame(
        [
            {
                "crop_path": str(crop1),
                "row_id": "row-1",
                "image_id": "img-1",
                "group_id": "g-1",
                "isbird": "yes",
                "readability": "readable",
                "specie": "correct",
                "behavior": "flying",
                "substrate": "bare_ground",
                "stance": "bipedal",
            },
            {
                "crop_path": str(crop2),
                "row_id": "row-2",
                "image_id": "img-2",
                "group_id": "g-2",
                "isbird": "yes",
                "readability": "readable",
                "specie": "correct",
                "behavior": "foraging",
                "substrate": "water",
                "stance": "unipedal",
            },
        ]
    )

    candidate = _artifact(tmp_path / "candidate", score_shift=0.0)
    baseline = _artifact(tmp_path / "baseline", score_shift=-4.0)

    def _load_model_b_artifact(checkpoint_path: Path, device: str):
        if "baseline" in str(checkpoint_path):
            return baseline
        return candidate

    monkeypatch.setattr(eval_mod, "load_model_b_artifact", _load_model_b_artifact)

    result = eval_mod.evaluate_checkpoint_on_frame(
        checkpoint_path=tmp_path / "candidate",
        frame=frame,
        batch_size=2,
        device="cpu",
        baseline_checkpoint_path=tmp_path / "baseline",
    )

    assert result.summary_metrics["behavior_visible_support"] == 2.0
    assert result.aggregate_metrics["primary_score"] >= 0.0
    assert result.comparison_to_baseline is not None
    assert "delta_summary_metrics" in result.comparison_to_baseline.to_dict()
    assert list(result.prediction_frame["row_id"]) == ["row-1", "row-2"]
    assert "behavior_pred" in result.prediction_frame.columns
