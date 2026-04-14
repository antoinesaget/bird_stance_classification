from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

import birdsys.ml_experiments.train_attributes as train_mod


class TinyAttributeModel(nn.Module):
    def __init__(self, backbone_name: str = "tiny", pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(8 * 8 * 3, 16), nn.ReLU())
        self.heads = list(train_mod.HEADS)
        for head in self.heads:
            setattr(self, head, nn.Linear(16, len(train_mod.ID_TO_LABELS[head])))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {head: getattr(self, head)(feat) for head in self.heads}

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True


def _write_image(path: Path, *, color: str) -> None:
    Image.new("RGB", (8, 8), color=color).save(path)


def _frame(tmp_path: Path) -> pd.DataFrame:
    rows = []
    colors = ["white", "gray", "blue", "green"]
    for idx, color in enumerate(colors):
        image_path = tmp_path / f"img_{idx}.jpg"
        _write_image(image_path, color=color)
        rows.append(
            {
                "crop_path": str(image_path),
                "group_id": f"group-{idx}",
                "image_id": f"image-{idx}",
                "isbird": "yes",
                "readability": "readable",
                "specie": "correct",
                "behavior": "resting" if idx % 2 == 0 else "foraging",
                "substrate": "water",
                "stance": "bipedal",
            }
        )
    return pd.DataFrame(rows)


def test_train_model_records_batch_and_epoch_debug_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(train_mod, "MultiHeadAttributeModel", TinyAttributeModel)

    frame = _frame(tmp_path)
    cfg = train_mod.TrainCfg(
        backbone="tiny",
        image_size=8,
        batch_size=2,
        num_workers=0,
        head_epochs=1,
        finetune_epochs=1,
        head_lr=1e-3,
        finetune_lr=1e-3,
        weight_decay=0.0,
        device="cpu",
    )

    result = train_mod.train_model(
        train_df=frame,
        eval_df=frame.iloc[:2].copy(),
        cfg=cfg,
        smoke=False,
        pretrained=False,
        device=torch.device("cpu"),
        progress_every_batches=1,
        weighted_sampling=False,
    )

    assert len(result.batch_history) >= 2
    assert len(result.epoch_history) == 2
    first_batch = result.batch_history[0]
    assert "global_grad_norm" in first_batch
    assert "readability_loss" in first_batch
    first_epoch = result.epoch_history[0]
    assert "train_primary_score" in first_epoch
    assert "eval_primary_score" in first_epoch
