#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout, next_version_dir
from birdsys.training.attributes import (
    BEHAVIOR_TO_ID,
    LEGS_TO_ID,
    READABILITY_TO_ID,
    SPECIE_TO_ID,
    SUBSTRATE_TO_ID,
    compute_head_masks,
    encode_labels,
)
from birdsys.training.metrics import confusion_matrix, macro_f1
from birdsys.training.models import MultiHeadAttributeModel


HEAD_CLASS_COUNTS = {
    "readability": 3,
    "specie": 3,
    "behavior": 7,
    "substrate": 4,
    "legs": 3,
}


@dataclass(frozen=True)
class TrainCfg:
    backbone: str
    image_size: int
    batch_size: int
    num_workers: int
    head_epochs: int
    finetune_epochs: int
    head_lr: float
    finetune_lr: float
    weight_decay: float
    device: str


class BirdAttributeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        image_path = pathlib.Path(str(row["crop_path"]))
        with Image.open(image_path).convert("RGB") as img:
            x = self.transform(img)

        labels = encode_labels(
            readability=str(row.get("readability") or ""),
            specie=str(row.get("specie") or ""),
            behavior=str(row.get("behavior") or ""),
            substrate=str(row.get("substrate") or ""),
            legs=str(row.get("legs") or ""),
        )
        masks = compute_head_masks(
            readability=str(row.get("readability") or ""),
            specie=str(row.get("specie") or ""),
            behavior=str(row.get("behavior") or ""),
            substrate=str(row.get("substrate") or ""),
        )

        return {
            "image": x,
            "readability_label": torch.tensor(labels["readability"] if labels["readability"] is not None else -1),
            "specie_label": torch.tensor(labels["specie"] if labels["specie"] is not None else -1),
            "behavior_label": torch.tensor(labels["behavior"] if labels["behavior"] is not None else -1),
            "substrate_label": torch.tensor(labels["substrate"] if labels["substrate"] is not None else -1),
            "legs_label": torch.tensor(labels["legs"] if labels["legs"] is not None else -1),
            "readability_mask": torch.tensor(masks.readability),
            "specie_mask": torch.tensor(masks.specie),
            "behavior_mask": torch.tensor(masks.behavior),
            "substrate_mask": torch.tensor(masks.substrate),
            "legs_mask": torch.tensor(masks.legs),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model B multi-head bird attribute classifier")
    parser.add_argument("--dataset-dir", required=True, help="Path to ds_vXXX with train/val/test parquet")
    parser.add_argument("--config", default="config/train_attributes.yaml")
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--output-dir", default="", help="Optional explicit output dir")
    parser.add_argument("--smoke", action="store_true", help="Use tiny subset and 1 epoch")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained backbone weights")
    return parser.parse_args()


def load_cfg(path: pathlib.Path) -> TrainCfg:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainCfg(
        backbone=str(raw.get("backbone", "convnextv2_small")),
        image_size=int(raw.get("image_size", 384)),
        batch_size=int(raw.get("batch_size", 16)),
        num_workers=int(raw.get("num_workers", 4)),
        head_epochs=int(raw.get("head_epochs", 1)),
        finetune_epochs=int(raw.get("finetune_epochs", 1)),
        head_lr=float(raw.get("head_lr", 1e-3)),
        finetune_lr=float(raw.get("finetune_lr", 1e-4)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        device=str(raw.get("device", "auto")),
    )


def pick_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "crop_path",
        "readability",
        "specie",
        "behavior",
        "substrate",
        "legs",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    out = df.copy()
    out["crop_path"] = out["crop_path"].astype(str)
    out = out[out["crop_path"].map(lambda p: pathlib.Path(p).exists())].reset_index(drop=True)
    return out


def compute_masked_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(iter(logits.values())).device)
    for head in ["readability", "specie", "behavior", "substrate", "legs"]:
        labels = batch[f"{head}_label"]
        mask = batch[f"{head}_mask"].bool()
        if mask.any():
            total = total + criterion(logits[head][mask], labels[mask])
    return total


def run_epoch(
    *,
    model: MultiHeadAttributeModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    steps = 0

    train_mode = optimizer is not None
    model.train(train_mode)

    for batch in loader:
        x = batch["image"].to(device)
        tensor_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        logits = model(x)
        loss = compute_masked_loss(logits, tensor_batch, criterion)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        steps += 1

    return total_loss / max(steps, 1)


def evaluate(
    *,
    model: MultiHeadAttributeModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    storage: dict[str, dict[str, list[int]]] = {
        head: {"true": [], "pred": []}
        for head in ["readability", "specie", "behavior", "substrate", "legs"]
    }

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            logits = model(x)

            for head in storage:
                labels = batch[f"{head}_label"].cpu().numpy()
                mask = batch[f"{head}_mask"].cpu().numpy().astype(bool)
                preds = torch.argmax(logits[head], dim=1).cpu().numpy()
                storage[head]["true"].extend(labels[mask].tolist())
                storage[head]["pred"].extend(preds[mask].tolist())

    metrics: dict[str, float] = {}
    legs_cm = np.zeros((3, 3), dtype=np.int64)
    for head, num_classes in HEAD_CLASS_COUNTS.items():
        y_true = np.array(storage[head]["true"], dtype=np.int64)
        y_pred = np.array(storage[head]["pred"], dtype=np.int64)
        if len(y_true) == 0:
            metrics[f"{head}_accuracy"] = 0.0
            metrics[f"{head}_f1"] = 0.0
            continue
        acc = float((y_true == y_pred).mean())
        f1 = macro_f1(y_true, y_pred, num_classes)
        metrics[f"{head}_accuracy"] = acc
        metrics[f"{head}_f1"] = f1

        if head == "legs":
            legs_cm = confusion_matrix(y_true, y_pred, num_classes)

    return metrics, legs_cm


def dataframe_from_split(path: pathlib.Path, smoke: bool) -> pd.DataFrame:
    frame = sanitize_df(pd.read_parquet(path))
    if smoke:
        return frame.head(min(len(frame), 128)).reset_index(drop=True)
    return frame


def main() -> int:
    args = parse_args()
    cfg = load_cfg(pathlib.Path(args.config).resolve())

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser().resolve()
    train_path = dataset_dir / "train.parquet"
    val_path = dataset_dir / "val.parquet"

    train_df = dataframe_from_split(train_path, args.smoke)
    val_df = dataframe_from_split(val_path, args.smoke)

    if train_df.empty or val_df.empty:
        raise RuntimeError("train/val splits are empty after filtering missing crop files")

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    layout = ensure_layout(data_root)
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_dir = next_version_dir(layout.models_attributes, "convnextv2s")

    effective = cfg
    if args.smoke:
        effective = TrainCfg(
            backbone=cfg.backbone,
            image_size=min(cfg.image_size, 224),
            batch_size=min(cfg.batch_size, 8),
            num_workers=0,
            head_epochs=1,
            finetune_epochs=1,
            head_lr=cfg.head_lr,
            finetune_lr=cfg.finetune_lr,
            weight_decay=cfg.weight_decay,
            device=cfg.device,
        )

    device = pick_device(effective.device)

    train_ds = BirdAttributeDataset(train_df, image_size=effective.image_size)
    val_ds = BirdAttributeDataset(val_df, image_size=effective.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=effective.batch_size,
        shuffle=True,
        num_workers=effective.num_workers,
        pin_memory=(device.type != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=effective.batch_size,
        shuffle=False,
        num_workers=effective.num_workers,
        pin_memory=(device.type != "cpu"),
    )

    model = MultiHeadAttributeModel(
        backbone_name=effective.backbone,
        pretrained=not args.no_pretrained,
    ).to(device)

    history: list[dict[str, float]] = []

    # Phase 1: train heads with frozen backbone
    model.freeze_backbone()
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=effective.head_lr,
        weight_decay=effective.weight_decay,
    )
    for epoch in range(effective.head_epochs):
        train_loss = run_epoch(model=model, loader=train_loader, device=device, optimizer=opt1)
        val_metrics, _ = evaluate(model=model, loader=val_loader, device=device)
        history.append(
            {
                "phase": 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics,
            }
        )

    # Phase 2: light fine-tuning
    model.unfreeze_backbone()
    opt2 = torch.optim.AdamW(model.parameters(), lr=effective.finetune_lr, weight_decay=effective.weight_decay)
    legs_cm = np.zeros((3, 3), dtype=np.int64)
    for epoch in range(effective.finetune_epochs):
        train_loss = run_epoch(model=model, loader=train_loader, device=device, optimizer=opt2)
        val_metrics, legs_cm = evaluate(model=model, loader=val_loader, device=device)
        history.append(
            {
                "phase": 2,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics,
            }
        )

    final_metrics = history[-1] if history else {}

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": effective.backbone,
            "image_size": effective.image_size,
            "pretrained": not args.no_pretrained,
            "label_maps": {
                "readability": READABILITY_TO_ID,
                "specie": SPECIE_TO_ID,
                "behavior": BEHAVIOR_TO_ID,
                "substrate": SUBSTRATE_TO_ID,
                "legs": LEGS_TO_ID,
            },
        },
        checkpoint_path,
    )

    config_out = out_dir / "config.yaml"
    config_out.write_text(
        yaml.safe_dump(
            {
                "dataset_dir": str(dataset_dir),
                "device": str(device),
                "smoke": bool(args.smoke),
                "pretrained": not args.no_pretrained,
                **effective.__dict__,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    metrics_out = out_dir / "metrics.json"
    metrics_out.write_text(
        json.dumps(
            {
                "history": history,
                "final": final_metrics,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    legs_cm_path = out_dir / "legs_confusion_matrix.csv"
    pd.DataFrame(legs_cm, index=["one", "two", "unsure"], columns=["one", "two", "unsure"]).to_csv(legs_cm_path)

    print(f"output_dir={out_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_out}")
    print(f"legs_confusion_matrix={legs_cm_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
