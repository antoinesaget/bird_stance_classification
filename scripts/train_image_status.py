#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
from dataclasses import dataclass
import datetime as dt
from typing import Any

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
from birdsys.reporting import diff_numeric_dict, find_previous_version_dir
from birdsys.training.metrics import confusion_matrix, macro_f1
from birdsys.training.models import ImageStatusModel

STATUS_TO_ID = {
    "no_usable_birds": 0,
    "has_usable_birds": 1,
}


@dataclass(frozen=True)
class TrainCfg:
    backbone: str
    device: str
    image_size: int
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float


class ImageStatusDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image_path = pathlib.Path(str(row["filepath"]))
        with Image.open(image_path).convert("RGB") as img:
            x = self.transform(img)
        y = STATUS_TO_ID[str(row["image_status"]).strip().lower()]
        return {"image": x, "label": torch.tensor(y, dtype=torch.long)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model C image_status classifier")
    parser.add_argument("--annotation-version", required=True)
    parser.add_argument("--config", default="config/train_image_status.yaml")
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained backbone weights")
    return parser.parse_args()


def load_cfg(path: pathlib.Path) -> TrainCfg:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainCfg(
        backbone=str(raw.get("backbone", "convnext_tiny")),
        device=str(raw.get("device", "auto")),
        image_size=int(raw.get("image_size", 384)),
        batch_size=int(raw.get("batch_size", 16)),
        num_workers=int(raw.get("num_workers", 4)),
        epochs=int(raw.get("epochs", 2)),
        learning_rate=float(raw.get("learning_rate", 1e-4)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
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


def stable_split_bucket(site_id: str | None, image_id: str) -> int:
    return abs(hash(f"{site_id or ''}:{image_id}")) % 100


def prepare_frame(data_root: pathlib.Path, annotation_version: str) -> pd.DataFrame:
    images_labels = pd.read_parquet(data_root / "labelstudio" / "normalized" / annotation_version / "images_labels.parquet")
    metadata = pd.read_parquet(data_root / "metadata" / "images.parquet")

    frame = images_labels.merge(metadata[["image_id", "filepath", "site_id"]], on="image_id", how="left", validate="one_to_one")
    frame = frame[frame["image_status"].isin(["has_usable_birds", "no_usable_birds"])].copy()
    frame = frame[frame["filepath"].map(lambda p: pathlib.Path(str(p)).exists())].reset_index(drop=True)
    frame["split_bucket"] = [stable_split_bucket(site, image_id) for site, image_id in zip(frame["site_id"], frame["image_id"])]
    frame["split"] = np.where(frame["split_bucket"] < 80, "train", np.where(frame["split_bucket"] < 90, "val", "test"))
    return frame


def run_epoch(model, loader, device, optimizer=None):
    criterion = nn.CrossEntropyLoss()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    steps = 0
    all_true = []
    all_pred = []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        steps += 1
        preds = torch.argmax(logits, dim=1)
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1 = macro_f1(y_true, y_pred, num_classes=2) if len(y_true) else 0.0
    cm = confusion_matrix(y_true, y_pred, num_classes=2) if len(y_true) else np.zeros((2, 2), dtype=np.int64)

    return {
        "loss": total_loss / max(steps, 1),
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
    }


def write_training_report(
    *,
    out_dir: pathlib.Path,
    annotation_version: str,
    device: torch.device,
    smoke: bool,
    pretrained: bool,
    history: list[dict[str, float]],
    test_metrics: dict[str, Any],
    counts: dict[str, Any],
    previous_metrics: dict[str, Any] | None,
) -> tuple[pathlib.Path, pathlib.Path]:
    current_summary = {
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["f1"]),
    }
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": out_dir.name,
        "annotation_version": annotation_version,
        "device": str(device),
        "smoke": bool(smoke),
        "pretrained": bool(pretrained),
        "history": history,
        "test_summary": current_summary,
        "counts": counts,
        "comparison_to_previous": None,
    }
    if previous_metrics is not None:
        prev_test = previous_metrics.get("test", {})
        prev_summary = {
            "test_loss": float(prev_test.get("loss", 0.0)),
            "test_accuracy": float(prev_test.get("accuracy", 0.0)),
            "test_f1": float(prev_test.get("f1", 0.0)),
        }
        report["comparison_to_previous"] = {
            "previous_model_version": previous_metrics.get("model_version"),
            "delta_test_metrics": diff_numeric_dict(current_summary, prev_summary),
        }

    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Model C Report: {out_dir.name}",
        "",
        f"- Annotation version: `{annotation_version}`",
        f"- Device: `{device}`",
        f"- Smoke run: `{smoke}`",
        f"- Pretrained backbone: `{pretrained}`",
        f"- Test summary: `{current_summary}`",
        f"- Counts: `{counts}`",
    ]
    if report["comparison_to_previous"] is not None:
        cmp = report["comparison_to_previous"]
        lines.append(
            f"- Delta vs `{cmp['previous_model_version']}`: `{cmp['delta_test_metrics']}`"
        )

    report_md = out_dir / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_json, report_md


def main() -> int:
    args = parse_args()
    cfg = load_cfg(pathlib.Path(args.config).resolve())

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    frame = prepare_frame(data_root, args.annotation_version)
    if frame.empty:
        raise RuntimeError("No rows available for image_status training")

    if args.smoke:
        frame = frame.groupby("split", group_keys=False).head(64).reset_index(drop=True)

    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()

    if train_df.empty or val_df.empty:
        raise RuntimeError("train/val splits are empty")

    layout = ensure_layout(data_root)
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    else:
        out_dir = next_version_dir(layout.models_image_status, "status")

    effective_epochs = 1 if args.smoke else cfg.epochs
    effective_workers = 0 if args.smoke else cfg.num_workers
    effective_batch_size = min(cfg.batch_size, 8) if args.smoke else cfg.batch_size
    image_size = min(cfg.image_size, 224) if args.smoke else cfg.image_size

    device = pick_device(cfg.device)

    train_loader = DataLoader(
        ImageStatusDataset(train_df, image_size=image_size),
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=(device.type != "cpu"),
    )
    val_loader = DataLoader(
        ImageStatusDataset(val_df, image_size=image_size),
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=(device.type != "cpu"),
    )
    test_loader = DataLoader(
        ImageStatusDataset(test_df, image_size=image_size),
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=(device.type != "cpu"),
    )

    model = ImageStatusModel(backbone_name=cfg.backbone, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(effective_epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }
        )

    test_metrics = run_epoch(model, test_loader, device, optimizer=None)

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": cfg.backbone,
            "image_size": image_size,
            "pretrained": not args.no_pretrained,
            "label_map": STATUS_TO_ID,
        },
        checkpoint_path,
    )

    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                **cfg.__dict__,
                "device_resolved": str(device),
                "smoke": bool(args.smoke),
                "pretrained": not args.no_pretrained,
                "annotation_version": args.annotation_version,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    metrics_payload = {
        "history": history,
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "f1": test_metrics["f1"],
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        },
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "class_balance": {
                k: int(v)
                for k, v in frame["image_status"].value_counts().sort_index().to_dict().items()
            },
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    previous_metrics: dict[str, Any] | None = None
    prev_dir = find_previous_version_dir(layout.models_image_status, "status", out_dir.name)
    if prev_dir is not None:
        prev_metrics_path = prev_dir / "metrics.json"
        if prev_metrics_path.exists():
            previous_metrics = json.loads(prev_metrics_path.read_text(encoding="utf-8"))
            previous_metrics["model_version"] = prev_dir.name

    report_json, report_md = write_training_report(
        out_dir=out_dir,
        annotation_version=args.annotation_version,
        device=device,
        smoke=bool(args.smoke),
        pretrained=not args.no_pretrained,
        history=history,
        test_metrics=test_metrics,
        counts=metrics_payload["counts"],
        previous_metrics=previous_metrics,
    )

    print(f"output_dir={out_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={out_dir / 'metrics.json'}")
    print(f"report_json={report_json}")
    print(f"report_md={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
