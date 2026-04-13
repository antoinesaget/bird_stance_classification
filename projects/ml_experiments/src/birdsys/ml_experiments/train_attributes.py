#!/usr/bin/env python3
"""Purpose: Train a final Model B artifact on a chosen dataset split"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from birdsys.core import diff_numeric_dict, ensure_layout, find_previous_version_dir, next_version_dir
from birdsys.core.attributes import (
    BEHAVIOR_TO_ID,
    READABILITY_TO_ID,
    SPECIE_TO_ID,
    STANCE_TO_ID,
    SUBSTRATE_TO_ID,
    compute_head_masks,
    encode_labels,
    normalize_behavior,
    normalize_choice,
    normalize_stance,
    normalize_substrate,
)
from birdsys.core.models import MultiHeadAttributeModel
from birdsys.ml_experiments.metrics import class_supports, confusion_matrix, macro_f1


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
HEADS = ["readability", "specie", "behavior", "substrate", "stance"]
CONFUSION_HEADS = ["behavior", "substrate", "stance"]
LABEL_MAPS = {
    "readability": READABILITY_TO_ID,
    "specie": SPECIE_TO_ID,
    "behavior": BEHAVIOR_TO_ID,
    "substrate": SUBSTRATE_TO_ID,
    "stance": STANCE_TO_ID,
}
ID_TO_LABELS = {
    head: [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
    for head, mapping in LABEL_MAPS.items()
}
HEAD_CLASS_COUNTS = {head: len(labels) for head, labels in ID_TO_LABELS.items()}


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


@dataclass(frozen=True)
class TrainResult:
    model: MultiHeadAttributeModel
    history: list[dict[str, float]]
    final_metrics: dict[str, float]
    confusion_matrices: dict[str, np.ndarray]
    train_visible_label_counts: dict[str, dict[str, int]]
    eval_visible_label_counts: dict[str, dict[str, int]]
    supported_labels: dict[str, list[str]]


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

        def _text(column: str) -> str:
            value = row.get(column)
            if value is None:
                return ""
            if isinstance(value, float) and np.isnan(value):
                return ""
            return str(value)

        labels = encode_labels(
            readability=_text("readability"),
            specie=_text("specie"),
            behavior=_text("behavior"),
            substrate=_text("substrate"),
            stance=_text("stance"),
        )
        masks = compute_head_masks(
            isbird=_text("isbird"),
            readability=_text("readability"),
            specie=_text("specie"),
            behavior=_text("behavior"),
            substrate=_text("substrate"),
        )

        return {
            "image": x,
            "readability_label": torch.tensor(labels["readability"] if labels["readability"] is not None else -1),
            "specie_label": torch.tensor(labels["specie"] if labels["specie"] is not None else -1),
            "behavior_label": torch.tensor(labels["behavior"] if labels["behavior"] is not None else -1),
            "substrate_label": torch.tensor(labels["substrate"] if labels["substrate"] is not None else -1),
            "stance_label": torch.tensor(labels["stance"] if labels["stance"] is not None else -1),
            "readability_mask": torch.tensor(masks.readability),
            "specie_mask": torch.tensor(masks.specie),
            "behavior_mask": torch.tensor(masks.behavior),
            "substrate_mask": torch.tensor(masks.substrate),
            "stance_mask": torch.tensor(masks.stance),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model B multi-head bird attribute classifier")
    parser.add_argument("--dataset-dir", required=True, help="Path to ds_vXXX with split parquet files")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "train_attributes.yaml"))
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--output-dir", default="", help="Optional explicit output dir")
    parser.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--eval-split", default="val", choices=["train", "val", "test", "none"])
    parser.add_argument("--schema-version", default="annotation_schema_v2")
    parser.add_argument("--smoke", action="store_true", help="Use tiny subset and 1 epoch")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained backbone weights")
    parser.add_argument("--no-weighted-sampling", action="store_true", help="Disable WeightedRandomSampler")
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=20,
        help="Emit running training metrics every N batches",
    )
    return parser.parse_args(argv)


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
    required = ["crop_path", "group_id", "image_id", "isbird", "readability", "specie", "behavior", "substrate", "stance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    out = df.copy()
    out["crop_path"] = out["crop_path"].astype(str)
    out = out[out["crop_path"].map(lambda p: pathlib.Path(p).exists())].reset_index(drop=True)
    out["isbird"] = out["isbird"].map(lambda value: normalize_choice(None if pd.isna(value) else str(value)))
    out = out[out["isbird"] == "yes"].reset_index(drop=True)
    for column, normalizer in (
        ("behavior", normalize_behavior),
        ("substrate", normalize_substrate),
        ("stance", normalize_stance),
    ):
        out[column] = out[column].map(lambda value: normalizer(None if pd.isna(value) else str(value)))
    for column in ("readability", "specie"):
        out[column] = out[column].map(lambda value: normalize_choice(None if pd.isna(value) else str(value)))
    return out


def dataframe_from_split(path: pathlib.Path, smoke: bool) -> pd.DataFrame:
    frame = sanitize_df(pd.read_parquet(path))
    if smoke:
        return frame.head(min(len(frame), 128)).reset_index(drop=True)
    return frame


def summarize_labels(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for column in ["isbird", "readability", "specie", "behavior", "substrate", "stance"]:
        vc = frame[column].fillna("<null>").value_counts().sort_index().to_dict()
        out[column] = {str(k): int(v) for k, v in vc.items()}
    return out


def collect_visible_label_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts = {head: {label: 0 for label in ID_TO_LABELS[head]} for head in HEADS}
    for _, row in frame.iterrows():
        masks = compute_head_masks(
            isbird=str(row["isbird"]),
            readability=str(row["readability"]),
            specie=str(row["specie"]),
            behavior=str(row["behavior"]) if pd.notna(row["behavior"]) else None,
            substrate=str(row["substrate"]) if pd.notna(row["substrate"]) else None,
        )
        normalized = {
            "readability": normalize_choice(str(row["readability"])) if pd.notna(row["readability"]) else None,
            "specie": normalize_choice(str(row["specie"])) if pd.notna(row["specie"]) else None,
            "behavior": normalize_behavior(str(row["behavior"])) if pd.notna(row["behavior"]) else None,
            "substrate": normalize_substrate(str(row["substrate"])) if pd.notna(row["substrate"]) else None,
            "stance": normalize_stance(str(row["stance"])) if pd.notna(row["stance"]) else None,
        }
        mask_map = {
            "readability": masks.readability,
            "specie": masks.specie,
            "behavior": masks.behavior,
            "substrate": masks.substrate,
            "stance": masks.stance,
        }
        for head in HEADS:
            label = normalized[head]
            if mask_map[head] and label in counts[head]:
                counts[head][label] += 1
    return counts


def supported_labels_from_counts(counts: dict[str, dict[str, int]]) -> dict[str, list[str]]:
    return {
        head: sorted([label for label, count in label_counts.items() if int(count) > 0], key=lambda label: LABEL_MAPS[head][label])
        for head, label_counts in counts.items()
    }


def compute_sampling_weights(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    counts = collect_visible_label_counts(frame)
    totals = {head: sum(head_counts.values()) for head, head_counts in counts.items()}

    raw_weights = []
    for _, row in frame.iterrows():
        masks = compute_head_masks(
            isbird=str(row["isbird"]),
            readability=str(row["readability"]),
            specie=str(row["specie"]),
            behavior=str(row["behavior"]) if pd.notna(row["behavior"]) else None,
            substrate=str(row["substrate"]) if pd.notna(row["substrate"]) else None,
        )
        candidates: list[float] = []
        items = {
            "readability": normalize_choice(str(row["readability"])) if pd.notna(row["readability"]) else None,
            "specie": normalize_choice(str(row["specie"])) if pd.notna(row["specie"]) else None,
            "behavior": normalize_behavior(str(row["behavior"])) if pd.notna(row["behavior"]) else None,
            "substrate": normalize_substrate(str(row["substrate"])) if pd.notna(row["substrate"]) else None,
            "stance": normalize_stance(str(row["stance"])) if pd.notna(row["stance"]) else None,
        }
        mask_map = {
            "readability": masks.readability,
            "specie": masks.specie,
            "behavior": masks.behavior,
            "substrate": masks.substrate,
            "stance": masks.stance,
        }
        for head in HEADS:
            label = items[head]
            count = counts[head].get(label or "", 0)
            if mask_map[head] and label and count > 0 and totals[head] > 0:
                candidates.append(math.sqrt(float(totals[head]) / float(count)))
        raw_weights.append(max(candidates) if candidates else 1.0)

    weights = np.asarray(raw_weights, dtype=np.float64)
    mean_weight = float(weights.mean()) if len(weights) else 1.0
    if mean_weight > 0:
        weights = weights / mean_weight
    weights = np.minimum(weights, 4.0)
    return weights, counts


def make_train_loader(
    dataset: BirdAttributeDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    weights: np.ndarray | None,
) -> DataLoader:
    if weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compute_masked_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(iter(logits.values())).device)
    for head in HEADS:
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
    phase_name: str,
    epoch_idx: int,
    total_epochs: int,
    progress_every_batches: int,
) -> float:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    steps = 0

    train_mode = optimizer is not None
    model.train(train_mode)
    epoch_started = time.monotonic()
    total_batches = max(1, len(loader))

    for batch_idx, batch in enumerate(loader, start=1):
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

        should_log = (
            progress_every_batches > 0
            and (batch_idx % progress_every_batches == 0 or batch_idx == total_batches)
        )
        if should_log:
            elapsed = max(1e-6, time.monotonic() - epoch_started)
            avg_loss = total_loss / max(steps, 1)
            batches_per_s = batch_idx / elapsed
            eta_s = (total_batches - batch_idx) / batches_per_s if batches_per_s > 0 else 0.0
            print(
                "train_progress "
                f"phase={phase_name} "
                f"epoch={epoch_idx}/{total_epochs} "
                f"batch={batch_idx}/{total_batches} "
                f"avg_loss={avg_loss:.5f} "
                f"batches_per_s={batches_per_s:.2f} "
                f"eta_s={eta_s:.1f}",
                flush=True,
            )

    return total_loss / max(steps, 1)


def evaluate(
    *,
    model: MultiHeadAttributeModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, dict[str, int]]]:
    model.eval()
    storage: dict[str, dict[str, list[int]]] = {head: {"true": [], "pred": []} for head in HEADS}

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
    confusion_matrices = {head: np.zeros((HEAD_CLASS_COUNTS[head], HEAD_CLASS_COUNTS[head]), dtype=np.int64) for head in CONFUSION_HEADS}
    support_counts: dict[str, dict[str, int]] = {}
    for head, num_classes in HEAD_CLASS_COUNTS.items():
        y_true = np.array(storage[head]["true"], dtype=np.int64)
        y_pred = np.array(storage[head]["pred"], dtype=np.int64)
        if len(y_true) == 0:
            metrics[f"{head}_accuracy"] = 0.0
            metrics[f"{head}_f1"] = 0.0
            support_counts[head] = {label: 0 for label in ID_TO_LABELS[head]}
            continue

        acc = float((y_true == y_pred).mean())
        f1 = macro_f1(y_true, y_pred, num_classes, ignore_absent_classes=True)
        metrics[f"{head}_accuracy"] = acc
        metrics[f"{head}_f1"] = f1
        support_counts[head] = {
            label: int(count)
            for label, count in zip(ID_TO_LABELS[head], class_supports(y_true, num_classes).tolist())
        }
        if head in confusion_matrices:
            confusion_matrices[head] = confusion_matrix(y_true, y_pred, num_classes)

    return metrics, confusion_matrices, support_counts


def train_model(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cfg: TrainCfg,
    smoke: bool,
    pretrained: bool,
    device: torch.device,
    progress_every_batches: int,
    weighted_sampling: bool,
) -> TrainResult:
    train_ds = BirdAttributeDataset(train_df, image_size=cfg.image_size)
    eval_ds = BirdAttributeDataset(eval_df, image_size=cfg.image_size)

    weights = None
    train_visible_label_counts = collect_visible_label_counts(train_df)
    if weighted_sampling:
        weights, train_visible_label_counts = compute_sampling_weights(train_df)

    eval_visible_label_counts = collect_visible_label_counts(eval_df)
    train_loader = make_train_loader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type != "cpu"),
        weights=weights,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type != "cpu"),
    )

    model = MultiHeadAttributeModel(
        backbone_name=cfg.backbone,
        pretrained=pretrained,
    ).to(device)

    history: list[dict[str, float]] = []

    model.freeze_backbone()
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    for epoch in range(cfg.head_epochs):
        epoch_number = epoch + 1
        epoch_started = time.monotonic()
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=opt1,
            phase_name="head",
            epoch_idx=epoch_number,
            total_epochs=cfg.head_epochs,
            progress_every_batches=progress_every_batches,
        )
        val_metrics, _, _ = evaluate(model=model, loader=eval_loader, device=device)
        history.append(
            {
                "phase": 1,
                "epoch": epoch_number,
                "train_loss": train_loss,
                **val_metrics,
            }
        )
        print(
            "epoch_done "
            f"phase=head "
            f"epoch={epoch_number}/{cfg.head_epochs} "
            f"train_loss={train_loss:.5f} "
            f"behavior_f1={val_metrics.get('behavior_f1', 0.0):.5f} "
            f"stance_f1={val_metrics.get('stance_f1', 0.0):.5f} "
            f"duration_s={time.monotonic() - epoch_started:.2f}",
            flush=True,
        )

    model.unfreeze_backbone()
    opt2 = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
    confusion_matrices = {head: np.zeros((HEAD_CLASS_COUNTS[head], HEAD_CLASS_COUNTS[head]), dtype=np.int64) for head in CONFUSION_HEADS}
    for epoch in range(cfg.finetune_epochs):
        epoch_number = epoch + 1
        epoch_started = time.monotonic()
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=opt2,
            phase_name="finetune",
            epoch_idx=epoch_number,
            total_epochs=cfg.finetune_epochs,
            progress_every_batches=progress_every_batches,
        )
        val_metrics, confusion_matrices, _ = evaluate(model=model, loader=eval_loader, device=device)
        history.append(
            {
                "phase": 2,
                "epoch": epoch_number,
                "train_loss": train_loss,
                **val_metrics,
            }
        )
        print(
            "epoch_done "
            f"phase=finetune "
            f"epoch={epoch_number}/{cfg.finetune_epochs} "
            f"train_loss={train_loss:.5f} "
            f"behavior_f1={val_metrics.get('behavior_f1', 0.0):.5f} "
            f"stance_f1={val_metrics.get('stance_f1', 0.0):.5f} "
            f"duration_s={time.monotonic() - epoch_started:.2f}",
            flush=True,
        )

    final_metrics, confusion_matrices, eval_visible_label_counts = evaluate(model=model, loader=eval_loader, device=device)
    if history:
        history[-1].update(final_metrics)

    return TrainResult(
        model=model,
        history=history,
        final_metrics=final_metrics,
        confusion_matrices=confusion_matrices,
        train_visible_label_counts=train_visible_label_counts,
        eval_visible_label_counts=eval_visible_label_counts,
        supported_labels=supported_labels_from_counts(train_visible_label_counts),
    )


def write_training_report(
    *,
    out_dir: pathlib.Path,
    dataset_dir: pathlib.Path,
    train_split: str,
    eval_split: str,
    device: torch.device,
    smoke: bool,
    pretrained: bool,
    weighted_sampling: bool,
    history: list[dict[str, float]],
    final_metrics: dict[str, float],
    train_rows: int,
    eval_rows: int,
    train_label_counts: dict[str, dict[str, int]],
    eval_label_counts: dict[str, dict[str, int]],
    train_visible_label_counts: dict[str, dict[str, int]],
    eval_visible_label_counts: dict[str, dict[str, int]],
    supported_labels: dict[str, list[str]],
    previous_metrics: dict[str, Any] | None,
) -> tuple[pathlib.Path, pathlib.Path]:
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": out_dir.name,
        "dataset_version": dataset_dir.name,
        "train_split": train_split,
        "eval_split": eval_split,
        "device": str(device),
        "smoke": bool(smoke),
        "pretrained": bool(pretrained),
        "weighted_sampling": bool(weighted_sampling),
        "train_rows": int(train_rows),
        "eval_rows": int(eval_rows),
        "train_label_counts": train_label_counts,
        "eval_label_counts": eval_label_counts,
        "train_visible_label_counts": train_visible_label_counts,
        "eval_visible_label_counts": eval_visible_label_counts,
        "supported_labels": supported_labels,
        "history": history,
        "final_metrics": final_metrics,
        "comparison_to_previous": None,
    }

    if previous_metrics is not None:
        prev_final = previous_metrics.get("final", {})
        comparable_keys = [key for key in sorted(final_metrics) if key.endswith("_accuracy") or key.endswith("_f1")]
        current_slice = {key: final_metrics[key] for key in comparable_keys}
        previous_slice = {key: prev_final.get(key) for key in comparable_keys}
        report["comparison_to_previous"] = {
            "previous_model_version": previous_metrics.get("model_version"),
            "delta_final_metrics": diff_numeric_dict(current_slice, previous_slice),
        }

    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Model B Report: {out_dir.name}",
        "",
        f"- Dataset: `{dataset_dir.name}`",
        f"- Splits: train `{train_split}`, eval `{eval_split}`",
        f"- Device: `{device}`",
        f"- Smoke run: `{smoke}`",
        f"- Pretrained backbone: `{pretrained}`",
        f"- Weighted sampling: `{weighted_sampling}`",
        f"- Rows: train `{train_rows}`, eval `{eval_rows}`",
        f"- Supported labels: `{supported_labels}`",
        f"- Final metrics: `{final_metrics}`",
    ]
    if report["comparison_to_previous"] is not None:
        cmp = report["comparison_to_previous"]
        lines.append(f"- Delta vs `{cmp['previous_model_version']}`: `{cmp['delta_final_metrics']}`")
    report_md = out_dir / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_json, report_md


def save_checkpoint(
    *,
    out_dir: pathlib.Path,
    model: MultiHeadAttributeModel,
    cfg: TrainCfg,
    pretrained: bool,
    schema_version: str,
    supported_labels: dict[str, list[str]],
    train_visible_label_counts: dict[str, dict[str, int]],
) -> pathlib.Path:
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": cfg.backbone,
            "image_size": cfg.image_size,
            "pretrained": pretrained,
            "schema_version": schema_version,
            "label_maps": LABEL_MAPS,
            "supported_labels": supported_labels,
            "train_label_counts": train_visible_label_counts,
        },
        checkpoint_path,
    )
    return checkpoint_path


def write_confusion_matrices(out_dir: pathlib.Path, confusion_matrices: dict[str, np.ndarray]) -> dict[str, pathlib.Path]:
    output: dict[str, pathlib.Path] = {}
    for head, matrix in confusion_matrices.items():
        out_path = out_dir / f"{head}_confusion_matrix.csv"
        labels = ID_TO_LABELS[head]
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(out_path)
        output[head] = out_path
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_cfg(pathlib.Path(args.config).resolve())

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser().resolve()
    train_path = dataset_dir / f"{args.train_split}.parquet"
    eval_path = dataset_dir / f"{args.eval_split}.parquet" if args.eval_split != "none" else None
    train_df = dataframe_from_split(train_path, args.smoke)
    eval_df = dataframe_from_split(eval_path, args.smoke) if eval_path is not None else train_df.head(0).copy()

    if train_df.empty:
        raise RuntimeError("train split is empty after filtering missing crop files and non-bird rows")
    if args.eval_split != "none" and eval_df.empty:
        raise RuntimeError("eval split is empty after filtering missing crop files and non-bird rows")

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
    total_planned_epochs = int(effective.head_epochs + effective.finetune_epochs)
    print(
        "train_setup "
        f"dataset={dataset_dir.name} "
        f"train_split={args.train_split} "
        f"eval_split={args.eval_split} "
        f"train_rows={len(train_df)} "
        f"eval_rows={len(eval_df)} "
        f"device={device} "
        f"total_epochs={total_planned_epochs} "
        f"batch_size={effective.batch_size} "
        f"weighted_sampling={not args.no_weighted_sampling}",
        flush=True,
    )

    result = train_model(
        train_df=train_df,
        eval_df=eval_df,
        cfg=effective,
        smoke=bool(args.smoke),
        pretrained=not args.no_pretrained,
        device=device,
        progress_every_batches=args.progress_every_batches,
        weighted_sampling=not args.no_weighted_sampling,
    )

    checkpoint_path = save_checkpoint(
        out_dir=out_dir,
        model=result.model,
        cfg=effective,
        pretrained=not args.no_pretrained,
        schema_version=args.schema_version,
        supported_labels=result.supported_labels,
        train_visible_label_counts=result.train_visible_label_counts,
    )

    config_out = out_dir / "config.yaml"
    config_out.write_text(
        yaml.safe_dump(
            {
                "dataset_dir": str(dataset_dir),
                "train_split": args.train_split,
                "eval_split": args.eval_split,
                "device": str(device),
                "smoke": bool(args.smoke),
                "pretrained": not args.no_pretrained,
                "weighted_sampling": not args.no_weighted_sampling,
                "schema_version": args.schema_version,
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
                "history": result.history,
                "final": result.final_metrics,
                "train_rows": int(len(train_df)),
                "eval_rows": int(len(eval_df)),
                "train_split": args.train_split,
                "eval_split": args.eval_split,
                "supported_labels": result.supported_labels,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    previous_metrics: dict[str, Any] | None = None
    prev_dir = find_previous_version_dir(layout.models_attributes, "convnextv2s", out_dir.name)
    if prev_dir is not None:
        prev_metrics_path = prev_dir / "metrics.json"
        if prev_metrics_path.exists():
            previous_metrics = json.loads(prev_metrics_path.read_text(encoding="utf-8"))
            previous_metrics["model_version"] = prev_dir.name

    report_json, report_md = write_training_report(
        out_dir=out_dir,
        dataset_dir=dataset_dir,
        train_split=args.train_split,
        eval_split=args.eval_split,
        device=device,
        smoke=bool(args.smoke),
        pretrained=not args.no_pretrained,
        weighted_sampling=not args.no_weighted_sampling,
        history=result.history,
        final_metrics=result.final_metrics,
        train_rows=len(train_df),
        eval_rows=len(eval_df),
        train_label_counts=summarize_labels(train_df),
        eval_label_counts=summarize_labels(eval_df),
        train_visible_label_counts=result.train_visible_label_counts,
        eval_visible_label_counts=result.eval_visible_label_counts,
        supported_labels=result.supported_labels,
        previous_metrics=previous_metrics,
    )

    confusion_outputs = write_confusion_matrices(out_dir, result.confusion_matrices)

    print(f"output_dir={out_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_out}")
    for head, path in confusion_outputs.items():
        print(f"{head}_confusion_matrix={path}")
    print(f"report_json={report_json}")
    print(f"report_md={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
