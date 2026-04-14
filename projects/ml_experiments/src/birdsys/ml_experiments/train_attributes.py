#!/usr/bin/env python3
"""Purpose: Train a final Model B artifact on a chosen dataset split."""
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
    compute_head_masks,
    encode_labels,
    normalize_behavior,
    normalize_choice,
    normalize_stance,
    normalize_substrate,
)
from birdsys.core.models import MultiHeadAttributeModel

from .common import (
    CONFUSION_HEADS,
    HEADS,
    ID_TO_LABELS,
    LABEL_MAPS,
    collect_visible_label_counts,
    summarize_dataset_labels,
)
from .metrics import MultiHeadMetrics, compute_multihead_metrics
from .model_b_evaluation import evaluate_checkpoint_on_frame, evaluation_result_to_dict
from .reports import write_evaluation_report, write_training_debug_artifacts


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


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
class LoaderEvaluation:
    loss: float
    head_losses: dict[str, float]
    metrics: MultiHeadMetrics


@dataclass(frozen=True)
class TrainResult:
    model: MultiHeadAttributeModel
    epoch_history: list[dict[str, Any]]
    batch_history: list[dict[str, Any]]
    final_train_evaluation: LoaderEvaluation
    final_eval_evaluation: LoaderEvaluation
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
    parser.add_argument("--dataset-dir", required=True, help="Path to ds_vXXX with train_pool/test/all_data parquet files")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "train_attributes.yaml"))
    parser.add_argument("--data-home", default=os.getenv("BIRD_DATA_HOME", "/data/birds"))
    parser.add_argument("--species-slug", default=os.getenv("BIRD_SPECIES_SLUG", "black_winged_stilt"))
    parser.add_argument("--output-dir", default="", help="Optional explicit output dir")
    parser.add_argument("--train-split", default="train_pool", choices=["train_pool", "test", "all_data"])
    parser.add_argument("--eval-split", default="none", choices=["train_pool", "test", "all_data", "none"])
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
    return summarize_dataset_labels(frame)


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


def make_loader(
    dataset: BirdAttributeDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    weights: np.ndarray | None,
    shuffle: bool,
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
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compute_head_losses(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    for head in HEADS:
        labels = batch[f"{head}_label"]
        mask = batch[f"{head}_mask"].bool()
        if mask.any():
            losses[head] = criterion(logits[head][mask], labels[mask])
        else:
            losses[head] = torch.tensor(0.0, device=next(iter(logits.values())).device)
    return losses


def _to_float_loss_dict(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {head: float(value.detach().cpu()) for head, value in losses.items()}


def _parameter_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return float(math.sqrt(total)) if total > 0 else 0.0


def evaluate_model(
    *,
    model: MultiHeadAttributeModel,
    loader: DataLoader,
    device: torch.device,
) -> LoaderEvaluation:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_head_losses = {head: 0.0 for head in HEADS}
    steps = 0
    storage: dict[str, dict[str, list[int]]] = {head: {"true": [], "pred": []} for head in HEADS}

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            logits = model(x)
            head_losses = compute_head_losses(logits, tensor_batch, criterion)
            loss = sum(head_losses.values(), torch.tensor(0.0, device=device))
            total_loss += float(loss.detach().cpu())
            for head, value in _to_float_loss_dict(head_losses).items():
                total_head_losses[head] += value
            steps += 1

            for head in storage:
                labels = batch[f"{head}_label"].cpu().numpy()
                mask = batch[f"{head}_mask"].cpu().numpy().astype(bool)
                preds = torch.argmax(logits[head], dim=1).cpu().numpy()
                storage[head]["true"].extend(labels[mask].tolist())
                storage[head]["pred"].extend(preds[mask].tolist())

    metrics = compute_multihead_metrics(
        storage=storage,
        id_to_labels=ID_TO_LABELS,
        heads=HEADS,
        confusion_heads=CONFUSION_HEADS,
    )
    average_head_losses = {
        head: (float(total_head_losses[head]) / max(steps, 1))
        for head in HEADS
    }
    return LoaderEvaluation(
        loss=float(total_loss / max(steps, 1)),
        head_losses=average_head_losses,
        metrics=metrics,
    )


def run_training_epoch(
    *,
    model: MultiHeadAttributeModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    phase_name: str,
    phase_epoch: int,
    total_phase_epochs: int,
    epoch_index: int,
    progress_every_batches: int,
    global_step_start: int,
) -> tuple[float, dict[str, float], list[dict[str, Any]], int]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_head_losses = {head: 0.0 for head in HEADS}
    batch_logs: list[dict[str, Any]] = []
    model.train(True)
    epoch_started = time.monotonic()
    total_batches = max(1, len(loader))

    global_step = int(global_step_start)
    for batch_idx, batch in enumerate(loader, start=1):
        global_step += 1
        x = batch["image"].to(device)
        tensor_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        logits = model(x)
        head_losses = compute_head_losses(logits, tensor_batch, criterion)
        loss = sum(head_losses.values(), torch.tensor(0.0, device=device))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        global_grad_norm = _parameter_grad_norm(model.parameters())
        backbone_grad_norm = _parameter_grad_norm(model.backbone.parameters())
        head_grad_norms = {
            head: _parameter_grad_norm(getattr(model, head).parameters())
            for head in model.heads
        }
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        float_head_losses = _to_float_loss_dict(head_losses)
        for head, value in float_head_losses.items():
            total_head_losses[head] += value

        elapsed = max(1e-6, time.monotonic() - epoch_started)
        batches_per_s = batch_idx / elapsed
        eta_s = (total_batches - batch_idx) / batches_per_s if batches_per_s > 0 else 0.0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        log_row: dict[str, Any] = {
            "phase": phase_name,
            "phase_epoch": int(phase_epoch),
            "total_phase_epochs": int(total_phase_epochs),
            "epoch_index": int(epoch_index),
            "global_step": int(global_step),
            "batch": int(batch_idx),
            "total_batches": int(total_batches),
            "total_loss": float(loss.detach().cpu()),
            "learning_rate": learning_rate,
            "batches_per_s": float(batches_per_s),
            "eta_s": float(eta_s),
            "global_grad_norm": float(global_grad_norm),
            "backbone_grad_norm": float(backbone_grad_norm),
        }
        for head in HEADS:
            log_row[f"{head}_loss"] = float(float_head_losses.get(head, 0.0))
            log_row[f"{head}_grad_norm"] = float(head_grad_norms.get(head, 0.0))
        batch_logs.append(log_row)

        should_log = (
            progress_every_batches > 0
            and (batch_idx % progress_every_batches == 0 or batch_idx == total_batches)
        )
        if should_log:
            avg_loss = total_loss / max(batch_idx, 1)
            print(
                "train_progress "
                f"phase={phase_name} "
                f"epoch={phase_epoch}/{total_phase_epochs} "
                f"epoch_index={epoch_index} "
                f"batch={batch_idx}/{total_batches} "
                f"avg_loss={avg_loss:.5f} "
                f"lr={learning_rate:.6g} "
                f"global_grad_norm={global_grad_norm:.5f} "
                f"backbone_grad_norm={backbone_grad_norm:.5f} "
                f"batches_per_s={batches_per_s:.2f} "
                f"eta_s={eta_s:.1f}",
                flush=True,
            )

    average_head_losses = {
        head: float(total_head_losses[head]) / max(total_batches, 1)
        for head in HEADS
    }
    return float(total_loss / max(total_batches, 1)), average_head_losses, batch_logs, global_step


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

    train_loader = make_loader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type != "cpu"),
        weights=weights,
        shuffle=weights is None,
    )
    train_eval_loader = make_loader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type != "cpu"),
        weights=None,
        shuffle=False,
    )
    eval_loader = make_loader(
        eval_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type != "cpu"),
        weights=None,
        shuffle=False,
    )

    model = MultiHeadAttributeModel(
        backbone_name=cfg.backbone,
        pretrained=pretrained,
    ).to(device)

    epoch_history: list[dict[str, Any]] = []
    batch_history: list[dict[str, Any]] = []
    global_step = 0
    epoch_index = 0

    phase_specs: list[tuple[str, int, torch.optim.Optimizer, bool]] = []
    model.freeze_backbone()
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    phase_specs.append(("head", cfg.head_epochs, opt1, False))

    model.unfreeze_backbone()
    opt2 = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
    phase_specs.append(("finetune", cfg.finetune_epochs, opt2, True))

    final_train_evaluation = LoaderEvaluation(0.0, {head: 0.0 for head in HEADS}, compute_multihead_metrics(storage={head: {"true": [], "pred": []} for head in HEADS}, id_to_labels=ID_TO_LABELS, heads=HEADS, confusion_heads=CONFUSION_HEADS))
    final_eval_evaluation = final_train_evaluation

    for phase_name, total_phase_epochs, optimizer, unfreeze in phase_specs:
        if total_phase_epochs <= 0:
            continue
        if unfreeze:
            model.unfreeze_backbone()
        else:
            model.freeze_backbone()

        for phase_epoch in range(1, total_phase_epochs + 1):
            epoch_index += 1
            epoch_started = time.monotonic()
            train_loss, train_head_losses, epoch_batch_logs, global_step = run_training_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                phase_name=phase_name,
                phase_epoch=phase_epoch,
                total_phase_epochs=total_phase_epochs,
                epoch_index=epoch_index,
                progress_every_batches=progress_every_batches,
                global_step_start=global_step,
            )
            batch_history.extend(epoch_batch_logs)
            final_train_evaluation = evaluate_model(model=model, loader=train_eval_loader, device=device)
            final_eval_evaluation = evaluate_model(model=model, loader=eval_loader, device=device)
            row: dict[str, Any] = {
                "phase": phase_name,
                "phase_epoch": int(phase_epoch),
                "epoch_index": int(epoch_index),
                "train_loss": float(train_loss),
                "eval_loss": float(final_eval_evaluation.loss),
                "train_primary_score": float(final_train_evaluation.metrics.aggregate_metrics["primary_score"]),
                "eval_primary_score": float(final_eval_evaluation.metrics.aggregate_metrics["primary_score"]),
                "duration_s": float(time.monotonic() - epoch_started),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            row.update({f"train_{head}_loss": float(value) for head, value in train_head_losses.items()})
            row.update({f"eval_{head}_loss": float(value) for head, value in final_eval_evaluation.head_losses.items()})
            row.update({f"train_{key}": float(value) for key, value in final_train_evaluation.metrics.aggregate_metrics.items()})
            row.update({f"eval_{key}": float(value) for key, value in final_eval_evaluation.metrics.aggregate_metrics.items()})
            row.update({f"train_{key}": float(value) for key, value in final_train_evaluation.metrics.summary_metrics.items()})
            row.update({f"eval_{key}": float(value) for key, value in final_eval_evaluation.metrics.summary_metrics.items()})
            epoch_history.append(row)

            print(
                "epoch_done "
                f"phase={phase_name} "
                f"epoch={phase_epoch}/{total_phase_epochs} "
                f"epoch_index={epoch_index} "
                f"train_loss={train_loss:.5f} "
                f"eval_loss={final_eval_evaluation.loss:.5f} "
                f"train_primary_score={final_train_evaluation.metrics.aggregate_metrics['primary_score']:.5f} "
                f"eval_primary_score={final_eval_evaluation.metrics.aggregate_metrics['primary_score']:.5f} "
                f"duration_s={time.monotonic() - epoch_started:.2f}",
                flush=True,
            )

    return TrainResult(
        model=model,
        epoch_history=epoch_history,
        batch_history=batch_history,
        final_train_evaluation=final_train_evaluation,
        final_eval_evaluation=final_eval_evaluation,
        train_visible_label_counts=train_visible_label_counts,
        eval_visible_label_counts=eval_visible_label_counts,
        supported_labels=supported_labels_from_counts(train_visible_label_counts),
    )


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


def _build_legacy_metrics_payload(
    *,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    result: TrainResult,
    primary_result,
) -> dict[str, Any]:
    return {
        "history": result.epoch_history,
        "final": primary_result.summary_metrics,
        "aggregate_final": primary_result.aggregate_metrics,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "supported_labels": result.supported_labels,
        "train_visible_label_counts": result.train_visible_label_counts,
        "eval_visible_label_counts": result.eval_visible_label_counts,
    }


def _write_training_report(
    *,
    out_dir: pathlib.Path,
    dataset_dir: pathlib.Path,
    train_split: str,
    eval_split: str,
    device: torch.device,
    smoke: bool,
    pretrained: bool,
    weighted_sampling: bool,
    epoch_history: list[dict[str, Any]],
    primary_result,
    train_result,
    debug_outputs: dict[str, pathlib.Path],
    previous_summary: dict[str, Any] | None,
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
        "supported_labels": train_result.supported_labels,
        "epoch_history": epoch_history,
        "primary_result": evaluation_result_to_dict(primary_result),
        "train_loader_primary_score": float(train_result.final_train_evaluation.metrics.aggregate_metrics["primary_score"]),
        "debug_artifacts": {key: str(path) for key, path in debug_outputs.items()},
        "comparison_to_previous": None,
    }

    if previous_summary is not None:
        previous_primary = previous_summary.get("summary_metrics", {})
        previous_aggregate = previous_summary.get("aggregate_metrics", {})
        report["comparison_to_previous"] = {
            "delta_summary_metrics": diff_numeric_dict(primary_result.summary_metrics, previous_primary),
            "delta_aggregate_metrics": diff_numeric_dict(primary_result.aggregate_metrics, previous_aggregate),
        }

    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Model B Training Report: {out_dir.name}",
        "",
        f"- Dataset: `{dataset_dir.name}`",
        f"- Train split: `{train_split}`",
        f"- Eval split: `{eval_split}`",
        f"- Device: `{device}`",
        f"- Smoke run: `{smoke}`",
        f"- Pretrained backbone: `{pretrained}`",
        f"- Weighted sampling: `{weighted_sampling}`",
        f"- Primary score: `{primary_result.aggregate_metrics['primary_score']:.5f}`",
        f"- Mean balanced accuracy: `{primary_result.aggregate_metrics['mean_balanced_accuracy']:.5f}`",
        f"- Summary metrics file: `{(out_dir / 'summary.json').name}`",
        f"- Training debug history: `{debug_outputs['epoch_history_csv'].name}`",
    ]
    if report["comparison_to_previous"] is not None:
        lines.append(f"- Delta vs previous aggregate metrics: `{report['comparison_to_previous']['delta_aggregate_metrics']}`")
    report_md = out_dir / "report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_json, report_md


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

    data_home = pathlib.Path(args.data_home).expanduser().resolve()
    layout = ensure_layout(data_home, args.species_slug)
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

    primary_eval_frame = eval_df if args.eval_split != "none" else train_df
    primary_result = evaluate_checkpoint_on_frame(
        checkpoint_path=checkpoint_path,
        frame=primary_eval_frame,
        device=str(device),
    )
    train_report_result = evaluate_checkpoint_on_frame(
        checkpoint_path=checkpoint_path,
        frame=train_df,
        device=str(device),
    )

    root_eval_outputs = write_evaluation_report(out_dir, primary_result)
    if args.eval_split != "none":
        write_evaluation_report(out_dir / "train_eval", train_report_result)

    debug_outputs = write_training_debug_artifacts(
        out_dir / "training_debug",
        epoch_history=result.epoch_history,
        batch_history=result.batch_history,
    )

    metrics_out = out_dir / "metrics.json"
    metrics_out.write_text(
        json.dumps(
            _build_legacy_metrics_payload(
                args=args,
                train_df=train_df,
                eval_df=eval_df,
                result=result,
                primary_result=primary_result,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    previous_summary: dict[str, Any] | None = None
    prev_dir = find_previous_version_dir(layout.models_attributes, "convnextv2s", out_dir.name)
    if prev_dir is not None:
        prev_summary_path = prev_dir / "summary.json"
        if prev_summary_path.exists():
            previous_summary = json.loads(prev_summary_path.read_text(encoding="utf-8"))

    report_json, report_md = _write_training_report(
        out_dir=out_dir,
        dataset_dir=dataset_dir,
        train_split=args.train_split,
        eval_split=args.eval_split,
        device=device,
        smoke=bool(args.smoke),
        pretrained=not args.no_pretrained,
        weighted_sampling=not args.no_weighted_sampling,
        epoch_history=result.epoch_history,
        primary_result=primary_result,
        train_result=result,
        debug_outputs=debug_outputs,
        previous_summary=previous_summary,
    )

    print(f"output_dir={out_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"summary_json={root_eval_outputs['summary_json']}")
    print(f"summary_csv={root_eval_outputs['summary_csv']}")
    print(f"metrics={metrics_out}")
    for head in primary_result.confusion_matrices:
        print(f"{head}_confusion_matrix={out_dir / f'{head}_confusion_matrix.csv'}")
    print(f"report_json={report_json}")
    print(f"report_md={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
