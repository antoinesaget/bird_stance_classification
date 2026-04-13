#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from common import (
    BEHAVIOR_TO_ID,
    ExperimentSpec,
    FoldContext,
    FoldOutput,
    HEADS,
    ID_TO_LABELS,
    LABEL_MAPS,
    READABILITY_TO_ID,
    SPECIE_TO_ID,
    STANCE_TO_ID,
    SUBSTRATE_TO_ID,
    collect_visible_label_counts,
    compute_head_masks,
    macro_f1,
    normalize_behavior,
    normalize_choice,
    normalize_stance,
    normalize_substrate,
    stable_hash_fraction,
    write_json,
)

BACKBONE = "convnextv2_nano.fcmae_ft_in1k"
IMAGE_SIZE = 256
BATCH_SIZE = 24
NUM_WORKERS = 4
HEAD_LR = 5.0e-4
FINETUNE_BACKBONE_LR = 3.0e-5
FINETUNE_HEAD_LR = 3.0e-4
WEIGHT_DECAY = 1.0e-4
SAMPLER_CAP = 2.5
GRAD_CLIP_NORM = 1.0
SCHEMA_VERSION = "annotation_schema_v2"
EVAL_EVERY_STEPS = 10
EVAL_EVERY_STEPS_TIGHT = 15
LATE_PHASE_EVAL_EVERY_STEPS = 4
HEAD_PHASE_RATIO = 0.45
WARMUP_PHASE_RATIO = 0.15
MIN_HEAD_SECONDS = 10.0
MIN_FINAL_SECONDS = 12.0
FINETUNE_END_LR_SCALE = 0.10
EARLY_STOP_PATIENCE = 4
MIN_EVALS_BEFORE_EARLY_STOP = 2
EARLY_STOP_MIN_DELTA = 0.003
EARLY_STOP_START_PROGRESS = 0.45
EARLY_STOP_RETENTION_FLOOR = 0.97
INNER_VAL_TARGET = 0.15
INNER_VAL_MIN = 0.10


class BirdAttributeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, *, train_mode: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        base = [transforms.Resize((image_size, image_size))]
        if train_mode:
            base.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01)],
                        p=0.40,
                    ),
                ]
            )
        base.append(transforms.ToTensor())
        self.transform = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[idx]
        image_path = Path(str(row["crop_path"]))
        with Image.open(image_path).convert("RGB") as image:
            x = self.transform(image)

        def _text(column: str) -> str:
            value = row.get(column)
            if value is None:
                return ""
            if isinstance(value, float) and np.isnan(value):
                return ""
            return str(value)

        readability = normalize_choice(_text("readability"))
        specie = normalize_choice(_text("specie"))
        behavior = normalize_behavior(_text("behavior"))
        substrate = normalize_substrate(_text("substrate"))
        stance = normalize_stance(_text("stance"))
        masks = compute_head_masks(
            isbird=_text("isbird"),
            readability=readability,
            specie=specie,
            behavior=behavior,
            substrate=substrate,
        )
        labels = {
            "readability": READABILITY_TO_ID.get(readability, -1),
            "specie": SPECIE_TO_ID.get(specie, -1),
            "behavior": BEHAVIOR_TO_ID.get(behavior, -1),
            "substrate": SUBSTRATE_TO_ID.get(substrate, -1),
            "stance": STANCE_TO_ID.get(stance, -1),
        }
        return {
            "row_id": str(row["row_id"]),
            "image": x,
            **{f"{head}_label": torch.tensor(labels[head], dtype=torch.long) for head in HEADS},
            **{
                "readability_mask": torch.tensor(masks.readability),
                "specie_mask": torch.tensor(masks.specie),
                "behavior_mask": torch.tensor(masks.behavior),
                "substrate_mask": torch.tensor(masks.substrate),
                "stance_mask": torch.tensor(masks.stance),
            },
        }


class MultiHeadAttributeModel(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = int(getattr(self.backbone, "num_features", 768))
        self.readability = nn.Linear(feat_dim, len(ID_TO_LABELS["readability"]))
        self.specie = nn.Linear(feat_dim, len(ID_TO_LABELS["specie"]))
        self.behavior = nn.Linear(feat_dim, len(ID_TO_LABELS["behavior"]))
        self.substrate = nn.Linear(feat_dim, len(ID_TO_LABELS["substrate"]))
        self.stance = nn.Linear(feat_dim, len(ID_TO_LABELS["stance"]))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "readability": self.readability(features),
            "specie": self.specie(features),
            "behavior": self.behavior(features),
            "substrate": self.substrate(features),
            "stance": self.stance(features),
        }

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        name="baseline_convnext_project7_stable_v1",
        model_family="multihead_timm_backbone",
        description="ConvNeXtV2 nano baseline with inner-val selection, warmup, clipping, and reduced augments",
        supports_export_now=True,
        seed_policy="fixed",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_inner_validation_split(
    frame: pd.DataFrame,
    *,
    group_column: str = "group_id",
    preferred_val_fraction: float = INNER_VAL_TARGET,
    minimum_val_fraction: float = INNER_VAL_MIN,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    ranked_groups = sorted(frame[group_column].astype(str).unique().tolist(), key=stable_hash_fraction)
    total_visible_stance = int(sum(collect_visible_label_counts(frame)["stance"].values()))
    min_stance_rows = 1 if total_visible_stance < 10 else min(3, total_visible_stance)
    candidate_fractions = [preferred_val_fraction, 0.14, 0.13, 0.12, 0.11, minimum_val_fraction]

    chosen_train = frame
    chosen_val = frame.iloc[0:0].copy()
    chosen_fraction = minimum_val_fraction
    for fraction in candidate_fractions:
        val_group_count = max(1, round(len(ranked_groups) * fraction))
        val_groups = set(ranked_groups[:val_group_count])
        val_df = frame[frame[group_column].astype(str).isin(val_groups)].reset_index(drop=True)
        train_df = frame[~frame[group_column].astype(str).isin(val_groups)].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            continue
        chosen_train = train_df
        chosen_val = val_df
        chosen_fraction = fraction
        visible_stance = int(sum(collect_visible_label_counts(val_df)["stance"].values()))
        if visible_stance >= min_stance_rows:
            break
    return chosen_train, chosen_val, float(chosen_fraction)


def resolve_inner_frames(train_parquet: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    train_path = Path(train_parquet)
    train_df = pd.read_parquet(train_path)
    if train_path.name.endswith("_inner_train.parquet"):
        val_path = train_path.with_name(train_path.name.replace("_inner_train.parquet", "_inner_val.parquet"))
        if not val_path.exists():
            raise FileNotFoundError(val_path)
        return train_df, pd.read_parquet(val_path), INNER_VAL_TARGET
    inner_train_df, inner_val_df, inner_val_fraction = build_inner_validation_split(train_df)
    return inner_train_df, inner_val_df, inner_val_fraction


def compute_sampling_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = collect_visible_label_counts(frame)
    totals = {head: sum(head_counts.values()) for head, head_counts in counts.items()}
    raw_weights: list[float] = []
    for _, row in frame.iterrows():
        masks = compute_head_masks(
            isbird=str(row["isbird"]),
            readability=str(row["readability"]),
            specie=str(row["specie"]),
            behavior=str(row["behavior"]) if pd.notna(row["behavior"]) else None,
            substrate=str(row["substrate"]) if pd.notna(row["substrate"]) else None,
        )
        mask_map = {
            "readability": masks.readability,
            "specie": masks.specie,
            "behavior": masks.behavior,
            "substrate": masks.substrate,
            "stance": masks.stance,
        }
        values = {
            "readability": normalize_choice(str(row["readability"])) if pd.notna(row["readability"]) else None,
            "specie": normalize_choice(str(row["specie"])) if pd.notna(row["specie"]) else None,
            "behavior": normalize_behavior(str(row["behavior"])) if pd.notna(row["behavior"]) else None,
            "substrate": normalize_substrate(str(row["substrate"])) if pd.notna(row["substrate"]) else None,
            "stance": normalize_stance(str(row["stance"])) if pd.notna(row["stance"]) else None,
        }
        candidates: list[float] = []
        for head in HEADS:
            label = values[head]
            count = counts[head].get(label or "", 0)
            if mask_map[head] and label and count > 0 and totals[head] > 0:
                candidates.append(math.sqrt(float(totals[head]) / float(count)))
        raw_weights.append(max(candidates) if candidates else 1.0)
    weights = np.asarray(raw_weights, dtype=np.float64)
    mean_weight = float(weights.mean()) if len(weights) else 1.0
    if mean_weight > 0:
        weights = weights / mean_weight
    return np.minimum(weights, SAMPLER_CAP)


def make_loader(frame: pd.DataFrame, *, train_mode: bool, device: torch.device) -> DataLoader:
    dataset = BirdAttributeDataset(frame, image_size=IMAGE_SIZE, train_mode=train_mode)
    common_loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": (device.type != "cpu"),
        "persistent_workers": NUM_WORKERS > 0,
    }
    if train_mode:
        weights = compute_sampling_weights(frame)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(dataset, sampler=sampler, shuffle=False, **common_loader_kwargs)
    return DataLoader(dataset, shuffle=False, **common_loader_kwargs)


def compute_batch_losses(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
    total = torch.tensor(0.0, device=next(iter(logits.values())).device)
    head_losses: dict[str, torch.Tensor] = {}
    head_counts: dict[str, int] = {}
    for head in HEADS:
        labels = batch[f"{head}_label"]
        mask = batch[f"{head}_mask"].bool()
        visible_count = int(mask.sum().item())
        head_counts[head] = visible_count
        if visible_count > 0:
            head_loss = criterion(logits[head][mask], labels[mask])
            total = total + head_loss
            head_losses[head] = head_loss
        else:
            head_losses[head] = torch.tensor(0.0, device=total.device)
    return total, head_losses, head_counts


def _empty_head_stats() -> dict[str, float]:
    return {head: 0.0 for head in HEADS}


def _accumulate_head_loss(
    accum: dict[str, float],
    counts: dict[str, int],
    head_losses: dict[str, torch.Tensor],
    head_visible: dict[str, int],
) -> None:
    for head in HEADS:
        if head_visible[head] > 0:
            accum[head] += float(head_losses[head].detach().cpu()) * float(head_visible[head])
            counts[head] += int(head_visible[head])


def _average_head_losses(accum: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    averaged = {}
    for head in HEADS:
        averaged[head] = float(accum[head] / counts[head]) if counts[head] > 0 else 0.0
    return averaged


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def evaluate_loader(
    model: MultiHeadAttributeModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, Any]:
    model.eval()
    storage = {head: {"true": [], "pred": []} for head in HEADS}
    total_loss_weighted = 0.0
    total_rows = 0
    head_loss_weighted = {head: 0.0 for head in HEADS}
    head_counts = {head: 0 for head in HEADS}
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            logits = model(x)
            total_loss, batch_head_losses, batch_head_counts = compute_batch_losses(logits, tensor_batch, criterion)
            batch_size_actual = int(x.shape[0])
            total_rows += batch_size_actual
            total_loss_weighted += float(total_loss.detach().cpu()) * batch_size_actual
            for head in HEADS:
                if batch_head_counts[head] > 0:
                    head_loss_weighted[head] += float(batch_head_losses[head].detach().cpu()) * float(batch_head_counts[head])
                    head_counts[head] += int(batch_head_counts[head])
                labels = tensor_batch[f"{head}_label"].detach().cpu().numpy()
                masks = tensor_batch[f"{head}_mask"].detach().cpu().numpy().astype(bool)
                preds = torch.argmax(logits[head], dim=1).detach().cpu().numpy()
                for idx, is_visible in enumerate(masks):
                    if is_visible:
                        storage[head]["true"].append(int(labels[idx]))
                        storage[head]["pred"].append(int(preds[idx]))

    metrics: dict[str, float] = {}
    for head in HEADS:
        y_true = np.array(storage[head]["true"], dtype=np.int64)
        y_pred = np.array(storage[head]["pred"], dtype=np.int64)
        if len(y_true) == 0:
            metrics[f"{head}_accuracy"] = 0.0
            metrics[f"{head}_f1"] = 0.0
            continue
        metrics[f"{head}_accuracy"] = float((y_true == y_pred).mean())
        metrics[f"{head}_f1"] = macro_f1(y_true, y_pred, len(ID_TO_LABELS[head]), ignore_absent_classes=True)
    mean_total_loss = float(total_loss_weighted / total_rows) if total_rows > 0 else 0.0
    mean_head_losses = {
        head: (float(head_loss_weighted[head] / head_counts[head]) if head_counts[head] > 0 else 0.0)
        for head in HEADS
    }
    search_score = (
        0.50 * metrics.get("stance_f1", 0.0)
        + 0.30 * metrics.get("behavior_f1", 0.0)
        + 0.20 * metrics.get("substrate_f1", 0.0)
    )
    return {
        "total_loss": mean_total_loss,
        "head_losses": mean_head_losses,
        "metrics": metrics,
        "search_score": float(search_score),
        "rows": int(total_rows),
    }


def predict_frame(model: MultiHeadAttributeModel, loader: DataLoader, device: torch.device) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    storage = {head: {"true": [], "pred": []} for head in HEADS}
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            logits = model(x)
            batch_size_actual = int(x.shape[0])
            preds_per_head = {head: torch.argmax(logits[head], dim=1).detach().cpu().numpy() for head in HEADS}
            for idx in range(batch_size_actual):
                row = {"row_id": batch["row_id"][idx]}
                for head in HEADS:
                    probs = torch.softmax(logits[head][idx], dim=0)
                    score, pred_idx = torch.max(probs, dim=0)
                    row[head] = ID_TO_LABELS[head][int(pred_idx)]
                    row[f"{head}_conf"] = float(score.item())
                    labels = batch[f"{head}_label"].cpu().numpy()
                    masks = batch[f"{head}_mask"].cpu().numpy().astype(bool)
                    if masks[idx]:
                        storage[head]["true"].append(int(labels[idx]))
                        storage[head]["pred"].append(int(preds_per_head[head][idx]))
                rows.append(row)

    metrics: dict[str, float] = {}
    for head in HEADS:
        y_true = np.array(storage[head]["true"], dtype=np.int64)
        y_pred = np.array(storage[head]["pred"], dtype=np.int64)
        if len(y_true) == 0:
            metrics[f"{head}_accuracy"] = 0.0
            metrics[f"{head}_f1"] = 0.0
            continue
        metrics[f"{head}_accuracy"] = float((y_true == y_pred).mean())
        metrics[f"{head}_f1"] = macro_f1(y_true, y_pred, len(ID_TO_LABELS[head]), ignore_absent_classes=True)
    return pd.DataFrame(rows), metrics


def save_checkpoint(model: MultiHeadAttributeModel, out_dir: Path, train_df: pd.DataFrame) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    supported_labels = {
        head: [label for label, count in counts.items() if int(count) > 0]
        for head, counts in collect_visible_label_counts(train_df).items()
    }
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": model.backbone_name,
            "image_size": IMAGE_SIZE,
            "schema_version": SCHEMA_VERSION,
            "label_maps": LABEL_MAPS,
            "supported_labels": supported_labels,
            "train_label_counts": collect_visible_label_counts(train_df),
        },
        checkpoint_path,
    )
    write_json(
        out_dir / "export_report.json",
        {
            "status": "ready",
            "checkpoint": str(checkpoint_path),
            "backbone": model.backbone_name,
            "image_size": IMAGE_SIZE,
            "schema_version": SCHEMA_VERSION,
            "model_family": build_experiment().model_family,
        },
    )
    return checkpoint_path


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(np.mean([group["lr"] for group in optimizer.param_groups]))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def set_finetune_lrs(optimizer: torch.optim.Optimizer, *, backbone_lr: float, head_lr: float) -> None:
    for group in optimizer.param_groups:
        group_name = group.get("name")
        if group_name == "backbone":
            group["lr"] = float(backbone_lr)
        else:
            group["lr"] = float(head_lr)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def should_pass_guardrails(val_snapshot: dict[str, Any]) -> bool:
    metrics = val_snapshot["metrics"]
    return metrics.get("readability_f1", 0.0) >= 0.30 and metrics.get("specie_f1", 0.0) >= 0.30


def append_history_point(
    history: list[dict[str, Any]],
    *,
    phase: str,
    step: int,
    epoch_fraction: float,
    elapsed_s: float,
    lr: float,
    train_snapshot: dict[str, Any],
    val_snapshot: dict[str, Any],
    best_so_far: bool,
) -> None:
    history.append(
        {
            "phase": phase,
            "step": int(step),
            "epoch_fraction": float(epoch_fraction),
            "elapsed_s": float(elapsed_s),
            "lr": float(lr),
            "train_total_loss": float(train_snapshot["total_loss"]),
            "val_total_loss": float(val_snapshot["total_loss"]),
            "train_head_losses": {head: float(train_snapshot["head_losses"][head]) for head in HEADS},
            "val_head_losses": {head: float(val_snapshot["head_losses"][head]) for head in HEADS},
            "val_metrics_snapshot": {key: float(value) for key, value in val_snapshot["metrics"].items()},
            "val_search_score": float(val_snapshot["search_score"]),
            "best_so_far": bool(best_so_far),
        }
    )


def run_fold(fold_ctx: FoldContext) -> FoldOutput:
    spec = build_experiment()
    run_dir = Path(fold_ctx.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(fold_ctx.seed))

    inner_train_df, inner_val_df, inner_val_fraction = resolve_inner_frames(fold_ctx.train_parquet)
    test_df = pd.read_parquet(fold_ctx.test_parquet)
    device = torch.device(fold_ctx.device if fold_ctx.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_loader = make_loader(inner_train_df, train_mode=True, device=device)
    val_loader = make_loader(inner_val_df, train_mode=False, device=device)
    test_loader = make_loader(test_df, train_mode=False, device=device)
    model = MultiHeadAttributeModel(backbone_name=BACKBONE, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history: list[dict[str, Any]] = []

    started = time.monotonic()
    budget_seconds = float(fold_ctx.budget_seconds)
    head_deadline = started + max(MIN_HEAD_SECONDS, budget_seconds * HEAD_PHASE_RATIO)
    warmup_deadline = started + budget_seconds * (HEAD_PHASE_RATIO + WARMUP_PHASE_RATIO)
    final_deadline = started + max(MIN_FINAL_SECONDS, budget_seconds - 1.0)
    eval_every_steps = EVAL_EVERY_STEPS if budget_seconds >= 45.0 else EVAL_EVERY_STEPS_TIGHT

    model.freeze_backbone()
    head_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    finetune_opt = None
    step_idx = 0
    batches_seen = 0
    unfreeze_step = None
    best_guardrail_snapshot: dict[str, Any] | None = None
    best_loss_snapshot: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, Any] | None = None
    latest_val_snapshot: dict[str, Any] | None = None
    best_early_stop_search_score: float | None = None
    no_improve_evals = 0
    evals_since_unfreeze = 0
    early_stop_reason: str | None = None
    train_loss_accum = 0.0
    train_total_batches = 0
    train_head_accum = _empty_head_stats()
    train_head_counts = {head: 0 for head in HEADS}

    def maybe_record(phase: str, optimizer: torch.optim.Optimizer, *, force: bool = False) -> bool:
        nonlocal best_guardrail_snapshot, best_loss_snapshot, best_state, best_selection, latest_val_snapshot
        nonlocal train_loss_accum, train_total_batches, train_head_accum, train_head_counts
        nonlocal no_improve_evals, evals_since_unfreeze, best_early_stop_search_score

        if train_total_batches == 0:
            return False
        val_snapshot = evaluate_loader(model, val_loader, device, criterion)
        latest_val_snapshot = val_snapshot
        train_snapshot = {
            "total_loss": float(train_loss_accum / train_total_batches),
            "head_losses": _average_head_losses(train_head_accum, train_head_counts),
        }

        selection_snapshot = {
            "step": step_idx,
            "phase": phase,
            "val_total_loss": float(val_snapshot["total_loss"]),
            "val_search_score": float(val_snapshot["search_score"]),
            "val_metrics": {key: float(value) for key, value in val_snapshot["metrics"].items()},
            "elapsed_s": float(time.monotonic() - started),
        }
        improved = False
        if should_pass_guardrails(val_snapshot):
            if best_guardrail_snapshot is None or val_snapshot["search_score"] > best_guardrail_snapshot["val_search_score"]:
                best_guardrail_snapshot = selection_snapshot
                best_state = clone_state_dict(model)
                best_selection = {"selected_by": "guardrail_search", **selection_snapshot}
                improved = True
        if best_loss_snapshot is None or val_snapshot["total_loss"] < best_loss_snapshot["val_total_loss"]:
            best_loss_snapshot = selection_snapshot
            if best_guardrail_snapshot is None:
                best_state = clone_state_dict(model)
                best_selection = {"selected_by": "val_total_loss", **selection_snapshot}
                improved = True

        if phase != "head":
            evals_since_unfreeze += 1
            if best_early_stop_search_score is None or val_snapshot["search_score"] > best_early_stop_search_score + EARLY_STOP_MIN_DELTA:
                best_early_stop_search_score = float(val_snapshot["search_score"])
                no_improve_evals = 0
            else:
                no_improve_evals += 1

        append_history_point(
            history,
            phase=phase,
            step=step_idx,
            epoch_fraction=float(batches_seen / max(1, len(train_loader))),
            elapsed_s=time.monotonic() - started,
            lr=current_lr(optimizer),
            train_snapshot=train_snapshot,
            val_snapshot=val_snapshot,
            best_so_far=improved,
        )
        train_loss_accum = 0.0
        train_total_batches = 0
        train_head_accum = _empty_head_stats()
        train_head_counts = {head: 0 for head in HEADS}
        return improved

    def train_batch(batch: dict[str, Any], optimizer: torch.optim.Optimizer, phase: str) -> None:
        nonlocal step_idx, batches_seen, train_loss_accum, train_total_batches
        x = batch["image"].to(device)
        tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            logits = model(x)
            loss, head_losses, head_counts = compute_batch_losses(logits, tensor_batch, criterion)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        step_idx += 1
        batches_seen += 1
        train_total_batches += 1
        train_loss_accum += float(loss.detach().cpu())
        _accumulate_head_loss(train_head_accum, train_head_counts, head_losses, head_counts)
        dynamic_eval_every = eval_every_steps
        if phase != "head":
            late_phase_threshold = warmup_deadline + max(0.0, (final_deadline - warmup_deadline) * 0.50)
            if time.monotonic() >= late_phase_threshold:
                dynamic_eval_every = LATE_PHASE_EVAL_EVERY_STEPS
        if step_idx == 1 or step_idx % dynamic_eval_every == 0:
            maybe_record(phase, optimizer)

    # Head-only phase
    while time.monotonic() < head_deadline:
        model.train(True)
        for batch in train_loader:
            train_batch(batch, head_opt, "head")
            if time.monotonic() >= head_deadline:
                break

    maybe_record("head", head_opt, force=True)

    # Finetune phases
    model.unfreeze_backbone()
    head_modules = [model.readability, model.specie, model.behavior, model.substrate, model.stance]
    head_params = []
    for module in head_modules:
        head_params.extend(list(module.parameters()))
    finetune_opt = torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters()), "lr": FINETUNE_BACKBONE_LR * 0.10, "name": "backbone"},
            {"params": head_params, "lr": FINETUNE_HEAD_LR * 0.10, "name": "heads"},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    unfreeze_step = step_idx
    while time.monotonic() < final_deadline:
        model.train(True)
        for batch in train_loader:
            now = time.monotonic()
            if now >= final_deadline:
                break
            if now <= warmup_deadline:
                denom = max(1e-6, warmup_deadline - head_deadline)
                progress = min(1.0, max(0.0, (now - head_deadline) / denom))
                backbone_lr = FINETUNE_BACKBONE_LR * (0.10 + 0.90 * progress)
                head_lr = FINETUNE_HEAD_LR * (0.10 + 0.90 * progress)
                phase = "warmup"
            else:
                denom = max(1e-6, final_deadline - warmup_deadline)
                progress = min(1.0, max(0.0, (now - warmup_deadline) / denom))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                backbone_lr = FINETUNE_BACKBONE_LR * (FINETUNE_END_LR_SCALE + (1.0 - FINETUNE_END_LR_SCALE) * cosine)
                head_lr = FINETUNE_HEAD_LR * (FINETUNE_END_LR_SCALE + (1.0 - FINETUNE_END_LR_SCALE) * cosine)
                phase = "finetune"
            set_finetune_lrs(finetune_opt, backbone_lr=backbone_lr, head_lr=head_lr)
            train_batch(batch, finetune_opt, phase)
            early_stop_gate = warmup_deadline + max(0.0, (final_deadline - warmup_deadline) * EARLY_STOP_START_PROGRESS)
            retention_triggered = (
                latest_val_snapshot is not None
                and best_early_stop_search_score is not None
                and float(latest_val_snapshot["search_score"]) < float(best_early_stop_search_score) * EARLY_STOP_RETENTION_FLOOR
            )
            if (
                now >= early_stop_gate
                and evals_since_unfreeze >= MIN_EVALS_BEFORE_EARLY_STOP
                and (no_improve_evals >= EARLY_STOP_PATIENCE or retention_triggered)
            ):
                if retention_triggered and latest_val_snapshot is not None and best_early_stop_search_score is not None:
                    ratio = float(latest_val_snapshot["search_score"]) / float(best_early_stop_search_score)
                    early_stop_reason = f"val_search_score retention {ratio:.3f} below {EARLY_STOP_RETENTION_FLOOR:.3f}"
                else:
                    early_stop_reason = f"no improvement for {no_improve_evals} evals"
                break
            if time.monotonic() >= final_deadline:
                break
        if early_stop_reason:
            break

    if finetune_opt is not None and early_stop_reason is None:
        maybe_record("finetune", finetune_opt, force=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_snapshot = evaluate_loader(model, val_loader, device, criterion)
    prediction_frame, local_metrics = predict_frame(model, test_loader, device)
    candidate_dir = run_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = candidate_dir / "predictions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)

    supported_labels = {
        head: [label for label, count in counts.items() if int(count) > 0]
        for head, counts in collect_visible_label_counts(inner_train_df).items()
    }
    training_summary = {
        "inner_train_rows": int(len(inner_train_df)),
        "inner_val_rows": int(len(inner_val_df)),
        "inner_val_fraction": float(inner_val_fraction),
        "eval_every_steps": int(eval_every_steps),
        "late_phase_eval_every_steps": int(LATE_PHASE_EVAL_EVERY_STEPS),
        "unfreeze_step": int(unfreeze_step or 0),
        "final_step": int(step_idx),
        "backbone": BACKBONE,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "head_lr": HEAD_LR,
        "finetune_backbone_lr": FINETUNE_BACKBONE_LR,
        "finetune_head_lr": FINETUNE_HEAD_LR,
        "sampler_cap": SAMPLER_CAP,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "early_stop_reason": early_stop_reason,
        "no_improve_evals": int(no_improve_evals),
        "early_stop_best_val_search_score": float(best_early_stop_search_score or 0.0),
        "early_stop_metric": "val_search_score",
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "early_stop_start_progress": EARLY_STOP_START_PROGRESS,
        "early_stop_retention_floor": EARLY_STOP_RETENTION_FLOOR,
    }
    selection = best_selection or {
        "selected_by": "final_state",
        "step": int(step_idx),
        "phase": "finetune",
        "val_total_loss": float(final_val_snapshot["total_loss"]),
        "val_search_score": float(final_val_snapshot["search_score"]),
        "val_metrics": {key: float(value) for key, value in final_val_snapshot["metrics"].items()},
        "elapsed_s": float(time.monotonic() - started),
    }
    artifact_meta = {
        "artifact_type": "canonical_label_predictions",
        "prediction_file": prediction_path.name,
        "training_history_file": "training_history.json",
        "model_family": spec.model_family,
        "experiment_name": spec.name,
        "description": spec.description,
        "schema_version": SCHEMA_VERSION,
        "supported_labels": supported_labels,
        "train_visible_label_counts": collect_visible_label_counts(inner_train_df),
        "inner_val_visible_label_counts": collect_visible_label_counts(inner_val_df),
        "backbone": BACKBONE,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "seed": int(fold_ctx.seed),
        "sampler_cap": SAMPLER_CAP,
    }
    write_json(candidate_dir / "artifact_meta.json", artifact_meta)
    write_json(
        candidate_dir / "training_history.json",
        {
            "history": history,
            "summary": training_summary,
            "selection": selection,
        },
    )

    export_hint: dict[str, Any] = {"status": "prediction_only"}
    if spec.supports_export_now and str(fold_ctx.fold_id) == "full_export":
        checkpoint_dir = candidate_dir / "current_backend"
        checkpoint_path = save_checkpoint(model, checkpoint_dir, inner_train_df)
        export_hint = {"status": "ready", "checkpoint": str(checkpoint_path)}

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    notes = (
        f"backbone={BACKBONE} image_size={IMAGE_SIZE} batch_size={BATCH_SIZE} "
        f"head_lr={HEAD_LR} finetune_backbone_lr={FINETUNE_BACKBONE_LR} "
        f"finetune_head_lr={FINETUNE_HEAD_LR} sampler_cap={SAMPLER_CAP} "
        f"early_stop_patience={EARLY_STOP_PATIENCE} early_stop_min_delta={EARLY_STOP_MIN_DELTA}"
    )
    return FoldOutput(
        metrics=local_metrics,
        train_seconds=float(time.monotonic() - started),
        peak_vram_mb=peak_vram_mb,
        candidate_artifact_dir=str(candidate_dir),
        export_hint=export_hint,
        notes=notes,
        best_step=int(selection["step"]),
        best_val_search_score=float(selection["val_search_score"]),
        best_val_loss=float(selection["val_total_loss"]),
        final_val_loss=float(final_val_snapshot["total_loss"]),
        inner_val_rows=int(len(inner_val_df)),
    )
