from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from birdsys.training.attributes import (
    BEHAVIOR_TO_ID,
    READABILITY_TO_ID,
    SPECIE_TO_ID,
    STANCE_TO_ID,
    SUBSTRATE_TO_ID,
    compute_head_masks,
    normalize_behavior,
    normalize_choice,
    normalize_stance,
    normalize_substrate,
)
from birdsys.training.metrics import class_supports, confusion_matrix, macro_f1
from birdsys.training.model_b_artifacts import (
    HEADS,
    LoadedModelBArtifact,
    apply_prediction_guards,
    decode_head_logits,
    fallback_label,
    load_model_b_artifact,
)

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
class EvaluationDiagnostics:
    rows_scored: int
    unsupported_label_coercions: dict[str, dict[str, int]]
    unreadable_or_incorrect_suppressions: dict[str, int]
    stance_suppressions_non_relevant: int
    prediction_label_counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class EvaluationResult:
    checkpoint_path: str
    artifact_mode: str
    device: str
    image_size: int
    schema_version: str
    supported_labels: dict[str, list[str]]
    train_label_counts: dict[str, dict[str, int]]
    metrics: dict[str, float]
    confusion_matrices: dict[str, np.ndarray]
    visible_label_counts: dict[str, dict[str, int]]
    dataset_label_counts: dict[str, dict[str, int]]
    diagnostics: EvaluationDiagnostics
    prediction_frame: pd.DataFrame


def sanitize_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["crop_path", "isbird", "readability", "specie", "behavior", "substrate", "stance"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    out = df.copy()
    out["crop_path"] = out["crop_path"].astype(str)
    out = out[out["crop_path"].map(lambda value: Path(value).exists())].reset_index(drop=True)
    out["isbird"] = out["isbird"].map(lambda value: normalize_choice(None if pd.isna(value) else str(value)))
    out = out[out["isbird"] == "yes"].reset_index(drop=True)
    out["readability"] = out["readability"].map(lambda value: normalize_choice(None if pd.isna(value) else str(value)))
    out["specie"] = out["specie"].map(lambda value: normalize_choice(None if pd.isna(value) else str(value)))
    out["behavior"] = out["behavior"].map(lambda value: normalize_behavior(None if pd.isna(value) else str(value)))
    out["substrate"] = out["substrate"].map(lambda value: normalize_substrate(None if pd.isna(value) else str(value)))
    out["stance"] = out["stance"].map(lambda value: normalize_stance(None if pd.isna(value) else str(value)))
    if "row_id" not in out.columns:
        out["row_id"] = [f"row_{idx:06d}" for idx in range(len(out))]
    out["row_id"] = out["row_id"].astype(str)
    return out.reset_index(drop=True)


def summarize_dataset_labels(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for column in ["isbird", "readability", "specie", "behavior", "substrate", "stance"]:
        counts = frame[column].fillna("<null>").value_counts().sort_index().to_dict()
        out[column] = {str(key): int(value) for key, value in counts.items()}
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


class CropPathDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        return {
            "crop_path": str(row["crop_path"]),
            "row_index": int(idx),
            "row_id": str(row["row_id"]),
        }


def _pick_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def _member_transforms(artifact: LoadedModelBArtifact) -> dict[str, transforms.Compose]:
    return {
        member.name: transforms.Compose(
            [
                transforms.Resize((member.image_size, member.image_size)),
                transforms.ToTensor(),
            ]
        )
        for member in artifact.members
    }


def _load_batch_images(crop_paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for raw_path in crop_paths:
        image_path = Path(str(raw_path))
        with Image.open(image_path).convert("RGB") as image:
            images.append(image.copy())
    return images


def evaluate_checkpoint_on_frame(
    *,
    checkpoint_path: Path,
    frame: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "auto",
) -> EvaluationResult:
    clean_frame = sanitize_eval_df(frame)
    eval_device = _pick_device(device)
    artifact = load_model_b_artifact(checkpoint_path, device=eval_device)
    image_size = max(int(member.image_size) for member in artifact.members)
    transforms_by_member = _member_transforms(artifact)

    dataset = CropPathDataset(clean_frame)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    storage: dict[str, dict[str, list[int]]] = {head: {"true": [], "pred": []} for head in HEADS}
    unsupported_label_coercions = {head: {} for head in HEADS}
    unreadable_or_incorrect_suppressions = {"behavior": 0, "substrate": 0, "stance": 0}
    stance_suppressions_non_relevant = [0]
    prediction_label_counts = {head: {label: 0 for label in ID_TO_LABELS[head]} for head in HEADS}
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            crop_paths = [str(item) for item in batch["crop_path"]]
            row_ids = [str(item) for item in batch["row_id"]]
            row_indices = [int(item) for item in batch["row_index"]]
            images = _load_batch_images(crop_paths)
            batch_size_actual = len(images)
            label_by_idx = [{head: fallback_label(head, artifact.id_to_label) for head in HEADS} for _ in range(batch_size_actual)]
            conf_by_idx = [{head: 0.0 for head in HEADS} for _ in range(batch_size_actual)]

            for member in artifact.members:
                transform = transforms_by_member[member.name]
                x = torch.stack([transform(image) for image in images], dim=0).to(eval_device)
                logits = member.model(x)
                for batch_idx in range(batch_size_actual):
                    for head in member.inference_heads:
                        label, score = decode_head_logits(
                            logits[head][batch_idx],
                            head,
                            id_to_label=member.id_to_label,
                            supported_labels=member.supported_labels,
                            diagnostics=unsupported_label_coercions,
                        )
                        label_by_idx[batch_idx][head] = label
                        conf_by_idx[batch_idx][head] = score

            for batch_idx in range(batch_size_actual):
                predictions = dict(label_by_idx[batch_idx])
                confidences = dict(conf_by_idx[batch_idx])
                predictions, confidences = apply_prediction_guards(
                    predictions=predictions,
                    confidences=confidences,
                    id_to_label=artifact.id_to_label,
                    unreadable_or_incorrect_suppressions=unreadable_or_incorrect_suppressions,
                    stance_suppressions_non_relevant=stance_suppressions_non_relevant,
                )
                row_index = row_indices[batch_idx]
                row_id = row_ids[batch_idx]
                row = clean_frame.iloc[row_index]
                masks = compute_head_masks(
                    isbird=str(row["isbird"]),
                    readability=str(row["readability"]),
                    specie=str(row["specie"]),
                    behavior=str(row["behavior"]) if pd.notna(row["behavior"]) else None,
                    substrate=str(row["substrate"]) if pd.notna(row["substrate"]) else None,
                )
                true_labels = {
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
                    pred_label = predictions[head]
                    if pred_label in prediction_label_counts[head]:
                        prediction_label_counts[head][pred_label] += 1
                    if not mask_map[head]:
                        continue
                    true_label = true_labels[head]
                    if true_label not in LABEL_MAPS[head]:
                        continue
                    storage[head]["true"].append(LABEL_MAPS[head][true_label])
                    storage[head]["pred"].append(LABEL_MAPS[head][pred_label])

                prediction_rows.append(
                    {
                        "row_id": row_id,
                        "readability": predictions["readability"],
                        "readability_conf": float(confidences["readability"]),
                        "specie": predictions["specie"],
                        "specie_conf": float(confidences["specie"]),
                        "behavior": predictions["behavior"],
                        "behavior_conf": float(confidences["behavior"]),
                        "substrate": predictions["substrate"],
                        "substrate_conf": float(confidences["substrate"]),
                        "stance": predictions["stance"],
                        "stance_conf": float(confidences["stance"]),
                    }
                )

    metrics: dict[str, float] = {}
    confusion_matrices = {
        head: np.zeros((HEAD_CLASS_COUNTS[head], HEAD_CLASS_COUNTS[head]), dtype=np.int64)
        for head in CONFUSION_HEADS
    }
    visible_label_counts = collect_visible_label_counts(clean_frame)
    for head, num_classes in HEAD_CLASS_COUNTS.items():
        y_true = np.array(storage[head]["true"], dtype=np.int64)
        y_pred = np.array(storage[head]["pred"], dtype=np.int64)
        if len(y_true) == 0:
            metrics[f"{head}_accuracy"] = 0.0
            metrics[f"{head}_f1"] = 0.0
            continue
        metrics[f"{head}_accuracy"] = float((y_true == y_pred).mean())
        metrics[f"{head}_f1"] = macro_f1(y_true, y_pred, num_classes, ignore_absent_classes=True)
        if head in confusion_matrices:
            confusion_matrices[head] = confusion_matrix(y_true, y_pred, num_classes)

    diagnostics = EvaluationDiagnostics(
        rows_scored=int(len(clean_frame)),
        unsupported_label_coercions=unsupported_label_coercions,
        unreadable_or_incorrect_suppressions=unreadable_or_incorrect_suppressions,
        stance_suppressions_non_relevant=int(stance_suppressions_non_relevant[0]),
        prediction_label_counts=prediction_label_counts,
    )
    return EvaluationResult(
        checkpoint_path=str(checkpoint_path),
        artifact_mode=artifact.mode,
        device=str(eval_device),
        image_size=image_size,
        schema_version=str(artifact.schema_version),
        supported_labels={
            head: sorted(values, key=lambda label: LABEL_MAPS[head].get(label, 999))
            for head, values in artifact.supported_labels.items()
        },
        train_label_counts=artifact.train_label_counts,
        metrics=metrics,
        confusion_matrices=confusion_matrices,
        visible_label_counts=visible_label_counts,
        dataset_label_counts=summarize_dataset_labels(clean_frame),
        diagnostics=diagnostics,
        prediction_frame=pd.DataFrame(prediction_rows),
    )


def evaluate_checkpoint_on_dataset(
    *,
    checkpoint_path: Path,
    dataset_dir: Path,
    split: str,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "auto",
) -> EvaluationResult:
    split_path = dataset_dir / f"{split}.parquet"
    frame = pd.read_parquet(split_path)
    return evaluate_checkpoint_on_frame(
        checkpoint_path=checkpoint_path,
        frame=frame,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def evaluation_result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    return {
        "checkpoint_path": result.checkpoint_path,
        "artifact_mode": result.artifact_mode,
        "device": result.device,
        "image_size": result.image_size,
        "schema_version": result.schema_version,
        "supported_labels": result.supported_labels,
        "train_label_counts": result.train_label_counts,
        "metrics": result.metrics,
        "visible_label_counts": result.visible_label_counts,
        "dataset_label_counts": result.dataset_label_counts,
        "diagnostics": asdict(result.diagnostics),
        "confusion_matrices": {
            head: result.confusion_matrices[head].tolist()
            for head in result.confusion_matrices
        },
    }


def confusion_to_csv_frame(result: EvaluationResult, head: str) -> pd.DataFrame:
    matrix = result.confusion_matrices[head]
    return pd.DataFrame(matrix, index=ID_TO_LABELS[head], columns=ID_TO_LABELS[head])


def class_support_counts_from_storage(y_true: np.ndarray, head: str) -> dict[str, int]:
    supports = class_supports(y_true, HEAD_CLASS_COUNTS[head]).tolist()
    return {label: int(count) for label, count in zip(ID_TO_LABELS[head], supports)}
