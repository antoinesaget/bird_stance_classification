from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from birdsys.training.attributes import (
    BEHAVIOR_TO_ID,
    READABILITY_TO_ID,
    SPECIE_TO_ID,
    STANCE_TO_ID,
    SUBSTRATE_TO_ID,
)
from birdsys.training.models import MultiHeadAttributeModel

HEADS = ["readability", "specie", "behavior", "substrate", "stance"]
LABEL_MAPS = {
    "readability": READABILITY_TO_ID,
    "specie": SPECIE_TO_ID,
    "behavior": BEHAVIOR_TO_ID,
    "substrate": SUBSTRATE_TO_ID,
    "stance": STANCE_TO_ID,
}
DEFAULT_LABELS = {
    head: [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
    for head, mapping in LABEL_MAPS.items()
}


@dataclass
class LoadedModelBMember:
    name: str
    source_path: Path
    model: MultiHeadAttributeModel
    backbone: str
    image_size: int
    schema_version: str
    checkpoint_heads: tuple[str, ...]
    inference_heads: tuple[str, ...]
    id_to_label: dict[str, list[str]]
    supported_labels: dict[str, set[str]]
    train_label_counts: dict[str, dict[str, int]]


@dataclass
class LoadedModelBArtifact:
    mode: str
    source_path: Path
    schema_version: str
    members: list[LoadedModelBMember]
    id_to_label: dict[str, list[str]]
    supported_labels: dict[str, set[str]]
    train_label_counts: dict[str, dict[str, int]]
    head_to_member: dict[str, str]


def _coerce_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _invert_map(mapping: dict[str, int], fallback: list[str]) -> list[str]:
    if not mapping:
        return fallback[:]
    size = max(int(value) for value in mapping.values()) + 1
    out = [""] * size
    for label, idx in mapping.items():
        out[int(idx)] = str(label)
    for idx, value in enumerate(out):
        if not value and idx < len(fallback):
            out[idx] = fallback[idx]
    return out


def decode_label_maps(raw: dict[str, Any]) -> dict[str, list[str]]:
    out = {head: labels[:] for head, labels in DEFAULT_LABELS.items()}
    for head, fallback in DEFAULT_LABELS.items():
        values = raw.get(head)
        if isinstance(values, dict):
            out[head] = _invert_map(values, fallback)
    return out


def decode_supported_labels(raw: dict[str, Any], id_to_label: dict[str, list[str]]) -> dict[str, set[str]]:
    out = {head: set(labels) for head, labels in id_to_label.items()}
    for head in HEADS:
        values = raw.get(head)
        if isinstance(values, list) and values:
            out[head] = {str(item) for item in values}
    return out


def decode_train_label_counts(raw: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        str(head): {str(label): int(count) for label, count in (counts or {}).items()}
        for head, counts in (raw or {}).items()
    }


def resolve_active_heads(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return tuple(HEADS)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Invalid active_heads value: {raw!r}")
    heads = tuple(str(item) for item in raw)
    invalid = [head for head in heads if head not in HEADS]
    if invalid:
        raise ValueError(f"Unsupported heads: {invalid}")
    return heads


def fallback_label(head: str, id_to_label: dict[str, list[str]]) -> str:
    labels = id_to_label.get(head) or DEFAULT_LABELS[head]
    if "unsure" in labels:
        return "unsure"
    return labels[0]


def coerce_supported_label(
    head: str,
    label: str,
    score: float,
    *,
    id_to_label: dict[str, list[str]],
    supported_labels: dict[str, set[str]],
    diagnostics: dict[str, dict[str, int]] | None = None,
) -> tuple[str, float]:
    allowed = supported_labels.get(head) or set()
    if not allowed or label in allowed:
        return label, score
    fallback = fallback_label(head, id_to_label)
    if diagnostics is not None:
        key = f"{label}->{fallback}"
        diagnostics.setdefault(head, {})
        diagnostics[head][key] = int(diagnostics[head].get(key, 0)) + 1
    return fallback, min(score, 0.50)


def decode_head_logits(
    logits: torch.Tensor,
    head: str,
    *,
    id_to_label: dict[str, list[str]],
    supported_labels: dict[str, set[str]],
    diagnostics: dict[str, dict[str, int]] | None = None,
) -> tuple[str, float]:
    probs = torch.softmax(logits, dim=0)
    score, idx = torch.max(probs, dim=0)
    labels = id_to_label[head]
    label = labels[int(idx)] if int(idx) < len(labels) and labels[int(idx)] else DEFAULT_LABELS[head][0]
    return coerce_supported_label(
        head,
        label,
        float(score.item()),
        id_to_label=id_to_label,
        supported_labels=supported_labels,
        diagnostics=diagnostics,
    )


def apply_prediction_guards(
    *,
    predictions: dict[str, str],
    confidences: dict[str, float],
    id_to_label: dict[str, list[str]],
    unreadable_or_incorrect_suppressions: dict[str, int] | None = None,
    stance_suppressions_non_relevant: list[int] | None = None,
) -> tuple[dict[str, str], dict[str, float]]:
    if predictions["readability"] == "unreadable" or predictions["specie"] == "incorrect":
        for head in ("behavior", "substrate", "stance"):
            predictions[head] = fallback_label(head, id_to_label)
            confidences[head] = min(float(confidences.get(head, 0.0)), 0.50)
            if unreadable_or_incorrect_suppressions is not None:
                unreadable_or_incorrect_suppressions[head] = int(unreadable_or_incorrect_suppressions.get(head, 0)) + 1
    elif not (
        predictions["behavior"] in {"resting", "backresting"}
        and predictions["substrate"] in {"bare_ground", "water", "unsure"}
    ):
        predictions["stance"] = fallback_label("stance", id_to_label)
        confidences["stance"] = min(float(confidences.get("stance", 0.0)), 0.55)
        if stance_suppressions_non_relevant is not None:
            stance_suppressions_non_relevant[0] = int(stance_suppressions_non_relevant[0]) + 1
    return predictions, confidences


def _load_member_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    name: str,
    manifest_heads: tuple[str, ...] | None,
) -> LoadedModelBMember:
    payload = torch.load(checkpoint_path, map_location="cpu")
    backbone = str(payload.get("backbone") or "convnextv2_tiny.fcmae_ft_in1k")
    image_size = int(payload.get("image_size") or 224)
    schema_version = str(payload.get("schema_version") or "annotation_schema_v2")
    checkpoint_heads = resolve_active_heads(payload.get("active_heads"))
    inference_heads = manifest_heads or checkpoint_heads
    missing = [head for head in inference_heads if head not in checkpoint_heads]
    if missing:
        raise ValueError(
            f"Bundle member {name!r} requests heads {missing} not present in checkpoint {checkpoint_path}"
        )

    model = MultiHeadAttributeModel(backbone_name=backbone, pretrained=False, heads=list(checkpoint_heads))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    model = model.to(device)

    id_to_label = decode_label_maps(payload.get("label_maps") or {})
    supported_labels = decode_supported_labels(payload.get("supported_labels") or {}, id_to_label)
    train_label_counts = decode_train_label_counts(payload.get("train_label_counts") or {})
    return LoadedModelBMember(
        name=name,
        source_path=checkpoint_path,
        model=model,
        backbone=backbone,
        image_size=image_size,
        schema_version=schema_version,
        checkpoint_heads=checkpoint_heads,
        inference_heads=inference_heads,
        id_to_label=id_to_label,
        supported_labels=supported_labels,
        train_label_counts=train_label_counts,
    )


def _combine_member_metadata(
    members: list[LoadedModelBMember],
) -> tuple[dict[str, list[str]], dict[str, set[str]], dict[str, dict[str, int]], dict[str, str]]:
    if not members:
        raise ValueError("Model B artifact must contain at least one member")

    id_to_label = {head: DEFAULT_LABELS[head][:] for head in HEADS}
    supported_labels = {head: set(DEFAULT_LABELS[head]) for head in HEADS}
    train_label_counts = {head: {} for head in HEADS}
    head_to_member: dict[str, str] = {}
    for member in members:
        for head in member.inference_heads:
            if head in head_to_member:
                raise ValueError(f"Head {head!r} is assigned to multiple bundle members")
            head_to_member[head] = member.name
            id_to_label[head] = member.id_to_label[head][:]
            supported_labels[head] = set(member.supported_labels.get(head) or set(member.id_to_label[head]))
            train_label_counts[head] = {
                str(label): int(count)
                for label, count in (member.train_label_counts.get(head) or {}).items()
            }
    missing_heads = [head for head in HEADS if head not in head_to_member]
    if missing_heads:
        raise ValueError(f"Bundle manifest does not cover all heads: {missing_heads}")
    return id_to_label, supported_labels, train_label_counts, head_to_member


def _resolve_single_checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        checkpoint_path = path / "checkpoint.pt"
        if checkpoint_path.exists():
            return checkpoint_path
    if path.is_file():
        return path
    raise FileNotFoundError(path)


def load_model_b_artifact(artifact_path: Path, *, device: torch.device | str) -> LoadedModelBArtifact:
    resolved_path = Path(artifact_path).expanduser().resolve()
    resolved_device = _coerce_device(device)

    manifest_path = resolved_path / "manifest.json"
    if resolved_path.is_dir() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_type = str(manifest.get("artifact_type") or "")
        if artifact_type != "model_b_specialist_bundle":
            raise ValueError(f"Unsupported Model B artifact_type={artifact_type!r} in {manifest_path}")
        member_specs = manifest.get("members")
        if not isinstance(member_specs, list) or not member_specs:
            raise ValueError(f"Invalid or empty members list in {manifest_path}")

        members: list[LoadedModelBMember] = []
        for raw_member in member_specs:
            if not isinstance(raw_member, dict):
                raise ValueError(f"Invalid member entry in {manifest_path}: {raw_member!r}")
            member_name = str(raw_member.get("name") or "").strip()
            checkpoint_ref = str(raw_member.get("checkpoint") or "").strip()
            if not member_name or not checkpoint_ref:
                raise ValueError(f"Each bundle member requires name and checkpoint in {manifest_path}")
            checkpoint_path = (resolved_path / checkpoint_ref).resolve()
            member_heads = resolve_active_heads(raw_member.get("heads")) if raw_member.get("heads") is not None else None
            member = _load_member_from_checkpoint(
                checkpoint_path,
                device=resolved_device,
                name=member_name,
                manifest_heads=member_heads,
            )
            expected_backbone = raw_member.get("backbone")
            if expected_backbone is not None and str(expected_backbone) != member.backbone:
                raise ValueError(
                    f"Bundle member {member_name!r} backbone mismatch: manifest={expected_backbone!r} checkpoint={member.backbone!r}"
                )
            expected_image_size = raw_member.get("image_size")
            if expected_image_size is not None and int(expected_image_size) != member.image_size:
                raise ValueError(
                    f"Bundle member {member_name!r} image_size mismatch: manifest={expected_image_size!r} checkpoint={member.image_size!r}"
                )
            members.append(member)

        id_to_label, supported_labels, train_label_counts, head_to_member = _combine_member_metadata(members)
        schema_version = str(manifest.get("schema_version") or members[0].schema_version)
        return LoadedModelBArtifact(
            mode="specialist_bundle",
            source_path=resolved_path,
            schema_version=schema_version,
            members=members,
            id_to_label=id_to_label,
            supported_labels=supported_labels,
            train_label_counts=train_label_counts,
            head_to_member=head_to_member,
        )

    checkpoint_path = _resolve_single_checkpoint_path(resolved_path)
    member = _load_member_from_checkpoint(
        checkpoint_path,
        device=resolved_device,
        name="model_b",
        manifest_heads=None,
    )
    id_to_label, supported_labels, train_label_counts, head_to_member = _combine_member_metadata([member])
    return LoadedModelBArtifact(
        mode="legacy_single_checkpoint",
        source_path=resolved_path,
        schema_version=member.schema_version,
        members=[member],
        id_to_label=id_to_label,
        supported_labels=supported_labels,
        train_label_counts=train_label_counts,
        head_to_member=head_to_member,
    )
