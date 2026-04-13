"""Purpose: Load Model B and turn detections into the attribute predictions returned to Label Studio"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from birdsys.core.model_b_artifacts import (
    DEFAULT_LABELS,
    HEADS,
    LoadedModelBArtifact,
    apply_prediction_guards,
    coerce_supported_label,
    decode_head_logits,
    fallback_label,
    load_model_b_artifact,
)

from .model_a_yolo import Detection

LOGGER = logging.getLogger("birdsys.ml_backend.model_b")


@dataclass(frozen=True)
class AttributePrediction:
    readability: str
    readability_conf: float
    specie: str
    specie_conf: float
    behavior: str
    behavior_conf: float
    substrate: str
    substrate_conf: float
    stance: str
    stance_conf: float


class AttributePredictor:
    """
    Model B inference adapter.

    The backend still offers heuristic prefill behavior when no checkpoint is
    available, but those heuristics now speak the canonical schema-v2 labels.
    """

    def __init__(self, checkpoint_path: Path | None = None, *, device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = self._pick_device(device)
        self.artifact: LoadedModelBArtifact | None = None
        self.model: Any | None = None
        self.mode = "heuristic"
        self.image_size = 224
        self.schema_version = "annotation_schema_v2"
        self.id_to_label = {head: labels[:] for head, labels in DEFAULT_LABELS.items()}
        self.supported_labels = {head: set(labels) for head, labels in DEFAULT_LABELS.items()}
        self.train_label_counts: dict[str, dict[str, int]] = {}
        self.members: list[dict[str, Any]] = []
        self.member_transforms: dict[str, transforms.Compose] = {}

        if checkpoint_path and checkpoint_path.exists():
            try:
                artifact = load_model_b_artifact(checkpoint_path, device=self.device)
                self.artifact = artifact
                self.model = artifact.members[0].model if artifact.members else None
                self.mode = artifact.mode
                self.image_size = max(member.image_size for member in artifact.members)
                self.schema_version = artifact.schema_version
                self.id_to_label = {head: labels[:] for head, labels in artifact.id_to_label.items()}
                self.supported_labels = {head: set(labels) for head, labels in artifact.supported_labels.items()}
                self.train_label_counts = {
                    str(head): {str(label): int(count) for label, count in (counts or {}).items()}
                    for head, counts in artifact.train_label_counts.items()
                }
                self.members = [
                    {
                        "name": member.name,
                        "loaded": True,
                        "checkpoint": str(member.source_path),
                        "backbone": member.backbone,
                        "image_size": int(member.image_size),
                        "checkpoint_heads": list(member.checkpoint_heads),
                        "inference_heads": list(member.inference_heads),
                    }
                    for member in artifact.members
                ]
                self.member_transforms = {
                    member.name: transforms.Compose(
                        [
                            transforms.Resize((member.image_size, member.image_size)),
                            transforms.ToTensor(),
                        ]
                    )
                    for member in artifact.members
                }
                LOGGER.info(
                    "Loaded Model B artifact=%s mode=%s device=%s members=%d schema=%s",
                    checkpoint_path,
                    self.mode,
                    self.device,
                    len(self.members),
                    self.schema_version,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to load Model B artifact=%s: %s", checkpoint_path, exc)
                self.artifact = None
                self.model = None

        if not self.member_transforms:
            self.member_transforms = {
                "heuristic": transforms.Compose(
                    [
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                    ]
                )
            }

    @staticmethod
    def _pick_device(requested: str | None = None) -> str:
        req = (requested or "auto").strip().lower()
        if req != "auto":
            if req == "cpu":
                return "cpu"
            if req == "mps":
                if torch.backends.mps.is_available():
                    return "mps"
                LOGGER.warning("Model B device=mps requested but MPS is unavailable; falling back to CPU")
                return "cpu"
            if req == "cuda" or req.startswith("cuda") or req.isdigit():
                if torch.cuda.is_available():
                    return "cuda"
                LOGGER.warning("Model B device=%s requested but CUDA is unavailable; falling back to CPU", req)
                return "cpu"
            return req

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _crop_box(image: Image.Image, det: Detection) -> Image.Image:
        width, height = image.size
        x1 = int(round(det.x * width))
        y1 = int(round(det.y * height))
        x2 = int(round((det.x + det.w) * width))
        y2 = int(round((det.y + det.h) * height))

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        return image.crop((x1, y1, x2, y2))

    def _fallback_label(self, head: str) -> str:
        return fallback_label(head, self.id_to_label)

    def _coerce_supported_label(self, head: str, label: str, score: float) -> tuple[str, float]:
        return coerce_supported_label(
            head,
            label,
            score,
            id_to_label=self.id_to_label,
            supported_labels=self.supported_labels,
        )

    def _heuristic(self, det: Detection) -> AttributePrediction:
        readability_score = min(0.99, max(0.05, 0.25 + 0.85 * det.score))
        if readability_score >= 0.78:
            readability = "readable"
        elif readability_score >= 0.48:
            readability = "occluded"
        else:
            readability = "unreadable"

        specie_conf = min(0.95, max(0.35, 0.40 + 0.55 * det.score))
        if det.score >= 0.68:
            specie = "correct"
        elif det.score >= 0.40:
            specie = "unsure"
        else:
            specie = "incorrect"

        if det.y < 0.30:
            behavior = "flying"
            behavior_conf = 0.72
        elif det.h > 0.16:
            behavior = "resting"
            behavior_conf = min(0.92, 0.55 + 0.35 * det.score)
        elif det.w > 0.18:
            behavior = "foraging"
            behavior_conf = 0.66
        else:
            behavior = "preening"
            behavior_conf = 0.60

        aspect_ratio = det.w / max(det.h, 1e-6)
        if behavior == "resting" and aspect_ratio > 2.3:
            behavior = "backresting"
            behavior_conf = min(0.9, behavior_conf + 0.08)

        if behavior in {"flying", "display"}:
            substrate = "air"
            substrate_conf = 0.80
        elif det.y + det.h > 0.62:
            substrate = "bare_ground"
            substrate_conf = min(0.90, 0.58 + 0.28 * det.score)
        else:
            substrate = "water"
            substrate_conf = min(0.90, 0.56 + 0.30 * det.score)

        if readability == "unreadable" or specie == "incorrect":
            behavior = "unsure"
            behavior_conf = 0.50
            substrate = "unsure"
            substrate_conf = 0.50
            stance = "unsure"
            stance_conf = 0.50
        elif behavior in {"resting", "backresting"} and substrate in {"bare_ground", "water", "unsure"}:
            if det.score >= 0.70:
                stance = "bipedal"
            elif det.score >= 0.52:
                stance = "unipedal"
            else:
                stance = "unsure"
            stance_conf = min(0.90, 0.52 + 0.34 * det.score)
        else:
            stance = "unsure"
            stance_conf = 0.55

        behavior, behavior_conf = self._coerce_supported_label("behavior", behavior, behavior_conf)
        substrate, substrate_conf = self._coerce_supported_label("substrate", substrate, substrate_conf)
        stance, stance_conf = self._coerce_supported_label("stance", stance, stance_conf)

        return AttributePrediction(
            readability=readability,
            readability_conf=float(readability_score),
            specie=specie,
            specie_conf=float(specie_conf),
            behavior=behavior,
            behavior_conf=float(behavior_conf),
            substrate=substrate,
            substrate_conf=float(substrate_conf),
            stance=stance,
            stance_conf=float(stance_conf),
        )

    def _predict_artifact(
        self,
        detections: list[Detection],
        *,
        image_path: Path,
    ) -> list[AttributePrediction]:
        if self.artifact is None:
            return [self._heuristic(det) for det in detections]

        with Image.open(image_path).convert("RGB") as image:
            if not detections:
                return []
            label_by_idx = [{head: self._fallback_label(head) for head in HEADS} for _ in detections]
            conf_by_idx = [{head: 0.0 for head in HEADS} for _ in detections]
            for member in self.artifact.members:
                transform = self.member_transforms[member.name]
                crops = [transform(self._crop_box(image, det)) for det in detections]
                x = torch.stack(crops, dim=0).to(self.device)
                with torch.no_grad():
                    logits = member.model(x)
                for idx in range(len(detections)):
                    for head in member.inference_heads:
                        label, score = decode_head_logits(
                            logits[head][idx],
                            head,
                            id_to_label=member.id_to_label,
                            supported_labels=member.supported_labels,
                        )
                        label_by_idx[idx][head] = label
                        conf_by_idx[idx][head] = score

        output: list[AttributePrediction] = []
        for idx in range(len(detections)):
            predictions = dict(label_by_idx[idx])
            confidences = dict(conf_by_idx[idx])
            predictions, confidences = apply_prediction_guards(
                predictions=predictions,
                confidences=confidences,
                id_to_label=self.id_to_label,
            )
            output.append(
                AttributePrediction(
                    readability=predictions["readability"],
                    readability_conf=float(confidences["readability"]),
                    specie=predictions["specie"],
                    specie_conf=float(confidences["specie"]),
                    behavior=predictions["behavior"],
                    behavior_conf=float(confidences["behavior"]),
                    substrate=predictions["substrate"],
                    substrate_conf=float(confidences["substrate"]),
                    stance=predictions["stance"],
                    stance_conf=float(confidences["stance"]),
                )
            )
        return output

    def predict(self, detections: list[Detection], *, image_path: Path | None = None) -> list[AttributePrediction]:
        if self.artifact is not None and image_path and image_path.exists():
            try:
                return self._predict_artifact(detections, image_path=image_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Model B artifact inference failed, falling back to heuristic: %s", exc)
        return [self._heuristic(det) for det in detections]
