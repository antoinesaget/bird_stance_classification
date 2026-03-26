from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from birdsys.training.models import MultiHeadAttributeModel

from .model_a_yolo import Detection

LOGGER = logging.getLogger("birdsys.ml_backend.model_b")

DEFAULT_LABELS = {
    "readability": ["readable", "occluded", "unreadable"],
    "specie": ["correct", "incorrect", "unsure"],
    "behavior": [
        "flying",
        "moving",
        "foraging",
        "resting",
        "backresting",
        "bathing",
        "calling",
        "preening",
        "display",
        "breeding",
        "other",
        "unsure",
    ],
    "substrate": ["bare_ground", "vegetation", "water", "air", "unsure"],
    "stance": ["unipedal", "bipedal", "sitting", "unsure"],
}


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

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = self._pick_device()
        self.model: MultiHeadAttributeModel | None = None
        self.image_size = 224
        self.schema_version = "annotation_schema_v2"
        self.id_to_label = {head: labels[:] for head, labels in DEFAULT_LABELS.items()}
        self.supported_labels = {head: set(labels) for head, labels in DEFAULT_LABELS.items()}
        self.train_label_counts: dict[str, dict[str, int]] = {}

        if checkpoint_path and checkpoint_path.exists():
            try:
                payload = torch.load(checkpoint_path, map_location="cpu")
                backbone = str(payload.get("backbone") or "convnextv2_small")
                self.image_size = int(payload.get("image_size") or 224)
                self.schema_version = str(payload.get("schema_version") or self.schema_version)

                label_maps = payload.get("label_maps") or {}
                self.id_to_label = self._decode_label_maps(label_maps)
                self.supported_labels = self._decode_supported_labels(payload.get("supported_labels") or {})
                self.train_label_counts = {
                    str(head): {str(label): int(count) for label, count in (counts or {}).items()}
                    for head, counts in (payload.get("train_label_counts") or {}).items()
                }

                model = MultiHeadAttributeModel(backbone_name=backbone, pretrained=False)
                model.load_state_dict(payload["model_state"], strict=True)
                model.eval()
                self.model = model.to(self.device)
                LOGGER.info(
                    "Loaded Model B checkpoint=%s backbone=%s device=%s image_size=%d schema=%s",
                    checkpoint_path,
                    backbone,
                    self.device,
                    self.image_size,
                    self.schema_version,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to load Model B checkpoint=%s: %s", checkpoint_path, exc)
                self.model = None

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _invert_map(mapping: dict[str, int], fallback: list[str]) -> list[str]:
        if not mapping:
            return fallback
        size = max(int(v) for v in mapping.values()) + 1
        out = [""] * size
        for label, idx in mapping.items():
            out[int(idx)] = str(label)
        for i, value in enumerate(out):
            if not value and i < len(fallback):
                out[i] = fallback[i]
        return out

    @classmethod
    def _decode_label_maps(cls, raw: dict) -> dict[str, list[str]]:
        out = {head: labels[:] for head, labels in DEFAULT_LABELS.items()}
        for head, fallback in DEFAULT_LABELS.items():
            values = raw.get(head)
            if isinstance(values, dict):
                out[head] = cls._invert_map(values, fallback)
        return out

    def _decode_supported_labels(self, raw: dict[str, list[str]]) -> dict[str, set[str]]:
        out = {head: set(labels) for head, labels in self.id_to_label.items()}
        for head in DEFAULT_LABELS:
            values = raw.get(head)
            if isinstance(values, list) and values:
                out[head] = {str(item) for item in values}
        return out

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
        if "unsure" in self.id_to_label.get(head, []):
            return "unsure"
        return self.id_to_label[head][0]

    def _coerce_supported_label(self, head: str, label: str, score: float) -> tuple[str, float]:
        allowed = self.supported_labels.get(head) or set()
        if not allowed or label in allowed:
            return label, score
        fallback = self._fallback_label(head)
        return fallback, min(score, 0.50)

    def _decode_head(self, logits: torch.Tensor, head: str) -> tuple[str, float]:
        probs = torch.softmax(logits, dim=0)
        score, idx = torch.max(probs, dim=0)
        labels = self.id_to_label[head]
        label = labels[int(idx)] if int(idx) < len(labels) and labels[int(idx)] else DEFAULT_LABELS[head][0]
        return self._coerce_supported_label(head, label, float(score.item()))

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

    def _predict_checkpoint(
        self,
        detections: list[Detection],
        *,
        image_path: Path,
    ) -> list[AttributePrediction]:
        if self.model is None:
            return [self._heuristic(det) for det in detections]

        with Image.open(image_path).convert("RGB") as image:
            crops = [self.transform(self._crop_box(image, det)) for det in detections]
        if not crops:
            return []

        x = torch.stack(crops, dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)

        output: list[AttributePrediction] = []
        for idx in range(len(detections)):
            readability, readability_conf = self._decode_head(logits["readability"][idx], "readability")
            specie, specie_conf = self._decode_head(logits["specie"][idx], "specie")
            behavior, behavior_conf = self._decode_head(logits["behavior"][idx], "behavior")
            substrate, substrate_conf = self._decode_head(logits["substrate"][idx], "substrate")
            stance, stance_conf = self._decode_head(logits["stance"][idx], "stance")

            if readability == "unreadable" or specie == "incorrect":
                behavior = self._fallback_label("behavior")
                behavior_conf = min(behavior_conf, 0.50)
                substrate = self._fallback_label("substrate")
                substrate_conf = min(substrate_conf, 0.50)
                stance = self._fallback_label("stance")
                stance_conf = min(stance_conf, 0.50)
            elif not (behavior in {"resting", "backresting"} and substrate in {"bare_ground", "water", "unsure"}):
                stance = self._fallback_label("stance")
                stance_conf = min(stance_conf, 0.55)

            output.append(
                AttributePrediction(
                    readability=readability,
                    readability_conf=readability_conf,
                    specie=specie,
                    specie_conf=specie_conf,
                    behavior=behavior,
                    behavior_conf=behavior_conf,
                    substrate=substrate,
                    substrate_conf=substrate_conf,
                    stance=stance,
                    stance_conf=stance_conf,
                )
            )
        return output

    def predict(self, detections: list[Detection], *, image_path: Path | None = None) -> list[AttributePrediction]:
        if self.model is not None and image_path and image_path.exists():
            try:
                return self._predict_checkpoint(detections, image_path=image_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Model B checkpoint inference failed, falling back to heuristic: %s", exc)
        return [self._heuristic(det) for det in detections]
