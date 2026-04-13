from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from birdsys.core.models import ImageStatusModel

from .model_a_yolo import Detection

LOGGER = logging.getLogger("birdsys.ml_backend.model_c")


class ImageStatusPredictor:
    """
    Model C inference adapter.

    Returns a probability for "has_usable_birds".
    """

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = self._pick_device()
        self.model: ImageStatusModel | None = None
        self.image_size = 224
        self.has_usable_idx = 1

        if checkpoint_path and checkpoint_path.exists():
            try:
                payload = torch.load(checkpoint_path, map_location="cpu")
                backbone = str(payload.get("backbone") or "convnext_tiny")
                self.image_size = int(payload.get("image_size") or 224)

                label_map = payload.get("label_map") or {}
                if isinstance(label_map, dict) and "has_usable_birds" in label_map:
                    self.has_usable_idx = int(label_map["has_usable_birds"])

                model = ImageStatusModel(backbone_name=backbone, pretrained=False)
                model.load_state_dict(payload["model_state"], strict=True)
                model.eval()
                self.model = model.to(self.device)
                LOGGER.info(
                    "Loaded Model C checkpoint=%s backbone=%s device=%s image_size=%d has_usable_idx=%d",
                    checkpoint_path,
                    backbone,
                    self.device,
                    self.image_size,
                    self.has_usable_idx,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to load Model C checkpoint=%s: %s", checkpoint_path, exc)
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

    def _predict_has_usable_prob_checkpoint(self, *, image_path: Path) -> float:
        if self.model is None:
            raise RuntimeError("Model C checkpoint is not loaded")
        with Image.open(image_path).convert("RGB") as image:
            x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        if self.has_usable_idx >= len(probs):
            return float(probs[-1].item())
        return float(probs[self.has_usable_idx].item())

    def predict_has_usable_prob(self, detections: list[Detection], *, image_path: Path | None = None) -> float:
        if self.model is not None and image_path and image_path.exists():
            try:
                return self._predict_has_usable_prob_checkpoint(image_path=image_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Model C checkpoint inference failed, falling back to heuristic: %s", exc)
        if not detections:
            return 0.08
        max_conf = max(det.score for det in detections)
        dense_bonus = min(0.2, 0.04 * len(detections))
        return min(0.99, 0.20 + 0.65 * max_conf + dense_bonus)

    def predict_label(
        self,
        detections: list[Detection],
        *,
        image_path: Path | None = None,
    ) -> tuple[str, float]:
        p = self.predict_has_usable_prob(detections, image_path=image_path)
        if p >= 0.5:
            return "has_usable_birds", p
        return "no_usable_birds", 1.0 - p
