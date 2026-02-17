from __future__ import annotations

from pathlib import Path

from .model_a_yolo import Detection


class ImageStatusPredictor:
    """
    Model C inference adapter.

    Returns a probability for "has_usable_birds".
    """

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path

    def predict_has_usable_prob(self, detections: list[Detection]) -> float:
        if not detections:
            return 0.08
        max_conf = max(det.score for det in detections)
        dense_bonus = min(0.2, 0.04 * len(detections))
        return min(0.99, 0.20 + 0.65 * max_conf + dense_bonus)

    def predict_label(self, detections: list[Detection]) -> tuple[str, float]:
        p = self.predict_has_usable_prob(detections)
        if p >= 0.5:
            return "has_usable_birds", p
        return "no_usable_birds", 1.0 - p
