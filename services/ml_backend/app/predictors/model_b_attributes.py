from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model_a_yolo import Detection


@dataclass(frozen=True)
class AttributePrediction:
    readability: str
    readability_conf: float
    activity: str
    activity_conf: float
    support: str
    support_conf: float
    legs: str
    legs_conf: float
    resting_back: str
    resting_back_conf: float


class AttributePredictor:
    """
    Model B inference adapter.

    For now this provides deterministic heuristic predictions so the backend can
    prefill Label Studio fields even before a trained checkpoint is available.
    """

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path

    def _heuristic(self, det: Detection) -> AttributePrediction:
        readable_conf = min(0.99, max(0.55, 0.55 + 0.35 * det.score))
        readable = "readable" if readable_conf >= 0.5 else "unreadable"

        standing_conf = min(0.95, max(0.10, det.h * 0.9 + 0.15))
        if standing_conf >= 0.45:
            activity = "standing"
            activity_conf = standing_conf
        else:
            activity = "flying"
            activity_conf = 1.0 - standing_conf

        if activity == "standing":
            support = "ground" if det.y + det.h > 0.55 else "water"
            support_conf = min(0.9, 0.55 + det.score * 0.25)
            legs = "two" if det.score >= 0.55 else "unsure"
            legs_conf = min(0.9, 0.50 + det.score * 0.35)
            resting_back = "no"
            resting_back_conf = min(0.95, 0.60 + det.score * 0.30)
        else:
            support = "air"
            support_conf = 0.75
            legs = "unsure"
            legs_conf = 0.5
            resting_back = "no"
            resting_back_conf = 0.5

        if readable == "unreadable":
            activity = "standing"
            support = "ground"
            legs = "unsure"
            resting_back = "no"
            activity_conf = 0.5
            support_conf = 0.5
            legs_conf = 0.5
            resting_back_conf = 0.5

        return AttributePrediction(
            readability=readable,
            readability_conf=float(readable_conf),
            activity=activity,
            activity_conf=float(activity_conf),
            support=support,
            support_conf=float(support_conf),
            legs=legs,
            legs_conf=float(legs_conf),
            resting_back=resting_back,
            resting_back_conf=float(resting_back_conf),
        )

    def predict(self, detections: list[Detection]) -> list[AttributePrediction]:
        return [self._heuristic(det) for det in detections]
