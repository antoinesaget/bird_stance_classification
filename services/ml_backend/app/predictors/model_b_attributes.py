from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model_a_yolo import Detection


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
    legs: str
    legs_conf: float


class AttributePredictor:
    """
    Model B inference adapter.

    For now this provides deterministic heuristic predictions so the backend can
    prefill Label Studio fields even before a trained checkpoint is available.
    """

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path

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
            substrate = "ground"
            substrate_conf = min(0.90, 0.58 + 0.28 * det.score)
        else:
            substrate = "water"
            substrate_conf = min(0.90, 0.56 + 0.30 * det.score)

        if readability == "unreadable" or specie == "incorrect":
            behavior = "unsure"
            behavior_conf = 0.50
            substrate = "unsure"
            substrate_conf = 0.50
            legs = "unsure"
            legs_conf = 0.5
        elif behavior in {"resting", "backresting"} and substrate in {"ground", "water"}:
            if det.score >= 0.70:
                legs = "two"
            elif det.score >= 0.52:
                legs = "one"
            else:
                legs = "unsure"
            legs_conf = min(0.90, 0.52 + 0.34 * det.score)
        else:
            legs = "unsure"
            legs_conf = 0.55

        return AttributePrediction(
            readability=readability,
            readability_conf=float(readability_score),
            specie=specie,
            specie_conf=float(specie_conf),
            behavior=behavior,
            behavior_conf=float(behavior_conf),
            substrate=substrate,
            substrate_conf=float(substrate_conf),
            legs=legs,
            legs_conf=float(legs_conf),
        )

    def predict(self, detections: list[Detection]) -> list[AttributePrediction]:
        return [self._heuristic(det) for det in detections]
