"""Purpose: Load Model A and turn detector outputs into the normalized detection objects used downstream"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO

logger = logging.getLogger("birdsys.ml_backend.model_a")


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    label: str = "Bird"


class YoloDetector:
    def __init__(
        self,
        *,
        weights: Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1280,
        max_det: int = 300,
        device: str = "auto",
    ) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = self._resolve_device(device)
        self.model = YOLO(str(self.weights))
        logger.info(
            "Loaded YOLO detector weights=%s device=%s conf=%.3f iou=%.3f imgsz=%d max_det=%d",
            self.weights,
            self.device,
            self.conf,
            self.iou,
            self.imgsz,
            self.max_det,
        )

    @staticmethod
    def _resolve_device(requested: str) -> str:
        req = (requested or "auto").strip().lower()
        if req != "auto":
            if req == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                logger.warning("MODEL_A_DEVICE=mps requested but MPS is unavailable; falling back to CPU")
                return "cpu"

            if req == "cpu":
                return "cpu"

            if req == "cuda" or req.isdigit() or req.startswith("cuda:"):
                if torch.cuda.is_available():
                    return req
                logger.warning("MODEL_A_DEVICE=%s requested but CUDA is unavailable; falling back to CPU", req)
                return "cpu"

            return req

        if torch.cuda.is_available():
            return "0"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def predict(self, image_path: Path) -> list[Detection]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        r = results[0]
        if r.boxes is None or r.boxes.xyxy is None:
            return []

        h, w = r.orig_shape
        boxes = r.boxes.xyxy.tolist()
        confs = r.boxes.conf.tolist() if r.boxes.conf is not None else [0.0] * len(boxes)

        output: list[Detection] = []
        for xyxy, conf in zip(boxes, confs):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            bw = max(0.0, (x2 - x1) / w)
            bh = max(0.0, (y2 - y1) / h)
            bx = max(0.0, min(1.0, x1 / w))
            by = max(0.0, min(1.0, y1 / h))
            output.append(
                Detection(
                    x=bx,
                    y=by,
                    w=min(1.0, bw),
                    h=min(1.0, bh),
                    score=float(conf),
                )
            )

        return output
