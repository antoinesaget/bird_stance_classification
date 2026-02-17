from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


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
        device: str = "cpu",
    ) -> None:
        self.weights = Path(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = device
        self.model = YOLO(str(self.weights))

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
