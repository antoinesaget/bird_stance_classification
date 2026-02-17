#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference for active-learning candidate scoring")
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--images-dir", default="", help="Optional image directory override")
    parser.add_argument("--metadata-parquet", default="", help="Optional metadata parquet override")
    parser.add_argument("--output-dir", default="", help="Run output dir. Default: derived/active_learning_infer/run_YYYYMMDD_HHMMSS")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--sample-mode", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-det", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-a", default=os.getenv("MODEL_A_WEIGHTS", "yolo11m.pt"))
    return parser.parse_args()


def iter_images(images_dir: pathlib.Path):
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def pick_images(all_paths: list[pathlib.Path], samples: int, mode: str, seed: int) -> list[pathlib.Path]:
    if samples <= 0 or samples >= len(all_paths):
        if mode == "random":
            rng = random.Random(seed)
            shuffled = list(all_paths)
            rng.shuffle(shuffled)
            return shuffled
        return all_paths
    if mode == "first":
        return all_paths[:samples]
    rng = random.Random(seed)
    return rng.sample(all_paths, samples)


def softmax3(a: float, b: float, c: float) -> tuple[float, float, float]:
    vals = [a, b, c]
    m = max(vals)
    exps = [pow(2.718281828, v - m) for v in vals]
    s = sum(exps)
    return exps[0] / s, exps[1] / s, exps[2] / s


def heuristic_attributes(det_conf: float, bbox_h: float) -> dict[str, object]:
    readable = min(0.99, max(0.05, 0.55 + 0.35 * det_conf))
    unreadable = 1.0 - readable

    stand_logit = 0.25 + bbox_h * 0.8
    fly_logit = 0.15 + (1.0 - bbox_h) * 0.5
    forage_logit = 0.10 + bbox_h * 0.3
    flying_p, foraging_p, standing_p = softmax3(fly_logit, forage_logit, stand_logit)

    ground_p, water_p, air_p = softmax3(0.55 + bbox_h, 0.2 + bbox_h * 0.6, 0.4 + (1.0 - bbox_h))

    one_p, two_p, unsure_p = softmax3(0.35, 0.55, 0.15 + (1.0 - det_conf))
    rest_no = min(0.95, 0.6 + 0.2 * det_conf)
    rest_yes = 1.0 - rest_no

    return {
        "readability_probs": {"readable": readable, "unreadable": unreadable},
        "activity_probs": {"flying": flying_p, "foraging": foraging_p, "standing": standing_p},
        "support_probs": {"ground": ground_p, "water": water_p, "air": air_p},
        "legs_probs": {"one": one_p, "two": two_p, "unsure": unsure_p},
        "resting_back_probs": {"yes": rest_yes, "no": rest_no},
    }


def heuristic_image_status(num_dets: int, max_conf: float) -> float:
    if num_dets <= 0:
        return 0.08
    return min(0.99, 0.25 + 0.55 * max_conf + 0.08 * min(num_dets, 4))


def main() -> int:
    args = parse_args()
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required. Install dependencies with `uv sync --python 3.11`.") from exc
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("ultralytics is required. Install dependencies with `uv sync --python 3.11`.") from exc

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    layout = ensure_layout(data_root)

    if args.images_dir:
        image_paths = list(iter_images(pathlib.Path(args.images_dir).expanduser().resolve()))
    else:
        metadata_path = pathlib.Path(args.metadata_parquet).expanduser().resolve() if args.metadata_parquet else layout.metadata / "images.parquet"
        if metadata_path.exists():
            mdf = pd.read_parquet(metadata_path)
            image_paths = [pathlib.Path(p) for p in mdf["filepath"].dropna().astype(str).tolist()]
        else:
            image_paths = list(iter_images(layout.raw_images))

    if not image_paths:
        raise RuntimeError("No images found for inference")

    selected = pick_images(image_paths, args.samples, args.sample_mode, args.seed)

    timestamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = data_root / "derived" / "active_learning_infer" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model_a)

    rows: list[dict[str, object]] = []
    for image_path in selected:
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
        dets = []
        max_conf = 0.0
        if results and results[0].boxes is not None:
            r = results[0]
            h, w = r.orig_shape
            boxes_xyxy = r.boxes.xyxy.tolist() if r.boxes.xyxy is not None else []
            confs = r.boxes.conf.tolist() if r.boxes.conf is not None else []
            for xyxy, conf in zip(boxes_xyxy, confs):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                bw = max(0.0, (x2 - x1) / w)
                bh = max(0.0, (y2 - y1) / h)
                bx = max(0.0, x1 / w)
                by = max(0.0, y1 / h)
                det_conf = float(conf)
                attrs = heuristic_attributes(det_conf, bh)
                dets.append(
                    {
                        "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
                        "det_conf": det_conf,
                        **attrs,
                    }
                )
                max_conf = max(max_conf, det_conf)

        image_status_prob = heuristic_image_status(len(dets), max_conf)
        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "detection_count": len(dets),
                "max_detection_conf": max_conf,
                "model_c_has_usable_prob": image_status_prob,
                "detections_json": json.dumps(dets, separators=(",", ":")),
            }
        )

    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    out_parquet = out_dir / "predictions.parquet"
    df.to_parquet(out_parquet, index=False)

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now().isoformat(),
                "samples": len(selected),
                "seed": args.seed,
                "model_a": args.model_a,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "max_det": args.max_det,
                "device": args.device,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"images_inferred={len(df)}")
    print(f"output={out_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
