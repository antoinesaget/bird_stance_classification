#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import time

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
    parser.add_argument("--progress-every", type=int, default=100, help="Emit running metrics every N images")
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


def softmax(vals: list[float]) -> list[float]:
    m = max(vals)
    exps = [pow(2.718281828, v - m) for v in vals]
    s = sum(exps)
    return [v / s for v in exps]


def heuristic_attributes(det_conf: float, bbox_h: float) -> dict[str, object]:
    readable = min(0.95, max(0.05, 0.35 + 0.55 * det_conf))
    unreadable = max(0.01, 1.0 - readable - 0.15)
    occluded = max(0.01, 1.0 - readable - unreadable)
    total_r = readable + occluded + unreadable
    readable, occluded, unreadable = readable / total_r, occluded / total_r, unreadable / total_r

    correct = min(0.95, max(0.10, 0.35 + 0.55 * det_conf))
    incorrect = max(0.05, 0.15 - 0.10 * det_conf)
    unsure_specie = max(0.05, 1.0 - correct - incorrect)
    total_s = correct + incorrect + unsure_specie
    correct, incorrect, unsure_specie = correct / total_s, incorrect / total_s, unsure_specie / total_s

    behavior_probs = softmax(
        [
            0.25 + (1.0 - bbox_h) * 0.6,  # flying
            0.20 + bbox_h * 0.35,  # foraging
            0.25 + bbox_h * 0.75,  # resting
            0.15 + bbox_h * 0.50,  # backresting
            0.20 + bbox_h * 0.25,  # preening
            0.05 + (1.0 - bbox_h) * 0.20,  # display
            0.10 + (1.0 - det_conf) * 0.40,  # unsure
        ]
    )

    substrate_probs = softmax(
        [
            0.55 + bbox_h,  # ground
            0.2 + bbox_h * 0.6,  # water
            0.4 + (1.0 - bbox_h),  # air
            0.15 + (1.0 - det_conf) * 0.5,  # unsure
        ]
    )

    one_p, two_p, unsure_p, sitting_p = softmax([0.35, 0.55, 0.15 + (1.0 - det_conf), 0.25])

    return {
        "readability_probs": {"readable": readable, "occluded": occluded, "unreadable": unreadable},
        "specie_probs": {"correct": correct, "incorrect": incorrect, "unsure": unsure_specie},
        "behavior_probs": {
            "flying": behavior_probs[0],
            "foraging": behavior_probs[1],
            "resting": behavior_probs[2],
            "backresting": behavior_probs[3],
            "preening": behavior_probs[4],
            "display": behavior_probs[5],
            "unsure": behavior_probs[6],
        },
        "substrate_probs": {
            "ground": substrate_probs[0],
            "water": substrate_probs[1],
            "air": substrate_probs[2],
            "unsure": substrate_probs[3],
        },
        "legs_probs": {"one": one_p, "two": two_p, "unsure": unsure_p, "sitting": sitting_p},
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
    total_images = len(selected)
    images_with_dets = 0
    detections_total = 0
    started_at = time.monotonic()

    for idx, image_path in enumerate(selected, start=1):
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
        detections_total += len(dets)
        if dets:
            images_with_dets += 1

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

        should_log = (
            args.progress_every > 0
            and (idx % args.progress_every == 0 or idx == total_images)
        )
        if should_log:
            elapsed = max(1e-6, time.monotonic() - started_at)
            ips = idx / elapsed
            eta_s = (total_images - idx) / ips if ips > 0 else 0.0
            print(
                "progress="
                f"{idx}/{total_images} "
                f"images_with_detections={images_with_dets} "
                f"detections_total={detections_total} "
                f"ips={ips:.3f} "
                f"eta_s={eta_s:.1f}",
                flush=True,
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

    elapsed_total = max(1e-6, time.monotonic() - started_at)
    print(f"images_inferred={len(df)}")
    print(f"images_with_detections={images_with_dets}")
    print(f"detections_total={detections_total}")
    print(f"duration_s={elapsed_total:.2f}")
    print(f"ips={len(df) / elapsed_total:.3f}")
    print(f"output={out_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
