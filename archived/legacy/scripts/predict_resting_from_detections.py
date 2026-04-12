#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from birdsys.paths import ensure_layout
from services.ml_backend.app.predictors.model_a_yolo import Detection
from services.ml_backend.app.predictors.model_b_attributes import AttributePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Model B on extracted detections and build resting-only tasks.json"
    )
    parser.add_argument("--data-root", default=os.getenv("BIRDS_DATA_ROOT", "/data/birds_project"))
    parser.add_argument("--detections-parquet", required=True, help="Path to infer_batch predictions.parquet")
    parser.add_argument("--model-b-checkpoint", required=True, help="Path to trained Model B checkpoint.pt")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Default: data_root/derived/model_b_infer/run_YYYYMMDD_HHMMSS",
    )
    parser.add_argument("--tasks-json", default="", help="Optional explicit path for output tasks.json")
    parser.add_argument(
        "--behaviors",
        default="resting",
        help="Comma-separated behavior labels to keep (default: resting)",
    )
    parser.add_argument(
        "--ls-local-files-prefix",
        default="/data/local-files/?d=",
        help="Label Studio local-files URL prefix",
    )
    parser.add_argument(
        "--export-crops",
        action="store_true",
        help="Also write extracted bird crops for each detection",
    )
    parser.add_argument("--crop-margin", type=float, default=1.2)
    parser.add_argument("--progress-every", type=int, default=100, help="Emit running metrics every N images")
    return parser.parse_args()


def parse_behavior_set(raw: str) -> set[str]:
    parts = [item.strip() for item in raw.split(",")]
    return {item for item in parts if item}


def to_ls_url(image_path: pathlib.Path, data_root: pathlib.Path, prefix: str) -> str:
    root = data_root.resolve()
    path = image_path.resolve()
    try:
        rel = path.relative_to(root)
        rel_for_ls = f"{root.name}/{rel.as_posix()}"
    except ValueError:
        rel_for_ls = path.as_posix()
    return f"{prefix}{rel_for_ls}"


def infer_site_id(image_path: pathlib.Path, data_root: pathlib.Path) -> str:
    raw_root = (data_root / "raw_images").resolve()
    path = image_path.resolve()
    try:
        rel = path.relative_to(raw_root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass
    return image_path.parent.name


def clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def expanded_crop(
    *,
    img_w: int,
    img_h: int,
    x: float,
    y: float,
    w: float,
    h: float,
    margin: float,
) -> tuple[int, int, int, int]:
    x1 = x * img_w
    y1 = y * img_h
    x2 = (x + w) * img_w
    y2 = (y + h) * img_h

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, (x2 - x1) * margin)
    bh = max(1.0, (y2 - y1) * margin)

    nx1 = int(round(cx - bw / 2.0))
    ny1 = int(round(cy - bh / 2.0))
    nx2 = int(round(cx + bw / 2.0))
    ny2 = int(round(cy + bh / 2.0))

    nx1 = clip(nx1, 0, img_w - 1)
    ny1 = clip(ny1, 0, img_h - 1)
    nx2 = clip(nx2, nx1 + 1, img_w)
    ny2 = clip(ny2, ny1 + 1, img_h)
    return nx1, ny1, nx2, ny2


def safe_name(image_id: str, det_idx: int) -> str:
    return f"{image_id}__{det_idx:03d}".replace(":", "_").replace("/", "_")


def main() -> int:
    args = parse_args()

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required. Install dependencies with `uv sync --python 3.11`.") from exc

    pil_import_error = None
    if args.export_crops:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            pil_import_error = exc
            Image = None  # type: ignore[assignment]
    else:
        Image = None  # type: ignore[assignment]

    if args.export_crops and pil_import_error is not None:
        raise RuntimeError("Pillow is required for --export-crops. Install dependencies with `uv sync --python 3.11`.") from pil_import_error

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    layout = ensure_layout(data_root)

    detections_path = pathlib.Path(args.detections_parquet).expanduser().resolve()
    if not detections_path.exists():
        raise FileNotFoundError(detections_path)

    checkpoint_path = pathlib.Path(args.model_b_checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    timestamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = data_root / "derived" / "model_b_infer" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = out_dir / "crops"
    if args.export_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    keep_behaviors = parse_behavior_set(args.behaviors)
    predictor = AttributePredictor(checkpoint_path=checkpoint_path)

    if predictor.model is None:
        raise RuntimeError(
            f"Could not load Model B checkpoint at {checkpoint_path}; refusing to run heuristic fallback."
        )

    df = pd.read_parquet(detections_path)
    required_columns = {"image_id", "image_path", "detections_json"}
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in detections parquet: {missing}")

    image_rows: list[dict[str, object]] = []
    bird_rows: list[dict[str, object]] = []
    crop_rows: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []

    total_images = len(df)
    started_at = time.monotonic()
    processed_images = 0
    processed_birds = 0
    images_with_kept = 0

    for idx, row in df.iterrows():
        processed_images += 1

        image_id = str(row["image_id"])
        image_path = pathlib.Path(str(row["image_path"])).expanduser().resolve()
        if not image_path.exists():
            image_rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "detections_count": 0,
                    "predictions_count": 0,
                    "kept_behaviors_count": 0,
                    "status": "missing_image",
                }
            )
            continue

        try:
            raw_detections = json.loads(str(row.get("detections_json") or "[]"))
        except json.JSONDecodeError:
            raw_detections = []

        detections: list[Detection] = []
        norm_boxes: list[tuple[float, float, float, float, float]] = []
        for det in raw_detections:
            bbox = det.get("bbox") or {}
            try:
                x = float(bbox.get("x", 0.0))
                y = float(bbox.get("y", 0.0))
                w = float(bbox.get("w", 0.0))
                h = float(bbox.get("h", 0.0))
                score = float(det.get("det_conf", 0.0))
            except (TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            detections.append(Detection(x=x, y=y, w=w, h=h, score=score))
            norm_boxes.append((x, y, w, h, score))

        attrs = predictor.predict(detections, image_path=image_path)
        kept_count = 0
        processed_birds += len(detections)

        if args.export_crops and detections:
            assert Image is not None
            with Image.open(image_path).convert("RGB") as img:
                img_w, img_h = img.size
                for det_idx, (x, y, w, h, score) in enumerate(norm_boxes):
                    x1, y1, x2, y2 = expanded_crop(
                        img_w=img_w,
                        img_h=img_h,
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        margin=args.crop_margin,
                    )
                    crop = img.crop((x1, y1, x2, y2))
                    crop_path = crops_dir / f"{safe_name(image_id, det_idx)}.jpg"
                    crop.save(crop_path, format="JPEG", quality=95)
                    crop_rows.append(
                        {
                            "image_id": image_id,
                            "detection_index": det_idx,
                            "crop_path": str(crop_path),
                            "bbox_x": x,
                            "bbox_y": y,
                            "bbox_w": w,
                            "bbox_h": h,
                            "det_conf": score,
                        }
                    )

        for det_idx, (det, attr) in enumerate(zip(detections, attrs)):
            is_kept = attr.behavior in keep_behaviors
            if is_kept:
                kept_count += 1
            bird_rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "detection_index": det_idx,
                    "bbox_x": det.x,
                    "bbox_y": det.y,
                    "bbox_w": det.w,
                    "bbox_h": det.h,
                    "det_conf": det.score,
                    "readability": attr.readability,
                    "readability_conf": attr.readability_conf,
                    "specie": attr.specie,
                    "specie_conf": attr.specie_conf,
                    "behavior": attr.behavior,
                    "behavior_conf": attr.behavior_conf,
                    "substrate": attr.substrate,
                    "substrate_conf": attr.substrate_conf,
                    "stance": attr.stance,
                    "stance_conf": attr.stance_conf,
                    "is_kept_behavior": bool(is_kept),
                }
            )

        image_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "detections_count": len(detections),
                "predictions_count": len(attrs),
                "kept_behaviors_count": kept_count,
                "status": "ok",
            }
        )

        if kept_count > 0:
            images_with_kept += 1
            tasks.append(
                {
                    "image_id": image_id,
                    "site_id": infer_site_id(image_path, data_root),
                    "image": to_ls_url(image_path, data_root, args.ls_local_files_prefix),
                    "kept_behaviors_count": kept_count,
                    "total_detected_birds": len(detections),
                }
            )

        should_log = (
            args.progress_every > 0
            and (processed_images % args.progress_every == 0 or processed_images == total_images)
        )
        if should_log:
            elapsed = max(1e-6, time.monotonic() - started_at)
            ips = processed_images / elapsed
            bps = processed_birds / elapsed
            eta_s = (total_images - processed_images) / ips if ips > 0 else 0.0
            print(
                "progress="
                f"{processed_images}/{total_images} "
                f"birds_processed={processed_birds} "
                f"images_with_kept_behaviors={images_with_kept} "
                f"ips={ips:.3f} "
                f"birds_per_s={bps:.3f} "
                f"eta_s={eta_s:.1f}",
                flush=True,
            )

    bird_df = pd.DataFrame(bird_rows)
    if not bird_df.empty:
        bird_df = bird_df.sort_values(["image_id", "detection_index"]).reset_index(drop=True)

    image_df = pd.DataFrame(image_rows)
    if not image_df.empty:
        image_df = image_df.sort_values(["image_id"]).reset_index(drop=True)
    tasks = sorted(tasks, key=lambda t: (str(t["site_id"]), str(t["image_id"])))

    birds_out = out_dir / "bird_predictions.parquet"
    images_out = out_dir / "image_summary.parquet"
    bird_df.to_parquet(birds_out, index=False)
    image_df.to_parquet(images_out, index=False)

    crops_manifest_path = out_dir / "crops_manifest.parquet"
    if args.export_crops:
        crops_df = pd.DataFrame(crop_rows)
        if not crops_df.empty:
            crops_df = crops_df.sort_values(["image_id", "detection_index"]).reset_index(drop=True)
        crops_df.to_parquet(crops_manifest_path, index=False)

    if args.tasks_json:
        tasks_json_path = pathlib.Path(args.tasks_json).expanduser().resolve()
        tasks_json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tasks_json_path = out_dir / "resting.tasks.json"

    tasks_json_path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")

    summary = {
        "generated_at": dt.datetime.now().isoformat(),
        "detections_parquet": str(detections_path),
        "model_b_checkpoint": str(checkpoint_path),
        "images_total": int(len(image_df)),
        "images_with_kept_behaviors": int(sum(1 for row in tasks)),
        "birds_total": int(len(bird_df)),
        "kept_behaviors": sorted(keep_behaviors),
        "tasks_json": str(tasks_json_path),
        "bird_predictions": str(birds_out),
        "image_summary": str(images_out),
        "crops_manifest": str(crops_manifest_path) if args.export_crops else None,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    elapsed_total = max(1e-6, time.monotonic() - started_at)
    print(f"images_total={summary['images_total']}")
    print(f"birds_total={summary['birds_total']}")
    print(f"images_with_kept_behaviors={summary['images_with_kept_behaviors']}")
    print(f"duration_s={elapsed_total:.2f}")
    print(f"ips={summary['images_total'] / elapsed_total:.3f}")
    print(f"birds_per_s={summary['birds_total'] / elapsed_total:.3f}")
    print(f"tasks_json={tasks_json_path}")
    print(f"bird_predictions={birds_out}")
    print(f"image_summary={images_out}")
    if args.export_crops:
        print(f"crops_manifest={crops_manifest_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
