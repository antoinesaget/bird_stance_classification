#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select active-learning batch from infer_batch outputs")
    parser.add_argument("--predictions", required=True, help="Path to infer_batch predictions.parquet")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-slice-pct", type=float, default=0.1)
    parser.add_argument("--out-dir", default="", help="Default: sibling output dir next to predictions")
    return parser.parse_args()


def diversity_score(image_id: str) -> float:
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0


def row_scores(row: pd.Series) -> dict[str, float]:
    try:
        detections = json.loads(row.get("detections_json") or "[]")
    except Exception:
        detections = []

    read_unc = 0.0
    resting_low_stance = 0.0
    disagreement = 0.0

    for det in detections:
        readability_probs = det.get("readability_probs") or {}
        read_p = float(readability_probs.get("readable", 0.4)) + 0.5 * float(readability_probs.get("occluded", 0.2))
        behavior_probs = det.get("behavior_probs") or {}
        resting_p = float(behavior_probs.get("resting", 0.0)) + float(behavior_probs.get("backresting", 0.0))
        substrate_probs = det.get("substrate_probs") or {}
        ground_water_p = float(substrate_probs.get("bare_ground", 0.0)) + float(substrate_probs.get("water", 0.0))
        stance_probs = det.get("stance_probs") or det.get("legs_probs") or {}
        stance_best = max(
            float(stance_probs.get("unipedal", stance_probs.get("one", 0.0))),
            float(stance_probs.get("bipedal", stance_probs.get("two", 0.0))),
            float(stance_probs.get("unsure", 0.0)),
            float(stance_probs.get("sitting", 0.0)),
        )

        read_unc = max(read_unc, 1.0 - abs(read_p - 0.5) * 2.0)
        resting_low_stance = max(resting_low_stance, read_p * resting_p * ground_water_p * (1.0 - stance_best))
        disagreement = max(
            disagreement,
            abs(
                resting_p
                - max(
                    float(stance_probs.get("unipedal", stance_probs.get("one", 0.0))),
                    float(stance_probs.get("bipedal", stance_probs.get("two", 0.0))),
                    float(stance_probs.get("sitting", 0.0)),
                )
            ),
        )

    diversity = diversity_score(str(row["image_id"]))
    return {
        "readability_uncertainty": read_unc,
        "resting_low_stance_conf": resting_low_stance,
        "disagreement": disagreement,
        "diversity": diversity,
    }


def main() -> int:
    args = parse_args()
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required. Install dependencies with `uv sync --python 3.11`.") from exc

    predictions_path = pathlib.Path(args.predictions).expanduser().resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    df = pd.read_parquet(predictions_path)
    if df.empty:
        raise RuntimeError("predictions parquet is empty")

    score_rows = []
    for _, row in df.iterrows():
        parts = row_scores(row)
        total = (
            0.40 * parts["readability_uncertainty"]
            + 0.35 * parts["resting_low_stance_conf"]
            + 0.15 * parts["disagreement"]
            + 0.10 * parts["diversity"]
        )
        score_rows.append({
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "detection_count": int(row.get("detection_count", 0)),
            "max_detection_conf": float(row.get("max_detection_conf", 0.0)),
            **parts,
            "total_score": float(total),
        })

    score_df = pd.DataFrame(score_rows).sort_values(["total_score", "image_id"], ascending=[False, True]).reset_index(drop=True)

    batch_size = min(args.batch_size, len(score_df))
    random_count = int(round(batch_size * args.random_slice_pct))
    random_count = max(0, min(random_count, batch_size))
    top_count = batch_size - random_count

    top_df = score_df.head(top_count).copy()
    remaining = score_df.iloc[top_count:].copy()

    random_df = remaining.sample(n=random_count, random_state=args.seed) if random_count > 0 and not remaining.empty else remaining.head(0)

    selected_df = pd.concat([top_df, random_df], ignore_index=True)
    selected_df = selected_df.sort_values(["total_score", "image_id"], ascending=[False, True]).reset_index(drop=True)
    selected_df["rank"] = selected_df.index + 1

    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = predictions_path.parent / f"selection_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_csv = out_dir / "selected_images.csv"
    selected_df.to_csv(selected_csv, index=False)

    report = {
        "generated_at": dt.datetime.now().isoformat(),
        "predictions": str(predictions_path),
        "seed": args.seed,
        "batch_size": batch_size,
        "top_count": top_count,
        "random_count": random_count,
        "weights": {
            "readability_uncertainty": 0.40,
            "resting_low_stance_conf": 0.35,
            "disagreement": 0.15,
            "diversity": 0.10,
        },
        "summary": {
            "selected_mean_score": float(selected_df["total_score"].mean()) if not selected_df.empty else 0.0,
            "selected_min_score": float(selected_df["total_score"].min()) if not selected_df.empty else 0.0,
            "selected_max_score": float(selected_df["total_score"].max()) if not selected_df.empty else 0.0,
        },
    }
    (out_dir / "selection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"selected_count={len(selected_df)}")
    print(f"selected_csv={selected_csv}")
    print(f"report={out_dir / 'selection_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
