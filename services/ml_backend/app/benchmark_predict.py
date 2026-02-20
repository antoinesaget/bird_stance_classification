from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import random
import time
import urllib.request
from statistics import mean

from services.ml_backend.app.response_contract import extract_predictions


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark /predict latency from inside the ml-backend container"
    )
    parser.add_argument("--predict-url", default="http://127.0.0.1:9090/predict")
    parser.add_argument("--images-dir", default="/data/birds_project/raw_images/scolop2")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", default="/data/birds_project/derived/benchmarks/ml_backend")
    return parser.parse_args()


def iter_images(images_dir: pathlib.Path):
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def pick_images(
    all_images: list[pathlib.Path],
    *,
    samples: int,
    mode: str,
    seed: int,
) -> list[pathlib.Path]:
    if samples <= 0 or samples >= len(all_images):
        if mode == "random":
            rng = random.Random(seed)
            shuffled = list(all_images)
            rng.shuffle(shuffled)
            return shuffled
        return all_images

    if mode == "first":
        return all_images[:samples]

    rng = random.Random(seed)
    return rng.sample(all_images, samples)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def post_predict(
    *,
    url: str,
    task_id: int,
    image_path: str,
    timeout_seconds: float,
) -> tuple[int, dict]:
    payload = {"tasks": [{"id": task_id, "data": {"image": image_path}}]}
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return int(resp.status), body


def main() -> int:
    args = parse_args()

    images_dir = pathlib.Path(args.images_dir).resolve()
    output_root = pathlib.Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(images_dir))
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    selected = pick_images(
        images,
        samples=args.samples,
        mode=args.sample_mode,
        seed=args.seed,
    )

    rows: list[dict[str, object]] = []
    latencies_ms: list[float] = []

    for idx, image in enumerate(selected, start=1):
        started = time.perf_counter()
        status = 0
        has_error = True
        result_count = 0
        score = 0.0
        error_text = ""
        try:
            status, body = post_predict(
                url=args.predict_url,
                task_id=idx,
                image_path=str(image),
                timeout_seconds=args.timeout_seconds,
            )
            predictions = extract_predictions(body)
            pred = predictions[0] if predictions else {}
            has_error = bool("error" in pred)
            result_count = int(len(pred.get("result") or []))
            score = float(pred.get("score") or 0.0)
            if has_error:
                error_text = str(pred.get("error"))
        except Exception as exc:  # noqa: BLE001
            status = 0
            has_error = True
            error_text = str(exc)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        rows.append(
            {
                "task_id": idx,
                "image_id": image.stem,
                "image_path": str(image),
                "status_code": status,
                "latency_ms": round(elapsed_ms, 4),
                "result_count": result_count,
                "has_error": has_error,
                "score": round(score, 6),
                "error": error_text,
            }
        )

    sorted_lat = sorted(latencies_ms)
    success_count = sum(1 for r in rows if not r["has_error"] and r["status_code"] == 200)
    summary = {
        "generated_at": dt.datetime.now().isoformat(),
        "predict_url": args.predict_url,
        "images_dir": str(images_dir),
        "samples_requested": args.samples,
        "samples_run": len(rows),
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "success_count": success_count,
        "error_count": len(rows) - success_count,
        "latency_ms": {
            "min": round(min(sorted_lat), 4) if sorted_lat else 0.0,
            "mean": round(mean(sorted_lat), 4) if sorted_lat else 0.0,
            "p50": round(percentile(sorted_lat, 0.50), 4) if sorted_lat else 0.0,
            "p95": round(percentile(sorted_lat, 0.95), 4) if sorted_lat else 0.0,
            "max": round(max(sorted_lat), 4) if sorted_lat else 0.0,
        },
    }

    run_dir = output_root / f"predict_latency_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    samples_csv = run_dir / "latency_samples.csv"
    report_json = run_dir / "latency_report.json"

    with samples_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "image_id",
                "image_path",
                "status_code",
                "latency_ms",
                "result_count",
                "has_error",
                "score",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"samples_run={len(rows)}")
    print(f"success_count={success_count}")
    print(f"latency_report={report_json}")
    print(f"latency_samples={samples_csv}")
    print(f"p95_ms={summary['latency_ms']['p95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
