#!/usr/bin/env python3
"""Blind adaptive pairwise model comparison app (1x2).

Controls:
- Q: left is better
- D: right is better
- S: both correct
- Z: both wrong
- N: finish now
- Esc: quit

The app preloads upcoming comparisons in a background queue so inference for
the next screens can run while you score the current one.
After each non-tie decision, it can enqueue a same-image follow-up that
compares the winner against another challenger.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import random
import re
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    variant: str
    backend: str = "ultralytics"
    checkpoints: tuple[str, ...] = ()
    hf_model_id: str | None = None
    prompt: str = "bird."
    fallback_checkpoints: tuple[str, ...] = ()
    note: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.family} {self.variant}".strip()


@dataclass
class LoadedModel:
    spec: ModelSpec
    backend: str
    model_obj: object
    resolved_source: str
    processor_obj: object | None = None
    runtime_device: str | None = None
    resolution_note: str = ""


@dataclass
class VoteStats:
    wins: float = 0.0
    losses: float = 0.0
    ties: int = 0
    correct: int = 0
    judged: int = 0

    @property
    def votes(self) -> float:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.votes <= 0:
            return 0.0
        return self.wins / self.votes

    @property
    def correct_rate(self) -> float:
        if self.judged <= 0:
            return 0.0
        return self.correct / self.judged


@dataclass
class TimingStats:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    values_ms: list[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        if value_ms <= 0:
            return
        self.count += 1
        self.total_ms += value_ms
        self.min_ms = min(self.min_ms, value_ms)
        self.max_ms = max(self.max_ms, value_ms)
        self.values_ms.append(value_ms)

    @property
    def mean_ms(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total_ms / self.count

    @property
    def p50_ms(self) -> float:
        if not self.values_ms:
            return 0.0
        return float(np.percentile(self.values_ms, 50))

    @property
    def p90_ms(self) -> float:
        if not self.values_ms:
            return 0.0
        return float(np.percentile(self.values_ms, 90))


@dataclass
class PredictionMetrics:
    num_boxes: int = 0
    mean_conf: float = 0.0
    max_conf: float = 0.0
    inference_ms: float = 0.0
    error: str = ""


@dataclass
class PairScreen:
    screen_index: int
    image_path: pathlib.Path
    left_model_key: str
    right_model_key: str
    left_frame: np.ndarray
    right_frame: np.ndarray
    left_metrics: PredictionMetrics
    right_metrics: PredictionMetrics
    same_image_depth: int = 0


MODEL_CATALOG: dict[str, ModelSpec] = {
    "yolov10n": ModelSpec("yolov10n", "YOLOv10", "n", checkpoints=("yolov10n.pt",)),
    "yolov10s": ModelSpec("yolov10s", "YOLOv10", "s", checkpoints=("yolov10s.pt",)),
    "yolov10m": ModelSpec("yolov10m", "YOLOv10", "m", checkpoints=("yolov10m.pt",)),
    "yolov10l": ModelSpec("yolov10l", "YOLOv10", "l", checkpoints=("yolov10l.pt",)),
    "yolov10x": ModelSpec("yolov10x", "YOLOv10", "x", checkpoints=("yolov10x.pt",)),
    "yolo11n": ModelSpec("yolo11n", "YOLO11", "n", checkpoints=("yolo11n.pt",)),
    "yolo11s": ModelSpec("yolo11s", "YOLO11", "s", checkpoints=("yolo11s.pt",)),
    "yolo11m": ModelSpec("yolo11m", "YOLO11", "m", checkpoints=("yolo11m.pt",)),
    "yolo11l": ModelSpec("yolo11l", "YOLO11", "l", checkpoints=("yolo11l.pt",)),
    "yolo11x": ModelSpec("yolo11x", "YOLO11", "x", checkpoints=("yolo11x.pt",)),
    "yolo26n": ModelSpec("yolo26n", "YOLO26", "n", checkpoints=("yolo26n.pt",)),
    "yolo26s": ModelSpec("yolo26s", "YOLO26", "s", checkpoints=("yolo26s.pt",)),
    "yolo26m": ModelSpec("yolo26m", "YOLO26", "m", checkpoints=("yolo26m.pt",)),
    "yolo26l": ModelSpec("yolo26l", "YOLO26", "l", checkpoints=("yolo26l.pt",)),
    "yolo26x": ModelSpec("yolo26x", "YOLO26", "x", checkpoints=("yolo26x.pt",)),
    "rtdetr-l": ModelSpec("rtdetr-l", "RT-DETR", "HGNetv2-L", checkpoints=("rtdetr-l.pt",)),
    "rtdetr-x": ModelSpec("rtdetr-x", "RT-DETR", "HGNetv2-X", checkpoints=("rtdetr-x.pt",)),
    "rtdetrv2-l": ModelSpec(
        "rtdetrv2-l",
        "RT-DETRv2",
        "L",
        checkpoints=("rtdetrv2-l.pt", "rtdetr-l.pt"),
        note="Fallback to rtdetr-l.",
    ),
    "rtdetrv2-x": ModelSpec(
        "rtdetrv2-x",
        "RT-DETRv2",
        "X",
        checkpoints=("rtdetrv2-x.pt", "rtdetr-x.pt"),
        note="Fallback to rtdetr-x.",
    ),
    "rtdetrv3-r18": ModelSpec(
        "rtdetrv3-r18",
        "RT-DETRv3",
        "R18",
        checkpoints=("rtdetrv3-r18.pt", "rtdetr-l.pt"),
        note="Fallback to rtdetr-l.",
    ),
    "rtdetrv3-r50": ModelSpec(
        "rtdetrv3-r50",
        "RT-DETRv3",
        "R50",
        checkpoints=("rtdetrv3-r50.pt", "rtdetr-l.pt"),
        note="Fallback to rtdetr-l.",
    ),
    "rtdetrv3-r101": ModelSpec(
        "rtdetrv3-r101",
        "RT-DETRv3",
        "R101",
        checkpoints=("rtdetrv3-r101.pt", "rtdetr-x.pt"),
        note="Fallback to rtdetr-x.",
    ),
    "rfdetr-nano": ModelSpec(
        "rfdetr-nano",
        "RF-DETR",
        "nano",
        checkpoints=("rfdetr-nano.pt", "yolov10n.pt"),
        note="Proxy fallback to yolov10n.",
    ),
    "rfdetr-small": ModelSpec(
        "rfdetr-small",
        "RF-DETR",
        "small",
        checkpoints=("rfdetr-small.pt", "yolo11s.pt"),
        note="Proxy fallback to yolo11s.",
    ),
    "rfdetr-medium": ModelSpec(
        "rfdetr-medium",
        "RF-DETR",
        "medium",
        checkpoints=("rfdetr-medium.pt", "yolo11m.pt"),
        note="Proxy fallback to yolo11m.",
    ),
    "rfdetr-2xlarge": ModelSpec(
        "rfdetr-2xlarge",
        "RF-DETR",
        "2x-large",
        checkpoints=("rfdetr-2xlarge.pt", "yolo11x.pt"),
        note="Proxy fallback to yolo11x.",
    ),
    "dino15-proxy": ModelSpec(
        "dino15-proxy",
        "DINO 1.5 Pro",
        "proxy",
        backend="groundingdino",
        hf_model_id="IDEA-Research/grounding-dino-tiny",
        prompt="bird.",
        fallback_checkpoints=("rtdetr-x.pt",),
        note="GroundingDINO tiny proxy with fallback to rtdetr-x.",
    ),
}

DEFAULT_MODEL_ORDER = [
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10l",
    "yolov10x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
    "rtdetr-l",
    "rtdetr-x",
    "rtdetrv2-l",
    "rtdetrv2-x",
    "rtdetrv3-r18",
    "rtdetrv3-r50",
    "rtdetrv3-r101",
    "rfdetr-nano",
    "rfdetr-small",
    "rfdetr-medium",
    "rfdetr-2xlarge",
    "dino15-proxy",
]


def infer_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, spec in MODEL_CATALOG.items():
        aliases[normalize_token(key)] = key
        aliases[normalize_token(spec.display_name)] = key
        aliases[normalize_token(spec.family + spec.variant)] = key
        for checkpoint in spec.checkpoints:
            aliases[normalize_token(checkpoint)] = key
    return aliases


def resolve_model_spec(token: str, alias_map: dict[str, str]) -> ModelSpec:
    mapped = alias_map.get(normalize_token(token))
    if mapped:
        return MODEL_CATALOG[mapped]
    return ModelSpec(
        key=token,
        family="Custom",
        variant=token,
        checkpoints=(token,),
        note="Custom model/path from CLI.",
    )


def parse_model_specs(models_csv: str, model_args: list[str]) -> list[ModelSpec]:
    alias_map = build_alias_map()
    tokens: list[str] = []
    if models_csv.strip():
        tokens.extend(part.strip() for part in models_csv.split(",") if part.strip())
    for item in model_args:
        tokens.extend(part.strip() for part in item.split(",") if part.strip())
    if not tokens:
        tokens = list(DEFAULT_MODEL_ORDER)

    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for token in tokens:
        spec = resolve_model_spec(token, alias_map)
        if spec.key not in seen:
            specs.append(spec)
            seen.add(spec.key)
    return specs


def resolve_images_dir(explicit_dir: str | None, repo_root: pathlib.Path) -> pathlib.Path:
    if explicit_dir:
        path = pathlib.Path(explicit_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Image directory not found: {path}")
        return path

    candidates = [
        repo_root / "scraped_images" / "scolop2_10k",
        repo_root / "scraped_images" / "scolop2_10K",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    joined = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find default image directory. Checked:\n"
        f"  - {joined}\n"
        "Use --images-dir to set it explicitly."
    )


def iter_images(images_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def pick_images(images: Sequence[pathlib.Path], sample_count: int, sample_mode: str, seed: int) -> list[pathlib.Path]:
    rng = random.Random(seed)
    all_images = list(images)

    if sample_count <= 0 or sample_count >= len(all_images):
        if sample_mode == "first":
            return all_images
        rng.shuffle(all_images)
        return all_images

    if sample_mode == "first":
        return all_images[:sample_count]

    return rng.sample(all_images, k=sample_count)


def next_image(image_pool: Sequence[pathlib.Path], cursor: int) -> tuple[pathlib.Path, int]:
    image = image_pool[cursor % len(image_pool)]
    return image, (cursor + 1) % len(image_pool)


def resolve_torch_device(requested: str):
    import torch

    req = requested.lower().strip()
    if req.startswith("cuda") and torch.cuda.is_available():
        return torch.device(req)
    if req == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def try_load_ultralytics(spec: ModelSpec) -> tuple[LoadedModel | None, str]:
    errors: list[str] = []
    for checkpoint in spec.checkpoints:
        try:
            loaded = LoadedModel(
                spec=spec,
                backend="ultralytics",
                model_obj=YOLO(checkpoint),
                resolved_source=checkpoint,
                resolution_note=(f"fallback from {spec.checkpoints[0]}" if checkpoint != spec.checkpoints[0] else ""),
            )
            return loaded, ""
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{checkpoint}: {exc}")
    return None, " | ".join(errors)


def try_load_groundingdino(spec: ModelSpec, requested_device: str) -> tuple[LoadedModel | None, str]:
    if not spec.hf_model_id:
        return None, "Missing hf_model_id for groundingdino backend."
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except Exception as exc:  # noqa: BLE001
        return None, f"transformers import failed: {exc}"
    try:
        processor = AutoProcessor.from_pretrained(spec.hf_model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(spec.hf_model_id)
        device = resolve_torch_device(requested_device)
        model.to(device)
        model.eval()
        loaded = LoadedModel(
            spec=spec,
            backend="groundingdino",
            model_obj=model,
            processor_obj=processor,
            runtime_device=str(device),
            resolved_source=spec.hf_model_id,
        )
        return loaded, ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def load_models(specs: Sequence[ModelSpec], requested_device: str) -> tuple[list[LoadedModel], dict[str, str]]:
    loaded: list[LoadedModel] = []
    failures: dict[str, str] = {}

    for spec in specs:
        if spec.backend == "groundingdino":
            model, error = try_load_groundingdino(spec, requested_device)
            if model is not None:
                loaded.append(model)
                continue

            fallback_spec = ModelSpec(
                key=spec.key,
                family=spec.family,
                variant=spec.variant,
                backend="ultralytics",
                checkpoints=spec.fallback_checkpoints,
                note=spec.note,
            )
            if fallback_spec.checkpoints:
                fallback_model, fallback_error = try_load_ultralytics(fallback_spec)
                if fallback_model is not None:
                    fallback_model.resolution_note = f"groundingdino unavailable ({error}); fallback checkpoint used."
                    loaded.append(fallback_model)
                    continue
                failures[spec.key] = f"{error} | fallback failed: {fallback_error}"
            else:
                failures[spec.key] = error
            continue

        model, error = try_load_ultralytics(spec)
        if model is not None:
            loaded.append(model)
        else:
            failures[spec.key] = error

    return loaded, failures


def estimate_best_probabilities(stats: dict[str, VoteStats], draws: int, rng: np.random.Generator) -> dict[str, float]:
    keys = list(stats.keys())
    if not keys:
        return {}
    samples: list[np.ndarray] = []
    for key in keys:
        st = stats[key]
        samples.append(rng.beta(1.0 + st.wins, 1.0 + st.losses, size=draws))
    matrix = np.vstack(samples)
    winner_idx = np.argmax(matrix, axis=0)
    counts = np.bincount(winner_idx, minlength=len(keys))
    probs = counts / draws
    return {keys[i]: float(probs[i]) for i in range(len(keys))}


def current_leader(best_probs: dict[str, float]) -> tuple[str | None, float]:
    if not best_probs:
        return None, 0.0
    key = max(best_probs, key=best_probs.get)
    return key, float(best_probs[key])


def compute_selection_weight(
    st: VoteStats,
    weight_floor: float,
    weight_mean: float,
    weight_uncertainty: float,
    novelty_weight: float,
    unseen_bonus: float,
) -> float:
    a = 1.0 + st.wins
    b = 1.0 + st.losses
    mean = a / (a + b)
    std = math.sqrt((a * b) / (((a + b) ** 2) * (a + b + 1.0)))
    novelty = novelty_weight / (1.0 + st.votes)
    unseen = unseen_bonus if st.votes == 0 else 0.0
    return max(1e-9, weight_floor + weight_mean * mean + weight_uncertainty * std + novelty + unseen)


def weighted_sample_without_replacement(
    keys: Sequence[str],
    weights: Sequence[float],
    k: int,
    rng: np.random.Generator,
) -> list[str]:
    pool = list(keys)
    w = np.array(weights, dtype=float)
    selected: list[str] = []
    for _ in range(min(k, len(pool))):
        if w.sum() <= 0:
            idx = int(rng.integers(0, len(pool)))
        else:
            probs = w / w.sum()
            idx = int(rng.choice(len(pool), p=probs))
        selected.append(pool.pop(idx))
        w = np.delete(w, idx)
    return selected


def sample_model_pair(
    model_keys: Sequence[str],
    stats: dict[str, VoteStats],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[str, str]:
    keys = list(model_keys)
    if len(keys) < 2:
        raise ValueError("Need at least two models for pair sampling.")

    weights = [
        compute_selection_weight(
            st=stats[key],
            weight_floor=args.weight_floor,
            weight_mean=args.weight_mean,
            weight_uncertainty=args.weight_uncertainty,
            novelty_weight=args.novelty_weight,
            unseen_bonus=args.unseen_bonus,
        )
        for key in keys
    ]
    pair = weighted_sample_without_replacement(keys, weights, 2, rng)
    if len(pair) < 2:
        raise ValueError("Pair sampling failed.")
    if rng.random() < 0.5:
        pair = [pair[1], pair[0]]
    return pair[0], pair[1]


def sample_weighted_model(
    candidate_keys: Sequence[str],
    stats: dict[str, VoteStats],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> str:
    if not candidate_keys:
        raise ValueError("No candidate models available.")
    weights = [
        compute_selection_weight(
            st=stats[key],
            weight_floor=args.weight_floor,
            weight_mean=args.weight_mean,
            weight_uncertainty=args.weight_uncertainty,
            novelty_weight=args.novelty_weight,
            unseen_bonus=args.unseen_bonus,
        )
        for key in candidate_keys
    ]
    selected = weighted_sample_without_replacement(candidate_keys, weights, 1, rng)
    if not selected:
        raise ValueError("Weighted sampling returned no model.")
    return selected[0]


def build_followup_pair(
    winner_key: str,
    previous_opponent_key: str,
    all_model_keys: Sequence[str],
    stats: dict[str, VoteStats],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[str, str] | None:
    candidates = [key for key in all_model_keys if key != winner_key and key != previous_opponent_key]
    if not candidates:
        candidates = [key for key in all_model_keys if key != winner_key]
    if not candidates:
        return None

    challenger = sample_weighted_model(candidates, stats, rng, args)
    if rng.random() < 0.5:
        return winner_key, challenger
    return challenger, winner_key


def run_ultralytics_prediction(
    loaded: LoadedModel,
    image_path: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, PredictionMetrics]:
    started = time.perf_counter()
    results = loaded.model_obj.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not results:
        fallback = cv2.imread(str(image_path))
        if fallback is None:
            fallback = np.zeros((640, 960, 3), dtype=np.uint8)
        return fallback, PredictionMetrics(inference_ms=elapsed_ms)

    result = results[0]
    box_count = int(len(result.boxes)) if result.boxes is not None else 0
    conf_values: list[float] = []
    if result.boxes is not None and result.boxes.conf is not None:
        conf_values = [float(v) for v in result.boxes.conf.tolist()]
    mean_conf = float(sum(conf_values) / len(conf_values)) if conf_values else 0.0
    max_conf = float(max(conf_values)) if conf_values else 0.0
    plotted = result.plot(labels=not args.hide_labels, conf=not args.hide_conf, line_width=args.line_width)
    return plotted, PredictionMetrics(
        num_boxes=box_count,
        mean_conf=mean_conf,
        max_conf=max_conf,
        inference_ms=elapsed_ms,
    )


def run_groundingdino_prediction(
    loaded: LoadedModel,
    image_path: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, PredictionMetrics]:
    from PIL import Image
    import torch

    started = time.perf_counter()
    image_pil = Image.open(image_path).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    processor = loaded.processor_obj
    model = loaded.model_obj
    if processor is None:
        raise RuntimeError("GroundingDINO processor missing.")

    inputs = processor(images=image_pil, text=[loaded.spec.prompt], return_tensors="pt")
    device = model.device if hasattr(model, "device") else resolve_torch_device(loaded.runtime_device or "cpu")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = [(image_bgr.shape[0], image_bgr.shape[1])]
    processed = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs["input_ids"],
        box_threshold=args.conf,
        text_threshold=args.dino_text_threshold,
        target_sizes=target_sizes,
    )[0]

    boxes = processed.get("boxes", [])
    scores = processed.get("scores", [])
    labels = processed.get("labels", [])

    out = image_bgr.copy()
    score_values: list[float] = []
    lw = 2 if args.line_width is None else max(1, args.line_width)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in (box.tolist() if hasattr(box, "tolist") else box)]
        score_f = float(score.item() if hasattr(score, "item") else score)
        label_txt = str(label)

        cv2.rectangle(out, (x1, y1), (x2, y2), (35, 220, 35), lw)
        parts: list[str] = []
        if not args.hide_labels:
            parts.append(label_txt)
        if not args.hide_conf:
            parts.append(f"{score_f:.2f}")
        if parts:
            cv2.putText(
                out,
                " ".join(parts),
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (35, 220, 35),
                2,
                cv2.LINE_AA,
            )
        score_values.append(score_f)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    mean_conf = float(sum(score_values) / len(score_values)) if score_values else 0.0
    max_conf = float(max(score_values)) if score_values else 0.0
    return out, PredictionMetrics(
        num_boxes=len(score_values),
        mean_conf=mean_conf,
        max_conf=max_conf,
        inference_ms=elapsed_ms,
    )


def run_prediction(loaded: LoadedModel, image_path: pathlib.Path, args: argparse.Namespace) -> tuple[np.ndarray, PredictionMetrics]:
    if loaded.backend == "groundingdino":
        return run_groundingdino_prediction(loaded, image_path, args)
    return run_ultralytics_prediction(loaded, image_path, args)


def prepare_pair_screen(
    screen_index: int,
    image_path: pathlib.Path,
    left_key: str,
    right_key: str,
    loaded_by_key: dict[str, LoadedModel],
    args: argparse.Namespace,
    same_image_depth: int = 0,
) -> PairScreen:
    left_loaded = loaded_by_key[left_key]
    right_loaded = loaded_by_key[right_key]

    try:
        left_frame, left_metrics = run_prediction(left_loaded, image_path, args)
    except Exception as exc:  # noqa: BLE001
        fallback = cv2.imread(str(image_path))
        if fallback is None:
            fallback = np.zeros((640, 960, 3), dtype=np.uint8)
        left_frame, left_metrics = fallback, PredictionMetrics(error=str(exc))

    try:
        right_frame, right_metrics = run_prediction(right_loaded, image_path, args)
    except Exception as exc:  # noqa: BLE001
        fallback = cv2.imread(str(image_path))
        if fallback is None:
            fallback = np.zeros((640, 960, 3), dtype=np.uint8)
        right_frame, right_metrics = fallback, PredictionMetrics(error=str(exc))

    return PairScreen(
        screen_index=screen_index,
        image_path=image_path,
        left_model_key=left_key,
        right_model_key=right_key,
        left_frame=left_frame,
        right_frame=right_frame,
        left_metrics=left_metrics,
        right_metrics=right_metrics,
        same_image_depth=same_image_depth,
    )


def fit_to_box(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    scale = min(out_w / w, out_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def render_side_panel(frame: np.ndarray, title: str, panel_w: int, panel_h: int) -> np.ndarray:
    panel = fit_to_box(frame, panel_w, panel_h)
    cv2.putText(panel, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (85, 85, 85), 2)
    return panel


def compose_pair_board(
    screen: PairScreen,
    leader_prob: float,
    winner_confidence: float,
    scored_pairs: int,
    max_screens: int,
    display_width: int,
    display_height: int,
) -> np.ndarray:
    header_h = 120
    gap = 12
    panel_w = max(260, int((display_width - gap) / 2))
    panel_h = max(250, display_height - header_h)

    left_panel = render_side_panel(screen.left_frame, "Left (Q)", panel_w, panel_h)
    right_panel = render_side_panel(screen.right_frame, "Right (D)", panel_w, panel_h)
    row = np.hstack([left_panel, np.zeros((panel_h, gap, 3), dtype=np.uint8), right_panel])

    board = np.zeros((header_h + row.shape[0], row.shape[1], 3), dtype=np.uint8)
    board[header_h:, :] = row

    lines = [
        f"Blind pairwise mode | Screen {screen.screen_index + 1}/{max_screens} | {screen.image_path.name}",
        (
            f"Scored pairs: {scored_pairs} | "
            f"Leader confidence: {leader_prob:.1%} (target {winner_confidence:.1%})"
        ),
        "Keys: Q=left better, D=right better, S=both correct, Z=both wrong | N=finish | Esc=quit",
    ]
    y = 30
    for line in lines:
        cv2.putText(board, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
        y += 32
    return board


def compose_loading_board(
    screen_ordinal: int,
    max_screens: int,
    elapsed_sec: float,
    display_width: int,
    display_height: int,
) -> np.ndarray:
    board = np.zeros((max(320, display_height), max(720, display_width), 3), dtype=np.uint8)
    lines = [
        "Preparing next comparison...",
        f"Screen {screen_ordinal}/{max_screens} | Waited: {elapsed_sec:.1f}s",
        "Startup can be slow on CPU with large models and high image resolution.",
        "Press Esc to quit while loading.",
    ]
    y = 70
    for line in lines:
        cv2.putText(board, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        y += 52
    return board


def wait_for_screen_with_loading(
    future: Future[PairScreen],
    window_name: str,
    screen_ordinal: int,
    max_screens: int,
    display_width: int,
    display_height: int,
) -> tuple[PairScreen | None, float]:
    start = time.monotonic()
    while True:
        try:
            return future.result(timeout=0.05), (time.monotonic() - start)
        except FuturesTimeoutError:
            elapsed = time.monotonic() - start
            board = compose_loading_board(
                screen_ordinal=screen_ordinal,
                max_screens=max_screens,
                elapsed_sec=elapsed,
                display_width=display_width,
                display_height=display_height,
            )
            cv2.imshow(window_name, board)
            key = cv2.waitKey(1)
            if key >= 0 and (key & 0xFF) == 27:
                return None, elapsed


def wait_for_choice() -> str:
    while True:
        key = cv2.waitKey(0)
        if key < 0:
            continue
        code = key & 0xFF
        if code in (ord("q"), ord("Q")):
            return "left"
        if code in (ord("d"), ord("D")):
            return "right"
        if code in (ord("s"), ord("S")):
            return "both_correct"
        if code in (ord("z"), ord("Z")):
            return "both_wrong"
        if code in (ord("n"), ord("N")):
            return "finish"
        if code == 27:
            return "quit"


def write_inventory(
    run_dir: pathlib.Path,
    requested_specs: Sequence[ModelSpec],
    loaded_models: Sequence[LoadedModel],
    failures: dict[str, str],
) -> None:
    loaded_by_key = {item.spec.key: item for item in loaded_models}
    path = run_dir / "model_inventory.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "key",
                "family",
                "variant",
                "status",
                "backend",
                "resolved_source",
                "checkpoint_candidates",
                "note_or_error",
            ],
        )
        writer.writeheader()
        for spec in requested_specs:
            loaded = loaded_by_key.get(spec.key)
            if loaded:
                writer.writerow(
                    {
                        "key": spec.key,
                        "family": spec.family,
                        "variant": spec.variant,
                        "status": "loaded",
                        "backend": loaded.backend,
                        "resolved_source": loaded.resolved_source,
                        "checkpoint_candidates": "|".join(spec.checkpoints),
                        "note_or_error": loaded.resolution_note or spec.note,
                    }
                )
            else:
                writer.writerow(
                    {
                        "key": spec.key,
                        "family": spec.family,
                        "variant": spec.variant,
                        "status": "failed_or_skipped",
                        "backend": spec.backend,
                        "resolved_source": "",
                        "checkpoint_candidates": "|".join(spec.checkpoints),
                        "note_or_error": failures.get(spec.key, spec.note),
                    }
                )


def write_decisions(run_dir: pathlib.Path, decisions: list[dict], spec_by_key: dict[str, ModelSpec]) -> None:
    path = run_dir / "decisions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "screen_index",
                "image_name",
                "choice",
                "left_model_key",
                "left_model_name",
                "left_is_correct",
                "left_inference_ms",
                "right_model_key",
                "right_model_name",
                "right_is_correct",
                "right_inference_ms",
                "followup_enqueued",
                "followup_left_model_key",
                "followup_right_model_key",
                "leader_key_after",
                "leader_prob_after",
            ],
        )
        writer.writeheader()
        for row in decisions:
            writer.writerow(
                {
                    "screen_index": row["screen_index"],
                    "image_name": row["image_name"],
                    "choice": row["choice"],
                    "left_model_key": row["left_model_key"],
                    "left_model_name": spec_by_key[row["left_model_key"]].display_name,
                    "left_is_correct": row.get("left_is_correct", ""),
                    "left_inference_ms": f"{row['left_inference_ms']:.3f}",
                    "right_model_key": row["right_model_key"],
                    "right_model_name": spec_by_key[row["right_model_key"]].display_name,
                    "right_is_correct": row.get("right_is_correct", ""),
                    "right_inference_ms": f"{row['right_inference_ms']:.3f}",
                    "followup_enqueued": row.get("followup_enqueued", ""),
                    "followup_left_model_key": row.get("followup_left_model_key", ""),
                    "followup_right_model_key": row.get("followup_right_model_key", ""),
                    "leader_key_after": row.get("leader_key_after", ""),
                    "leader_prob_after": f"{row.get('leader_prob_after', 0.0):.6f}",
                }
            )


def write_scoreboard(run_dir: pathlib.Path, stats: dict[str, VoteStats], probs: dict[str, float], spec_by_key: dict[str, ModelSpec]) -> None:
    ranking = sorted(
        stats.keys(),
        key=lambda key: (probs.get(key, 0.0), stats[key].win_rate, stats[key].wins),
        reverse=True,
    )
    path = run_dir / "scoreboard.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "model_key",
                "model_name",
                "wins",
                "losses",
                "ties",
                "votes",
                "win_rate",
                "correct",
                "judged",
                "correct_rate",
                "posterior_best_prob",
            ],
        )
        writer.writeheader()
        for idx, key in enumerate(ranking, start=1):
            st = stats[key]
            writer.writerow(
                {
                    "rank": idx,
                    "model_key": key,
                    "model_name": spec_by_key[key].display_name,
                    "wins": f"{st.wins:.3f}",
                    "losses": f"{st.losses:.3f}",
                    "ties": st.ties,
                    "votes": f"{st.votes:.3f}",
                    "win_rate": f"{st.win_rate:.6f}",
                    "correct": st.correct,
                    "judged": st.judged,
                    "correct_rate": f"{st.correct_rate:.6f}",
                    "posterior_best_prob": f"{probs.get(key, 0.0):.6f}",
                }
            )


def write_inference_stats(run_dir: pathlib.Path, timings: dict[str, TimingStats], spec_by_key: dict[str, ModelSpec]) -> None:
    path = run_dir / "inference_stats.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_key", "model_name", "count", "mean_ms", "p50_ms", "p90_ms", "min_ms", "max_ms"],
        )
        writer.writeheader()
        for key in sorted(timings.keys()):
            ts = timings[key]
            writer.writerow(
                {
                    "model_key": key,
                    "model_name": spec_by_key[key].display_name,
                    "count": ts.count,
                    "mean_ms": f"{ts.mean_ms:.3f}",
                    "p50_ms": f"{ts.p50_ms:.3f}",
                    "p90_ms": f"{ts.p90_ms:.3f}",
                    "min_ms": f"{(0.0 if ts.min_ms == float('inf') else ts.min_ms):.3f}",
                    "max_ms": f"{ts.max_ms:.3f}",
                }
            )


def write_tradeoff_report(
    run_dir: pathlib.Path,
    stats: dict[str, VoteStats],
    probs: dict[str, float],
    timings: dict[str, TimingStats],
    spec_by_key: dict[str, ModelSpec],
) -> None:
    ranking_quality = sorted(
        stats.keys(),
        key=lambda key: (probs.get(key, 0.0), stats[key].win_rate, stats[key].wins),
        reverse=True,
    )
    ranking_speed = sorted(
        stats.keys(),
        key=lambda key: (timings[key].mean_ms if timings[key].count > 0 else float("inf")),
    )
    speed_rank = {key: idx + 1 for idx, key in enumerate(ranking_speed)}

    path = run_dir / "tradeoff_report.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "quality_rank",
                "speed_rank",
                "model_key",
                "model_name",
                "posterior_best_prob",
                "win_rate",
                "wins",
                "losses",
                "ties",
                "correct",
                "judged",
                "correct_rate",
                "mean_ms",
                "p90_ms",
            ],
        )
        writer.writeheader()
        for idx, key in enumerate(ranking_quality, start=1):
            st = stats[key]
            ts = timings[key]
            writer.writerow(
                {
                    "quality_rank": idx,
                    "speed_rank": speed_rank[key],
                    "model_key": key,
                    "model_name": spec_by_key[key].display_name,
                    "posterior_best_prob": f"{probs.get(key, 0.0):.6f}",
                    "win_rate": f"{st.win_rate:.6f}",
                    "wins": f"{st.wins:.3f}",
                    "losses": f"{st.losses:.3f}",
                    "ties": st.ties,
                    "correct": st.correct,
                    "judged": st.judged,
                    "correct_rate": f"{st.correct_rate:.6f}",
                    "mean_ms": f"{ts.mean_ms:.3f}",
                    "p90_ms": f"{ts.p90_ms:.3f}",
                }
            )


def write_summary(
    run_dir: pathlib.Path,
    stop_reason: str,
    winner_key: str | None,
    winner_prob: float,
    spec_by_key: dict[str, ModelSpec],
    scored_pairs: int,
    screens_generated: int,
) -> None:
    path = run_dir / "summary.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now().isoformat(),
                "stop_reason": stop_reason,
                "winner_key": winner_key,
                "winner_name": spec_by_key[winner_key].display_name if winner_key else None,
                "winner_probability": winner_prob,
                "scored_pairs": scored_pairs,
                "screens_generated": screens_generated,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blind adaptive pairwise model comparison with background prefetch.",
    )
    parser.add_argument("--repo-root", default=None, help="Repository root (default: inferred).")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Image directory. Defaults to scraped_images/scolop2_10k (or scolop2_10K).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root. A timestamped run directory is created.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Image pool size (<=0 means all). Larger default for stronger coverage.",
    )
    parser.add_argument("--sample-mode", choices=["random", "first"], default="random", help="How to choose image pool.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model aliases/checkpoints. Empty means full model table set.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Extra model aliases/checkpoints (repeatable; comma-separated accepted).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold for Ultralytics models.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size for Ultralytics models.")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections for Ultralytics models.")
    parser.add_argument("--device", default="cpu", help="Inference device (cpu/mps/cuda:0).")
    parser.add_argument("--dino-text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--weight-floor", type=float, default=0.03, help="Base sampling weight.")
    parser.add_argument("--weight-mean", type=float, default=0.75, help="Sampling weight on estimated quality.")
    parser.add_argument("--weight-uncertainty", type=float, default=1.00, help="Sampling weight on uncertainty.")
    parser.add_argument("--novelty-weight", type=float, default=0.60, help="Sampling weight on low-coverage models.")
    parser.add_argument("--unseen-bonus", type=float, default=0.25, help="Extra sampling bonus for unseen models.")
    parser.add_argument("--min-screens", type=int, default=40, help="Min scored screens before confidence stop.")
    parser.add_argument("--max-screens", type=int, default=600, help="Hard cap on screens.")
    parser.add_argument(
        "--min-votes-for-leader",
        type=int,
        default=40,
        help="Minimum leader vote mass before confidence stop.",
    )
    parser.add_argument("--winner-confidence", type=float, default=0.95, help="Posterior threshold to stop.")
    parser.add_argument("--posterior-draws", type=int, default=20000, help="Monte Carlo draws.")
    parser.add_argument("--prefetch-size", type=int, default=4, help="How many future screens to keep preloaded.")
    parser.add_argument("--prefetch-workers", type=int, default=1, help="Background workers for preloading.")
    parser.add_argument(
        "--same-image-followup",
        action="store_true",
        default=True,
        help="After non-tie results, enqueue winner vs challenger on the same image.",
    )
    parser.add_argument(
        "--no-same-image-followup",
        action="store_false",
        dest="same_image_followup",
        help="Disable winner follow-up comparisons on the same image.",
    )
    parser.add_argument(
        "--max-same-image-followup-depth",
        type=int,
        default=1,
        help="Maximum chained follow-up depth on the same image (0 disables chaining).",
    )
    parser.add_argument(
        "--max-same-image-followups-per-image",
        type=int,
        default=1,
        help="Maximum number of follow-up screens allowed for one image.",
    )
    parser.add_argument("--display-max-width", type=int, default=1860, help="Max UI width.")
    parser.add_argument("--display-max-height", type=int, default=980, help="Max UI height.")
    parser.add_argument("--window-name", default="Blind Pairwise Judge", help="OpenCV window title.")
    parser.add_argument("--hide-labels", action="store_true", help="Hide class labels.")
    parser.add_argument("--hide-conf", action="store_true", help="Hide confidence labels.")
    parser.add_argument("--line-width", type=int, default=None, help="Bounding-box line width.")
    parser.add_argument("--dry-run-load", action="store_true", help="Only load models then exit.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_screens < 1:
        print("--max-screens must be >= 1", file=sys.stderr)
        return 1
    if args.prefetch_size < 1:
        print("--prefetch-size must be >= 1", file=sys.stderr)
        return 1
    if args.prefetch_workers < 1:
        print("--prefetch-workers must be >= 1", file=sys.stderr)
        return 1
    if args.max_same_image_followup_depth < 0:
        print("--max-same-image-followup-depth must be >= 0", file=sys.stderr)
        return 1
    if args.max_same_image_followups_per_image < 0:
        print("--max-same-image-followups-per-image must be >= 0", file=sys.stderr)
        return 1

    repo_root = pathlib.Path(args.repo_root).expanduser().resolve() if args.repo_root else infer_repo_root()
    output_root = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (repo_root / "model_compare_pairwise").resolve()
    )
    run_dir = output_root / dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        images_dir = resolve_images_dir(args.images_dir, repo_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    all_images = list(iter_images(images_dir))
    if not all_images:
        print(f"No images found in {images_dir}", file=sys.stderr)
        return 1
    image_pool = pick_images(all_images, args.samples, args.sample_mode, args.seed)

    requested_specs = parse_model_specs(args.models, args.model)
    loaded_models, load_failures = load_models(requested_specs, requested_device=args.device)
    if len(loaded_models) < 2:
        print(f"Need at least 2 loadable models. Loaded: {len(loaded_models)}", file=sys.stderr)
        for key, reason in load_failures.items():
            print(f"  - {key}: {reason}", file=sys.stderr)
        write_inventory(run_dir, requested_specs, loaded_models, load_failures)
        return 1

    loaded_by_key = {item.spec.key: item for item in loaded_models}
    spec_by_key = {item.spec.key: item.spec for item in loaded_models}
    stats = {key: VoteStats() for key in loaded_by_key.keys()}
    timings = {key: TimingStats() for key in loaded_by_key.keys()}

    write_inventory(run_dir, requested_specs, loaded_models, load_failures)
    image_pool_path = run_dir / "image_pool.txt"
    image_pool_path.write_text(
        "\n".join(str(path.relative_to(images_dir)) for path in image_pool) + "\n",
        encoding="utf-8",
    )

    print(f"Images directory: {images_dir}")
    print(f"Image pool size: {len(image_pool)}")
    print(f"Requested models: {len(requested_specs)}")
    print(f"Loaded models: {len(loaded_models)}")
    if load_failures:
        print("Skipped/failed models:")
        for key, reason in load_failures.items():
            print(f"  - {key}: {reason}")
    print(f"Run directory: {run_dir}")
    print("Blind mode active: Q left, D right, S both correct, Z both wrong (S/Z do not affect wins/losses).")
    print("")

    if args.dry_run_load:
        print("Dry run complete. Exiting after model loading.")
        return 0

    selection_rng = np.random.default_rng(args.seed + 17)
    posterior_rng = np.random.default_rng(args.seed + 99)
    image_cursor = 0
    next_screen_index = 0
    scored_pairs = 0
    displayed_screens: list[PairScreen] = []
    decisions: list[dict] = []
    stop_reason = "user_quit"
    same_image_followup_counts: dict[str, int] = {}

    prefetch_queue: deque[Future[PairScreen]] = deque()

    def schedule_more(executor: ThreadPoolExecutor) -> None:
        nonlocal image_cursor, next_screen_index
        while len(prefetch_queue) < args.prefetch_size and next_screen_index < args.max_screens:
            image_path, image_cursor_next = next_image(image_pool, image_cursor)
            left_key, right_key = sample_model_pair(list(loaded_by_key.keys()), stats, selection_rng, args)
            future = executor.submit(
                prepare_pair_screen,
                next_screen_index,
                image_path,
                left_key,
                right_key,
                loaded_by_key,
                args,
                0,
            )
            prefetch_queue.append(future)
            next_screen_index += 1
            image_cursor = image_cursor_next

    def enqueue_specific_screen(
        executor: ThreadPoolExecutor,
        image_path: pathlib.Path,
        left_key: str,
        right_key: str,
        same_image_depth: int = 0,
    ) -> bool:
        nonlocal next_screen_index
        if next_screen_index >= args.max_screens:
            return False
        future = executor.submit(
            prepare_pair_screen,
            next_screen_index,
            image_path,
            left_key,
            right_key,
            loaded_by_key,
            args,
            same_image_depth,
        )
        prefetch_queue.append(future)
        next_screen_index += 1
        return True

    actions_path = run_dir / "actions.csv"
    with actions_path.open("w", newline="", encoding="utf-8") as actions_file:
        action_writer = csv.DictWriter(
            actions_file,
            fieldnames=[
                "timestamp",
                "screen_index",
                "image_name",
                "choice",
                "left_model_key",
                "right_model_key",
                "left_is_correct",
                "right_is_correct",
                "followup_enqueued",
                "followup_left_model_key",
                "followup_right_model_key",
                "leader_key_after",
                "leader_prob_after",
                "left_inference_ms",
                "right_inference_ms",
            ],
        )
        action_writer.writeheader()

        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, min(1600, args.display_max_width), min(980, args.display_max_height))

        with ThreadPoolExecutor(max_workers=args.prefetch_workers) as executor:
            schedule_more(executor)

            while True:
                if not prefetch_queue:
                    stop_reason = "queue_empty"
                    break

                future = prefetch_queue.popleft()
                try:
                    screen, wait_sec = wait_for_screen_with_loading(
                        future=future,
                        window_name=args.window_name,
                        screen_ordinal=(len(displayed_screens) + 1),
                        max_screens=args.max_screens,
                        display_width=args.display_max_width,
                        display_height=args.display_max_height,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"Screen preparation failed: {exc}", file=sys.stderr)
                    stop_reason = "screen_prepare_error"
                    break
                if screen is None:
                    stop_reason = "user_quit"
                    break
                if wait_sec >= 1.0:
                    print(
                        f"Prepared screen {screen.screen_index + 1}/{args.max_screens} in {wait_sec:.1f}s.",
                        flush=True,
                    )

                displayed_screens.append(screen)
                timings[screen.left_model_key].add(screen.left_metrics.inference_ms)
                timings[screen.right_model_key].add(screen.right_metrics.inference_ms)

                # Keep preparing future screens while user is scoring current one.
                schedule_more(executor)

                probs = estimate_best_probabilities(stats, args.posterior_draws, posterior_rng)
                _, leader_prob = current_leader(probs)

                board = compose_pair_board(
                    screen=screen,
                    leader_prob=leader_prob,
                    winner_confidence=args.winner_confidence,
                    scored_pairs=scored_pairs,
                    max_screens=args.max_screens,
                    display_width=args.display_max_width,
                    display_height=args.display_max_height,
                )
                cv2.imshow(args.window_name, board)
                choice = wait_for_choice()

                if choice == "quit":
                    stop_reason = "user_quit"
                    break
                if choice == "finish":
                    stop_reason = "manual_finish"
                    break

                left_stats = stats[screen.left_model_key]
                right_stats = stats[screen.right_model_key]
                if choice == "left":
                    left_stats.wins += 1.0
                    right_stats.losses += 1.0
                    left_stats.correct += 1
                    left_stats.judged += 1
                    right_stats.judged += 1
                    left_is_correct = 1
                    right_is_correct = 0
                elif choice == "right":
                    right_stats.wins += 1.0
                    left_stats.losses += 1.0
                    right_stats.correct += 1
                    left_stats.judged += 1
                    right_stats.judged += 1
                    left_is_correct = 0
                    right_is_correct = 1
                elif choice == "both_correct":
                    left_stats.ties += 1
                    right_stats.ties += 1
                    left_stats.correct += 1
                    right_stats.correct += 1
                    left_stats.judged += 1
                    right_stats.judged += 1
                    left_is_correct = 1
                    right_is_correct = 1
                elif choice == "both_wrong":
                    left_stats.ties += 1
                    right_stats.ties += 1
                    left_stats.judged += 1
                    right_stats.judged += 1
                    left_is_correct = 0
                    right_is_correct = 0
                else:
                    continue

                scored_pairs += 1
                probs_after = estimate_best_probabilities(stats, args.posterior_draws, posterior_rng)
                leader_key_after, leader_prob_after = current_leader(probs_after)

                row = {
                    "screen_index": screen.screen_index + 1,
                    "image_name": screen.image_path.name,
                    "choice": choice,
                    "left_model_key": screen.left_model_key,
                    "right_model_key": screen.right_model_key,
                    "left_is_correct": left_is_correct,
                    "right_is_correct": right_is_correct,
                    "followup_enqueued": "",
                    "followup_left_model_key": "",
                    "followup_right_model_key": "",
                    "leader_key_after": leader_key_after or "",
                    "leader_prob_after": leader_prob_after,
                    "left_inference_ms": screen.left_metrics.inference_ms,
                    "right_inference_ms": screen.right_metrics.inference_ms,
                }
                decisions.append(row)

                if args.same_image_followup and choice in {"left", "right"}:
                    winner_key = screen.left_model_key if choice == "left" else screen.right_model_key
                    loser_key = screen.right_model_key if choice == "left" else screen.left_model_key
                    image_key = str(screen.image_path.resolve())
                    current_image_followups = same_image_followup_counts.get(image_key, 0)
                    if screen.same_image_depth >= args.max_same_image_followup_depth:
                        row["followup_enqueued"] = "depth_limit"
                    elif current_image_followups >= args.max_same_image_followups_per_image:
                        row["followup_enqueued"] = "image_limit"
                    else:
                        followup_pair = build_followup_pair(
                            winner_key=winner_key,
                            previous_opponent_key=loser_key,
                            all_model_keys=list(loaded_by_key.keys()),
                            stats=stats,
                            rng=selection_rng,
                            args=args,
                        )
                        if followup_pair is None:
                            row["followup_enqueued"] = "no_candidate"
                        else:
                            enqueued = enqueue_specific_screen(
                                executor=executor,
                                image_path=screen.image_path,
                                left_key=followup_pair[0],
                                right_key=followup_pair[1],
                                same_image_depth=screen.same_image_depth + 1,
                            )
                            if enqueued:
                                row["followup_enqueued"] = "yes"
                                row["followup_left_model_key"] = followup_pair[0]
                                row["followup_right_model_key"] = followup_pair[1]
                                same_image_followup_counts[image_key] = current_image_followups + 1
                            else:
                                row["followup_enqueued"] = "no_capacity"

                action_writer.writerow(
                    {
                        "timestamp": dt.datetime.now().isoformat(),
                        "screen_index": row["screen_index"],
                        "image_name": row["image_name"],
                        "choice": row["choice"],
                        "left_model_key": row["left_model_key"],
                        "right_model_key": row["right_model_key"],
                        "left_is_correct": row["left_is_correct"],
                        "right_is_correct": row["right_is_correct"],
                        "followup_enqueued": row["followup_enqueued"],
                        "followup_left_model_key": row["followup_left_model_key"],
                        "followup_right_model_key": row["followup_right_model_key"],
                        "leader_key_after": row["leader_key_after"],
                        "leader_prob_after": f"{row['leader_prob_after']:.6f}",
                        "left_inference_ms": f"{row['left_inference_ms']:.3f}",
                        "right_inference_ms": f"{row['right_inference_ms']:.3f}",
                    }
                )
                actions_file.flush()

                if scored_pairs >= args.min_screens and leader_key_after:
                    if stats[leader_key_after].votes >= args.min_votes_for_leader and leader_prob_after >= args.winner_confidence:
                        stop_reason = "winner_confidence_reached"
                        break

                if scored_pairs >= args.max_screens:
                    stop_reason = "max_screens_scored"
                    break

            for pending in prefetch_queue:
                pending.cancel()

        cv2.destroyAllWindows()

    final_probs = estimate_best_probabilities(stats, args.posterior_draws, posterior_rng)
    winner_key, winner_prob = current_leader(final_probs)

    write_decisions(run_dir, decisions, spec_by_key)
    write_scoreboard(run_dir, stats, final_probs, spec_by_key)
    write_inference_stats(run_dir, timings, spec_by_key)
    write_tradeoff_report(run_dir, stats, final_probs, timings, spec_by_key)
    write_summary(
        run_dir=run_dir,
        stop_reason=stop_reason,
        winner_key=winner_key,
        winner_prob=winner_prob,
        spec_by_key=spec_by_key,
        scored_pairs=scored_pairs,
        screens_generated=len(displayed_screens),
    )

    print("")
    print("Pairwise adaptive session completed.")
    print(f"Stop reason: {stop_reason}")
    if winner_key:
        print(f"Winner: {winner_key} ({spec_by_key[winner_key].display_name}) P(best)={winner_prob:.1%}")
    else:
        print("Winner: n/a")
    print(f"Scored pairs: {scored_pairs}")
    print(f"Screens generated: {len(displayed_screens)}")
    print(f"Actions log: {run_dir / 'actions.csv'}")
    print(f"Decisions: {run_dir / 'decisions.csv'}")
    print(f"Scoreboard: {run_dir / 'scoreboard.csv'}")
    print(f"Inference stats: {run_dir / 'inference_stats.csv'}")
    print(f"Tradeoff report: {run_dir / 'tradeoff_report.csv'}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"Inventory: {run_dir / 'model_inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
