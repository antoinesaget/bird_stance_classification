#!/usr/bin/env python3
"""Blind N-model comparison app (1xN) with click-to-mark bad predictions.

Controls:
- Left click panel: toggle "bad prediction" for that panel
- 1..9: toggle bad for panel 1..9
- Space: confirm current image and move to next
- C: clear all bad marks on current image
- N: finish run now
- Esc: quit
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import random
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from compare_detection_models import (
    LoadedModel,
    ModelSpec,
    PredictionMetrics,
    TimingStats,
    infer_repo_root,
    iter_images,
    load_models,
    parse_model_specs,
    pick_images,
    resolve_images_dir,
    run_prediction,
    write_inventory,
)

DEFAULT_MODELS = "yolo11l,yolo26l,yolo11m,rtdetr-l,yolo26x"


@dataclass
class FiveScreen:
    screen_index: int
    image_path: pathlib.Path
    panel_model_keys: list[str]
    frames: list[np.ndarray]
    metrics: list[PredictionMetrics]


@dataclass
class ModelClickStats:
    shown: int = 0
    bad_votes: int = 0

    @property
    def good_votes(self) -> int:
        return self.shown - self.bad_votes

    @property
    def bad_rate(self) -> float:
        if self.shown <= 0:
            return 0.0
        return self.bad_votes / self.shown

    @property
    def good_rate(self) -> float:
        if self.shown <= 0:
            return 0.0
        return self.good_votes / self.shown


@dataclass
class UiState:
    panel_regions: list[tuple[int, int, int, int]]
    bad_flags: list[bool]
    render_width: int = 0
    render_height: int = 0
    dirty: bool = True


def fit_to_box(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((out_h, out_w, 3), 18, dtype=np.uint8)
    # Blur-fill background to avoid black letterbox bars.
    background = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=16, sigmaY=16)
    background = cv2.addWeighted(background, 0.40, np.full_like(background, 18), 0.60, 0.0)
    scale = min(out_w / w, out_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = background
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def render_panel(frame: np.ndarray, panel_index: int, panel_w: int, panel_h: int, is_bad: bool) -> np.ndarray:
    panel = fit_to_box(frame, panel_w, panel_h)
    title_h = 42
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w - 1, title_h), (12, 14, 18), -1)
    panel = cv2.addWeighted(overlay, 0.65, panel, 0.35, 0.0)
    cv2.putText(
        panel,
        f"Panel {panel_index}",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (238, 238, 238),
        2,
        cv2.LINE_AA,
    )

    if is_bad:
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w - 1, panel_h - 1), (40, 40, 185), -1)
        panel = cv2.addWeighted(overlay, 0.13, panel, 0.87, 0.0)
        cv2.rectangle(panel, (panel_w - 98, 10), (panel_w - 12, 38), (34, 34, 205), -1)
        cv2.putText(panel, "BAD", (panel_w - 88, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        border_color = (44, 44, 224)
        border_thickness = 5
    else:
        border_color = (98, 98, 108)
        border_thickness = 2

    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), border_color, border_thickness)
    return panel


def compose_board(
    screen: FiveScreen,
    bad_flags: list[bool],
    scored_images: int,
    total_images: int,
    render_width: int,
    render_height: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    board_w = max(1000, render_width)
    board_h = max(520, render_height)
    board = np.full((board_h, board_w, 3), (16, 18, 22), dtype=np.uint8)

    outer_pad = 12
    header_h = max(92, min(136, int(board_h * 0.18)))
    gap = 10
    panel_count = len(screen.frames)
    usable_w = max(600, board_w - outer_pad * 2)
    panel_w = max(180, int((usable_w - gap * (panel_count - 1)) / panel_count))
    panel_h = max(260, board_h - header_h - outer_pad * 2)
    row_w = panel_count * panel_w + (panel_count - 1) * gap
    x_start = max(outer_pad, (board_w - row_w) // 2)
    y_start = outer_pad + header_h

    cv2.rectangle(board, (0, 0), (board_w - 1, header_h), (10, 12, 16), -1)
    panel_regions: list[tuple[int, int, int, int]] = []

    for idx in range(panel_count):
        panel_img = render_panel(
            frame=screen.frames[idx],
            panel_index=idx + 1,
            panel_w=panel_w,
            panel_h=panel_h,
            is_bad=bad_flags[idx],
        )
        x0 = x_start + idx * (panel_w + gap)
        x1 = x0 + panel_w
        board[y_start : y_start + panel_h, x0:x1] = panel_img
        panel_regions.append((x0, y_start, x1, y_start + panel_h))

    marked_bad = sum(1 for item in bad_flags if item)
    panel_count = len(screen.frames)
    key_hint_end = min(panel_count, 9)
    if key_hint_end > 0:
        key_hint = f"1-{key_hint_end}"
    else:
        key_hint = "1-9"
    lines = [
        f"Blind model review ({panel_count} panels)  |  Image {screen.screen_index + 1}/{total_images}  |  {screen.image_path.name}",
        f"Marked BAD: {marked_bad}/{panel_count}   Completed: {scored_images}",
        f"Controls: click panel or keys {key_hint} toggle BAD  |  Space next  |  C clear  |  N finish  |  Esc quit",
    ]
    y = 34
    cv2.putText(board, lines[0], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(board, lines[1], (14, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (220, 230, 235), 2, cv2.LINE_AA)
    cv2.putText(board, lines[2], (14, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (190, 205, 215), 2, cv2.LINE_AA)

    return board, panel_regions


def make_mouse_callback(state: UiState, window_name: str) -> Callable[[int, int, int, int, object], None]:
    def _callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not state.panel_regions:
            return

        adj_x, adj_y = x, y
        if state.render_width > 0 and state.render_height > 0:
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
                if win_w > 0 and win_h > 0 and (win_w != state.render_width or win_h != state.render_height):
                    adj_x = int(round(x * state.render_width / win_w))
                    adj_y = int(round(y * state.render_height / win_h))
            except Exception:  # noqa: BLE001
                pass

        for idx, (x0, y0, x1, y1) in enumerate(state.panel_regions):
            if x0 <= adj_x < x1 and y0 <= adj_y < y1:
                state.bad_flags[idx] = not state.bad_flags[idx]
                state.dirty = True
                break

    return _callback


def get_window_render_size(window_name: str, fallback_width: int, fallback_height: int) -> tuple[int, int]:
    try:
        _, _, w, h = cv2.getWindowImageRect(window_name)
        if w >= 320 and h >= 240:
            return int(w), int(h)
    except Exception:  # noqa: BLE001
        pass
    return fallback_width, fallback_height


def wait_for_user_action(
    state: UiState,
    window_name: str,
    board_builder: Callable[[list[bool], int, int], tuple[np.ndarray, list[tuple[int, int, int, int]]]],
    fallback_width: int,
    fallback_height: int,
) -> str:
    while True:
        if state.dirty:
            render_w, render_h = get_window_render_size(window_name, fallback_width, fallback_height)
            board, regions = board_builder(state.bad_flags, render_w, render_h)
            state.panel_regions = regions
            state.render_width = board.shape[1]
            state.render_height = board.shape[0]
            cv2.imshow(window_name, board)
            state.dirty = False

        key = cv2.waitKey(30)
        if key < 0:
            continue
        code = key & 0xFF
        if code in (32, 10, 13):
            return "next"
        if ord("1") <= code <= ord("9"):
            idx = code - ord("1")
            if idx < len(state.bad_flags):
                state.bad_flags[idx] = not state.bad_flags[idx]
                state.dirty = True
            continue
        if code in (ord("c"), ord("C")):
            for idx in range(len(state.bad_flags)):
                state.bad_flags[idx] = False
            state.dirty = True
            continue
        if code in (ord("n"), ord("N")):
            return "finish"
        if code == 27:
            return "quit"


def prepare_five_screen(
    screen_index: int,
    image_path: pathlib.Path,
    panel_model_keys: list[str],
    loaded_by_key: dict[str, LoadedModel],
    args: argparse.Namespace,
) -> FiveScreen:
    frames: list[np.ndarray] = []
    metrics: list[PredictionMetrics] = []

    for key in panel_model_keys:
        loaded = loaded_by_key[key]
        try:
            frame, metric = run_prediction(loaded, image_path, args)
        except Exception as exc:  # noqa: BLE001
            fallback = cv2.imread(str(image_path))
            if fallback is None:
                fallback = np.zeros((640, 960, 3), dtype=np.uint8)
            frame = fallback
            metric = PredictionMetrics(error=str(exc))
        frames.append(frame)
        metrics.append(metric)

    return FiveScreen(
        screen_index=screen_index,
        image_path=image_path,
        panel_model_keys=panel_model_keys,
        frames=frames,
        metrics=metrics,
    )


def write_actions(
    run_dir: pathlib.Path,
    actions: list[dict],
    spec_by_key: dict[str, ModelSpec],
    panel_count: int,
) -> None:
    path = run_dir / "actions.csv"
    fieldnames = [
        "timestamp",
        "screen_index",
        "image_name",
        "bad_count",
    ]
    for idx in range(panel_count):
        n = idx + 1
        fieldnames.extend(
            [
                f"panel_{n}_model_key",
                f"panel_{n}_model_name",
                f"panel_{n}_bad",
                f"panel_{n}_inference_ms",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in actions:
            out = {
                "timestamp": row["timestamp"],
                "screen_index": row["screen_index"],
                "image_name": row["image_name"],
                "bad_count": row["bad_count"],
            }
            for idx in range(panel_count):
                n = idx + 1
                model_key = row[f"panel_{n}_model_key"]
                out[f"panel_{n}_model_key"] = model_key
                out[f"panel_{n}_model_name"] = spec_by_key[model_key].display_name
                out[f"panel_{n}_bad"] = row[f"panel_{n}_bad"]
                out[f"panel_{n}_inference_ms"] = f"{row[f'panel_{n}_inference_ms']:.3f}"
            writer.writerow(out)


def write_scoreboard(
    run_dir: pathlib.Path,
    stats: dict[str, ModelClickStats],
    timings: dict[str, TimingStats],
    spec_by_key: dict[str, ModelSpec],
) -> list[str]:
    ranking = sorted(
        stats.keys(),
        key=lambda key: (
            stats[key].bad_rate,
            -stats[key].good_votes,
            timings[key].mean_ms if timings[key].count > 0 else float("inf"),
        ),
    )

    path = run_dir / "scoreboard.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "model_key",
                "model_name",
                "shown",
                "bad_votes",
                "good_votes",
                "bad_rate",
                "good_rate",
                "mean_ms",
                "p90_ms",
                "min_ms",
                "max_ms",
            ],
        )
        writer.writeheader()
        for rank, key in enumerate(ranking, start=1):
            st = stats[key]
            ts = timings[key]
            writer.writerow(
                {
                    "rank": rank,
                    "model_key": key,
                    "model_name": spec_by_key[key].display_name,
                    "shown": st.shown,
                    "bad_votes": st.bad_votes,
                    "good_votes": st.good_votes,
                    "bad_rate": f"{st.bad_rate:.6f}",
                    "good_rate": f"{st.good_rate:.6f}",
                    "mean_ms": f"{ts.mean_ms:.3f}",
                    "p90_ms": f"{ts.p90_ms:.3f}",
                    "min_ms": f"{(0.0 if ts.min_ms == float('inf') else ts.min_ms):.3f}",
                    "max_ms": f"{ts.max_ms:.3f}",
                }
            )
    return ranking


def write_summary(
    run_dir: pathlib.Path,
    stop_reason: str,
    winner_key: str | None,
    stats: dict[str, ModelClickStats],
    spec_by_key: dict[str, ModelSpec],
    scored_images: int,
    total_images: int,
) -> None:
    path = run_dir / "summary.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now().isoformat(),
                "stop_reason": stop_reason,
                "winner_key": winner_key,
                "winner_name": (spec_by_key[winner_key].display_name if winner_key else None),
                "winner_bad_rate": (stats[winner_key].bad_rate if winner_key else None),
                "scored_images": scored_images,
                "total_images": total_images,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blind N-model comparison: click bad predictions, then Space for next image.",
    )
    parser.add_argument("--repo-root", default=None, help="Repository root (default: inferred).")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Image directory. Defaults to scraped_images/scolop2_10k (or scolop2_10K).",
    )
    parser.add_argument("--output-dir", default=None, help="Output root (timestamped run dir is created).")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Image pool size (<=0 means all).",
    )
    parser.add_argument("--sample-mode", choices=["random", "first"], default="random", help="Image sampling mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="Comma-separated model aliases/checkpoints. Must resolve to at least 2 loaded models.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Extra model aliases/checkpoints (repeatable; comma-separated accepted).",
    )
    parser.add_argument("--device", default="cpu", help="Inference device (cpu/mps/cuda:0).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold for Ultralytics models.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections.")
    parser.add_argument("--dino-text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--hide-labels", action="store_true", help="Hide class labels.")
    parser.add_argument("--hide-conf", action="store_true", help="Hide confidence labels.")
    parser.add_argument("--line-width", type=int, default=None, help="Bounding-box line width.")
    parser.add_argument("--prefetch-size", type=int, default=4, help="Future screens to preload.")
    parser.add_argument("--prefetch-workers", type=int, default=1, help="Background prefetch workers.")
    parser.add_argument("--display-max-width", type=int, default=2520, help="Max UI width.")
    parser.add_argument("--display-max-height", type=int, default=980, help="Max UI height.")
    parser.add_argument("--window-name", default="Blind 5-Model Review", help="OpenCV window title.")
    parser.add_argument(
        "--shuffle-panels",
        action="store_true",
        dest="shuffle_panels",
        help="Randomize panel-model mapping on each image.",
    )
    parser.add_argument(
        "--no-shuffle-panels",
        action="store_false",
        dest="shuffle_panels",
        help="Keep fixed panel-model mapping across images.",
    )
    parser.set_defaults(shuffle_panels=True)
    parser.add_argument("--dry-run-load", action="store_true", help="Only load models then exit.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.prefetch_size < 1:
        print("--prefetch-size must be >= 1", file=sys.stderr)
        return 1
    if args.prefetch_workers < 1:
        print("--prefetch-workers must be >= 1", file=sys.stderr)
        return 1

    repo_root = pathlib.Path(args.repo_root).expanduser().resolve() if args.repo_root else infer_repo_root()
    output_root = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (repo_root / "model_compare_five_click").resolve()
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
    write_inventory(run_dir, requested_specs, loaded_models, load_failures)

    if len(loaded_models) < 2:
        print(
            f"This app requires at least 2 loadable models. Loaded: {len(loaded_models)}",
            file=sys.stderr,
        )
        print("Loaded model keys:", file=sys.stderr)
        for item in loaded_models:
            print(f"  - {item.spec.key}", file=sys.stderr)
        if load_failures:
            print("Skipped/failed models:", file=sys.stderr)
            for key, reason in load_failures.items():
                print(f"  - {key}: {reason}", file=sys.stderr)
        return 1

    loaded_by_key = {item.spec.key: item for item in loaded_models}
    spec_by_key = {item.spec.key: item.spec for item in loaded_models}
    stats = {key: ModelClickStats() for key in loaded_by_key.keys()}
    timings = {key: TimingStats() for key in loaded_by_key.keys()}

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
    shortcut_max = min(len(loaded_models), 9)
    print(
        f"Blind mode active: click bad panels (or keys 1..{shortcut_max}), Space next, C clear, N finish, Esc quit."
    )
    print("")

    if args.dry_run_load:
        print("Dry run complete. Exiting after model loading.")
        return 0

    panel_rng = random.Random(args.seed + 2026)
    next_screen_index = 0
    scored_images = 0
    actions: list[dict] = []
    stop_reason = "user_quit"
    prefetch_queue: deque[Future[FiveScreen]] = deque()

    def schedule_more(executor: ThreadPoolExecutor) -> None:
        nonlocal next_screen_index
        while len(prefetch_queue) < args.prefetch_size and next_screen_index < len(image_pool):
            image_path = image_pool[next_screen_index]
            panel_model_keys = list(loaded_by_key.keys())
            if args.shuffle_panels:
                panel_rng.shuffle(panel_model_keys)
            future = executor.submit(
                prepare_five_screen,
                next_screen_index,
                image_path,
                panel_model_keys,
                loaded_by_key,
                args,
            )
            prefetch_queue.append(future)
            next_screen_index += 1

    ui_state = UiState(
        panel_regions=[],
        bad_flags=[False for _ in range(len(loaded_by_key))],
        render_width=0,
        render_height=0,
        dirty=True,
    )
    mouse_callback = make_mouse_callback(ui_state, args.window_name)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, args.display_max_width, args.display_max_height)
    cv2.setMouseCallback(args.window_name, mouse_callback)

    with ThreadPoolExecutor(max_workers=args.prefetch_workers) as executor:
        schedule_more(executor)

        while True:
            if not prefetch_queue:
                stop_reason = "queue_empty"
                break

            future = prefetch_queue.popleft()
            try:
                screen = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"Screen preparation failed: {exc}", file=sys.stderr)
                stop_reason = "screen_prepare_error"
                break

            schedule_more(executor)

            ui_state.bad_flags = [False for _ in range(len(screen.panel_model_keys))]
            ui_state.dirty = True

            def board_builder(
                flags: list[bool],
                render_w: int,
                render_h: int,
            ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
                return compose_board(
                    screen=screen,
                    bad_flags=flags,
                    scored_images=scored_images,
                    total_images=len(image_pool),
                    render_width=render_w,
                    render_height=render_h,
                )

            action = wait_for_user_action(
                state=ui_state,
                window_name=args.window_name,
                board_builder=board_builder,
                fallback_width=args.display_max_width,
                fallback_height=args.display_max_height,
            )
            if action == "quit":
                stop_reason = "user_quit"
                break
            if action == "finish":
                stop_reason = "manual_finish"
                break

            row: dict[str, object] = {
                "timestamp": dt.datetime.now().isoformat(),
                "screen_index": screen.screen_index + 1,
                "image_name": screen.image_path.name,
                "bad_count": 0,
            }
            bad_count = 0
            for idx, key in enumerate(screen.panel_model_keys):
                marked_bad = bool(ui_state.bad_flags[idx])
                if marked_bad:
                    bad_count += 1
                stats[key].shown += 1
                if marked_bad:
                    stats[key].bad_votes += 1
                timings[key].add(screen.metrics[idx].inference_ms)

                panel_num = idx + 1
                row[f"panel_{panel_num}_model_key"] = key
                row[f"panel_{panel_num}_bad"] = int(marked_bad)
                row[f"panel_{panel_num}_inference_ms"] = screen.metrics[idx].inference_ms

            row["bad_count"] = bad_count
            actions.append(row)
            scored_images += 1

        for pending in prefetch_queue:
            pending.cancel()

    cv2.destroyAllWindows()

    write_actions(run_dir, actions, spec_by_key, panel_count=len(loaded_by_key))
    ranking = write_scoreboard(run_dir, stats, timings, spec_by_key)
    winner_key = ranking[0] if ranking else None
    write_summary(
        run_dir=run_dir,
        stop_reason=stop_reason,
        winner_key=winner_key,
        stats=stats,
        spec_by_key=spec_by_key,
        scored_images=scored_images,
        total_images=len(image_pool),
    )

    print("")
    print("Model click review completed.")
    print(f"Stop reason: {stop_reason}")
    if winner_key:
        winner_stats = stats[winner_key]
        print(
            f"Best (lowest bad rate): {winner_key} ({spec_by_key[winner_key].display_name}) "
            f"bad_rate={winner_stats.bad_rate:.1%} good={winner_stats.good_votes}/{winner_stats.shown}"
        )
    else:
        print("Best model: n/a")
    print(f"Scored images: {scored_images}")
    print(f"Actions log: {run_dir / 'actions.csv'}")
    print(f"Scoreboard: {run_dir / 'scoreboard.csv'}")
    print(f"Summary: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
