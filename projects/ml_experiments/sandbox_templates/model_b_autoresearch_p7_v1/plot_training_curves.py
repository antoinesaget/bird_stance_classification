#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import HEADS

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fold training curves for a sandbox run")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def resolve_run_dir(run_id: str) -> Path:
    matches = sorted((ROOT / "runs").glob(f"{run_id}*"))
    if not matches:
        raise FileNotFoundError(run_id)
    return matches[0]


def _load_history(fold_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    payload = json.loads((fold_dir / "candidate" / "training_history.json").read_text(encoding="utf-8"))
    history = pd.DataFrame(payload.get("history") or [])
    summary = payload.get("summary") or {}
    selection = payload.get("selection") or {}
    return history, summary, selection


def _transition_spike_ratio(frame: pd.DataFrame, unfreeze_step: int) -> float | None:
    if frame.empty or unfreeze_step <= 0:
        return None
    pre = frame[frame["step"] < unfreeze_step]["val_total_loss"].tail(5)
    post = frame[frame["step"] >= unfreeze_step]["val_total_loss"].head(1)
    if pre.empty or post.empty:
        return None
    baseline = float(pre.median())
    if baseline <= 0:
        return None
    return float(post.iloc[0] / baseline)


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_id)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    total_loss_rows: list[dict] = []
    search_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    generated_plots: list[str] = []

    for fold_dir in sorted((run_dir / "folds").glob("fold_*")):
        history_path = fold_dir / "candidate" / "training_history.json"
        if not history_path.exists():
            continue
        frame, summary, selection = _load_history(fold_dir)
        if frame.empty:
            continue
        frame["fold_id"] = fold_dir.name
        unfreeze_step = int(summary.get("unfreeze_step", 0))
        best_step = int(selection.get("step", 0))
        total_loss_rows.extend(frame[["fold_id", "step", "train_total_loss", "val_total_loss"]].to_dict(orient="records"))
        search_rows.extend(frame[["fold_id", "step", "val_search_score"]].to_dict(orient="records"))

        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        ax.plot(frame["step"], frame["train_total_loss"], label="train_total_loss", color="#1d3557")
        ax.plot(frame["step"], frame["val_total_loss"], label="val_total_loss", color="#e76f51")
        if unfreeze_step > 0:
            ax.axvline(unfreeze_step, color="#264653", linestyle="--", label="unfreeze")
        if best_step > 0:
            ax.axvline(best_step, color="#2a9d8f", linestyle=":", label="best_step")
        ax.set_title(f"Train vs Val Loss {fold_dir.name}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        path = plots_dir / f"{fold_dir.name}_total_loss.png"
        fig.savefig(path, bbox_inches="tight")
        generated_plots.append(str(path))
        plt.close(fig)

        fig, axes = plt.subplots(len(HEADS), 1, figsize=(8, 10), dpi=160, sharex=True)
        for idx, head in enumerate(HEADS):
            axes[idx].plot(frame["step"], frame["train_head_losses"].map(lambda item: item[head]), label=f"{head}_train", color="#457b9d")
            axes[idx].plot(frame["step"], frame["val_head_losses"].map(lambda item: item[head]), label=f"{head}_val", color="#f4a261")
            if unfreeze_step > 0:
                axes[idx].axvline(unfreeze_step, color="#264653", linestyle="--")
            if best_step > 0:
                axes[idx].axvline(best_step, color="#2a9d8f", linestyle=":")
            axes[idx].set_ylabel(head)
            axes[idx].legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("Step")
        fig.suptitle(f"Per-Head Losses {fold_dir.name}")
        fig.tight_layout()
        path = plots_dir / f"{fold_dir.name}_head_losses.png"
        fig.savefig(path, bbox_inches="tight")
        generated_plots.append(str(path))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        ax.plot(frame["step"], frame["val_search_score"], label="val_search_score", color="#2a9d8f")
        if unfreeze_step > 0:
            ax.axvline(unfreeze_step, color="#264653", linestyle="--", label="unfreeze")
        if best_step > 0:
            ax.axvline(best_step, color="#e63946", linestyle=":", label="best_step")
        ax.set_title(f"Validation Search Score {fold_dir.name}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Search score")
        ax.legend()
        fig.tight_layout()
        path = plots_dir / f"{fold_dir.name}_search_score.png"
        fig.savefig(path, bbox_inches="tight")
        generated_plots.append(str(path))
        plt.close(fig)

        diagnostics_rows.append(
            {
                "fold_id": fold_dir.name,
                "transition_spike_ratio": _transition_spike_ratio(frame, unfreeze_step),
                "best_step": best_step,
                "final_step": int(summary.get("final_step", int(frame["step"].max()))),
            }
        )

    if not total_loss_rows:
        raise RuntimeError("No training histories found")

    total_loss_frame = pd.DataFrame(total_loss_rows)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for fold_id, fold_df in total_loss_frame.groupby("fold_id"):
        ax.plot(fold_df["step"], fold_df["train_total_loss"], alpha=0.35, label=f"{fold_id} train")
        ax.plot(fold_df["step"], fold_df["val_total_loss"], alpha=0.9, linestyle="--", label=f"{fold_id} val")
    ax.set_title("Train vs Val Total Loss Across Folds")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = plots_dir / "all_folds_train_vs_val_total_loss.png"
    fig.savefig(path, bbox_inches="tight")
    generated_plots.append(str(path))
    plt.close(fig)

    search_frame = pd.DataFrame(search_rows)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for fold_id, fold_df in search_frame.groupby("fold_id"):
        ax.plot(fold_df["step"], fold_df["val_search_score"], label=fold_id)
    ax.set_title("Validation Search Score Across Folds")
    ax.set_xlabel("Step")
    ax.set_ylabel("Search score")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "all_folds_val_search_score.png"
    fig.savefig(path, bbox_inches="tight")
    generated_plots.append(str(path))
    plt.close(fig)

    diagnostics = pd.DataFrame(diagnostics_rows)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.bar(diagnostics["fold_id"], diagnostics["transition_spike_ratio"].fillna(0.0), color="#e76f51")
    ax.axhline(2.0, color="#264653", linestyle="--", label="acceptance max")
    ax.set_title("Phase Transition Diagnostics")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Transition spike ratio")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "phase_transition_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    generated_plots.append(str(path))
    plt.close(fig)

    report = {
        "run_id": run_dir.name,
        "plots": sorted(generated_plots),
        "diagnostics": diagnostics_rows,
    }
    (run_dir / "training_curves_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
