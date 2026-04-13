#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _load_best(path_name: str) -> dict:
    path = ROOT / "best" / path_name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_thresholds() -> dict:
    path = ROOT / "baselines" / "reference_thresholds.json"
    return json.loads(path.read_text(encoding="utf-8"))


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


def _fold_dynamics(run_dir: Path, thresholds: dict) -> dict:
    acceptance = thresholds["dynamics_acceptance"]
    criterion = acceptance.get("criterion", "selected_checkpoint_stability")
    fold_rows = []
    for fold_dir in sorted((run_dir / "folds").glob("fold_*")):
        history_path = fold_dir / "candidate" / "training_history.json"
        if not history_path.exists():
            continue
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload.get("history") or [])
        summary = payload.get("summary") or {}
        selection = payload.get("selection") or {}
        if frame.empty:
            continue
        final_step = int(summary.get("final_step", int(frame["step"].max())))
        best_step = int(selection.get("step", 0))
        best_val_loss = float(selection.get("val_total_loss", frame["val_total_loss"].min()))
        final_val_loss = float(frame["val_total_loss"].iloc[-1])
        best_val_score = float(selection.get("val_search_score", frame["val_search_score"].max()))
        final_val_score = float(frame["val_search_score"].iloc[-1])
        spike_ratio = _transition_spike_ratio(frame, int(summary.get("unfreeze_step", 0)))
        best_after_min_fraction = best_step >= int(acceptance["best_step_min_fraction"] * max(1, final_step))
        final_loss_within_limit = final_val_loss <= best_val_loss * (1.0 + acceptance["final_val_loss_worse_pct_max"])
        final_score_retention = final_val_score / best_val_score if best_val_score > 0 else 0.0
        final_score_within_limit = final_score_retention >= acceptance.get("final_val_search_score_retention_min", 0.95)
        runaway_divergence = bool(len(frame) >= 6 and frame["val_total_loss"].tail(max(3, len(frame) // 3)).mean() > best_val_loss * 1.20)
        selected_checkpoint_exists = bool(selection.get("step") is not None)
        checkpoint_stability_ok = bool(
            selected_checkpoint_exists
            and best_after_min_fraction
            and (spike_ratio is None or spike_ratio <= acceptance["transition_spike_ratio_max"])
        )
        fold_rows.append(
            {
                "fold_id": fold_dir.name,
                "selected_by": str(selection.get("selected_by", "")),
                "best_step": best_step,
                "final_step": final_step,
                "best_val_loss": best_val_loss,
                "final_val_loss": final_val_loss,
                "best_vs_final_val_loss_delta": final_val_loss - best_val_loss,
                "best_val_search_score": best_val_score,
                "final_val_search_score": final_val_score,
                "best_vs_final_val_search_score_delta": final_val_score - best_val_score,
                "final_val_search_score_retention": final_score_retention,
                "transition_spike_ratio": spike_ratio,
                "selected_checkpoint_exists": selected_checkpoint_exists,
                "checkpoint_stability_ok": checkpoint_stability_ok,
                "best_after_min_fraction": bool(best_after_min_fraction),
                "final_loss_within_limit": bool(final_loss_within_limit),
                "final_score_within_limit": bool(final_score_within_limit),
                "runaway_divergence": runaway_divergence,
            }
        )

    frame = pd.DataFrame(fold_rows)
    if frame.empty:
        return {"folds": [], "acceptance": {"passes": False, "reasons": ["no_history"]}}

    min_passing_folds = int(acceptance["minimum_passing_folds"])
    passing_transition = int((frame["transition_spike_ratio"].fillna(0.0) <= acceptance["transition_spike_ratio_max"]).sum())
    passing_selected_checkpoint = int(frame["selected_checkpoint_exists"].sum())
    passing_checkpoint_stability = int(frame["checkpoint_stability_ok"].sum())
    passing_best_step = int(frame["best_after_min_fraction"].sum())
    passing_final_loss = int(frame["final_loss_within_limit"].sum())
    passing_final_score = int(frame["final_score_within_limit"].sum())
    passing_no_divergence = int((~frame["runaway_divergence"]).sum())
    if criterion == "selected_checkpoint_stability":
        passes = (
            passing_selected_checkpoint >= min_passing_folds
            and passing_transition >= min_passing_folds
            and passing_best_step >= min_passing_folds
            and passing_checkpoint_stability >= min_passing_folds
        )
    else:
        passes = (
            passing_transition >= min_passing_folds
            and passing_best_step >= min_passing_folds
            and passing_final_score >= min_passing_folds
            and passing_no_divergence >= min_passing_folds
        )
    return {
        "folds": fold_rows,
        "aggregate": {
            "criterion": criterion,
            "passing_transition_folds": passing_transition,
            "passing_selected_checkpoint_folds": passing_selected_checkpoint,
            "passing_checkpoint_stability_folds": passing_checkpoint_stability,
            "passing_best_step_folds": passing_best_step,
            "passing_final_loss_folds": passing_final_loss,
            "passing_final_score_folds": passing_final_score,
            "passing_no_divergence_folds": passing_no_divergence,
            "minimum_passing_folds": min_passing_folds,
        },
        "acceptance": {
            "passes": bool(passes),
        },
    }


def _write_baseline_dynamics(best_research: dict, thresholds: dict) -> dict:
    if not best_research or not best_research.get("run_id"):
        payload = {"status": "no_best_research"}
        (ROOT / "baseline_dynamics_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (ROOT / "baseline_dynamics_report.md").write_text("# Baseline Dynamics Report\n\nNo kept research run yet.\n", encoding="utf-8")
        return payload

    run_dir = ROOT / "runs" / str(best_research["run_id"])
    dynamics = _fold_dynamics(run_dir, thresholds)
    mean_search_score = float(best_research.get("search_score", 0.0))
    score_floor = float(thresholds["baseline_anchor"]["score_floor"])
    score_pass = mean_search_score >= score_floor
    guardrail_pass = best_research.get("guardrail_status") == "pass"
    acceptance = {
        "dynamics_pass": bool(dynamics["acceptance"]["passes"]),
        "score_pass": bool(score_pass),
        "guardrail_pass": bool(guardrail_pass),
        "overall_pass": bool(dynamics["acceptance"]["passes"] and score_pass and guardrail_pass),
    }
    payload = {
        "run_id": best_research["run_id"],
        "search_score": mean_search_score,
        "score_floor": score_floor,
        "folds": dynamics.get("folds", []),
        "aggregate": dynamics.get("aggregate", {}),
        "acceptance": acceptance,
    }
    (ROOT / "baseline_dynamics_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Baseline Dynamics Report",
        "",
        f"- run_id: `{best_research['run_id']}`",
        f"- search_score: `{mean_search_score:.6f}`",
        f"- score_floor: `{score_floor:.6f}`",
        f"- dynamics_pass: `{acceptance['dynamics_pass']}`",
        f"- score_pass: `{acceptance['score_pass']}`",
        f"- guardrail_pass: `{acceptance['guardrail_pass']}`",
        f"- overall_pass: `{acceptance['overall_pass']}`",
        f"- dynamics_criterion: `{dynamics.get('aggregate', {}).get('criterion', 'selected_checkpoint_stability')}`",
        "- interpretation: validation search score is primary; validation loss is a calibration and overfit diagnostic.",
        "- decision rule: judge the selected checkpoint we would actually keep, not the unused tail after the peak.",
        "",
        markdown_table(pd.DataFrame(payload["folds"])) if payload["folds"] else "No fold diagnostics available.",
    ]
    (ROOT / "baseline_dynamics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    results_path = ROOT / "results.tsv"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    frame = pd.read_csv(results_path, sep="\t")
    if frame.empty:
        raise RuntimeError("results.tsv has no runs yet")

    numeric_columns = [
        "search_score",
        "stance_f1",
        "behavior_f1",
        "substrate_f1",
        "readability_f1",
        "specie_f1",
        "stance_acc",
        "peak_vram_gb",
        "wall_seconds",
    ]
    for column in numeric_columns:
        frame[column] = frame[column].astype(float)

    frame["guardrail_rank"] = (frame["guardrail_status"] == "pass").astype(int)
    frame["servable_rank"] = (frame["export_status"] == "ready").astype(int)
    leaderboard = frame.sort_values(
        ["search_score", "guardrail_rank", "servable_rank", "peak_vram_gb"],
        ascending=[False, False, False, True],
    ).drop(columns=["guardrail_rank", "servable_rank"])

    best_research = _load_best("research_best.json")
    best_servable = _load_best("servable_best.json")
    thresholds = _load_thresholds()
    baseline_dynamics = _write_baseline_dynamics(best_research, thresholds)

    leaderboard_csv = ROOT / "leaderboard.csv"
    leaderboard_md = ROOT / "leaderboard.md"
    leaderboard.to_csv(leaderboard_csv, index=False)
    leaderboard_md.write_text(
        "\n".join(
            [
                "# Sandbox Leaderboard",
                "",
                f"- best research run: `{best_research.get('run_id')}`",
                f"- best servable-now run: `{best_servable.get('run_id')}`",
                "",
                markdown_table(leaderboard.head(20)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plots_dir = ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.plot(range(len(frame)), frame["search_score"], marker="o", color="#2a9d8f")
    ax.set_title("Objective Over Time")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Search score")
    fig.tight_layout()
    fig.savefig(plots_dir / "objective_over_time.png", bbox_inches="tight")
    plt.close(fig)

    top = leaderboard.head(8)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    top_plot = top[["run_id", "stance_f1", "behavior_f1", "substrate_f1", "readability_f1", "specie_f1"]].set_index("run_id")
    top_plot.plot(kind="bar", ax=ax)
    ax.set_title("F1 Breakdown Top Runs")
    ax.set_ylabel("F1")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots_dir / "f1_breakdown_top_runs.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    colors = ["#2a9d8f" if value == "ready" else "#8d99ae" for value in frame["export_status"]]
    ax.scatter(frame["peak_vram_gb"], frame["search_score"], c=colors)
    ax.set_title("VRAM vs Search Score")
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("Search score")
    fig.tight_layout()
    fig.savefig(plots_dir / "vram_vs_score.png", bbox_inches="tight")
    plt.close(fig)

    for _, row in leaderboard.head(3).iterrows():
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        ax.axis("off")
        text = (
            f"Run: {row['run_id']}\n"
            f"Score: {row['search_score']:.4f}\n"
            f"Stance F1: {row['stance_f1']:.4f}\n"
            f"Behavior F1: {row['behavior_f1']:.4f}\n"
            f"Substrate F1: {row['substrate_f1']:.4f}\n"
            f"Readability F1: {row['readability_f1']:.4f}\n"
            f"Specie F1: {row['specie_f1']:.4f}\n"
            f"Export: {row['export_status']}\n"
            f"Status: {row['keep_discard']}\n"
            f"Description: {row['description']}"
        )
        ax.text(0.02, 0.95, text, va="top", ha="left", fontsize=11, family="monospace")
        fig.tight_layout()
        fig.savefig(plots_dir / f"top_run_card_{row['run_id']}.png", bbox_inches="tight")
        plt.close(fig)

    summary = {
        "results_tsv": str(results_path),
        "leaderboard_csv": str(leaderboard_csv),
        "leaderboard_md": str(leaderboard_md),
        "baseline_dynamics_report_json": str(ROOT / "baseline_dynamics_report.json"),
        "baseline_dynamics_report_md": str(ROOT / "baseline_dynamics_report.md"),
        "plots": sorted(str(path) for path in plots_dir.glob("*.png")),
        "top_run": leaderboard.iloc[0].to_dict(),
        "best_research": best_research,
        "best_servable": best_servable,
        "baseline_dynamics": baseline_dynamics,
    }
    (ROOT / "inspect_results_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
