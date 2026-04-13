from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "sandbox_templates" / "model_b_autoresearch_p7_v1"
if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))

import inspect_results  # noqa: E402
import prepare  # noqa: E402


def make_frame() -> pd.DataFrame:
    rows = []
    for idx in range(30):
        rows.append(
            {
                "row_id": f"row_{idx:04d}",
                "group_id": f"group_{idx // 3}",
                "image_id": f"img_{idx // 3}",
                "crop_path": "/tmp/fake.jpg",
                "isbird": "yes",
                "readability": "readable",
                "specie": "correct",
                "behavior": "resting" if idx % 4 else "moving",
                "substrate": "water" if idx % 4 else "vegetation",
                "stance": "bipedal" if idx % 5 else "unipedal",
            }
        )
    return pd.DataFrame(rows)


def test_inner_validation_split_is_deterministic_and_group_safe():
    frame = make_frame()
    train_a, val_a, frac_a = prepare.build_inner_validation_split(frame)
    train_b, val_b, frac_b = prepare.build_inner_validation_split(frame)

    assert frac_a == frac_b
    assert set(train_a["row_id"]) == set(train_b["row_id"])
    assert set(val_a["row_id"]) == set(val_b["row_id"])
    assert set(train_a["group_id"]).isdisjoint(set(val_a["group_id"]))
    assert len(train_a) + len(val_a) == len(frame)
    assert len(val_a) > 0


def test_fold_dynamics_acceptance_flags(tmp_path: Path):
    run_dir = tmp_path / "runs" / "demo"
    candidate_dir = run_dir / "folds" / "fold_0" / "candidate"
    candidate_dir.mkdir(parents=True)
    history = [
        {
            "phase": "head",
            "step": 1,
            "epoch_fraction": 0.1,
            "elapsed_s": 1.0,
            "lr": 0.001,
            "train_total_loss": 8.0,
            "val_total_loss": 7.0,
            "train_head_losses": {head: 1.0 for head in ["readability", "specie", "behavior", "substrate", "stance"]},
            "val_head_losses": {head: 1.0 for head in ["readability", "specie", "behavior", "substrate", "stance"]},
            "val_metrics_snapshot": {"readability_f1": 0.31, "specie_f1": 0.31},
            "val_search_score": 0.10,
            "best_so_far": False,
        },
        {
            "phase": "head",
            "step": 10,
            "epoch_fraction": 0.3,
            "elapsed_s": 5.0,
            "lr": 0.001,
            "train_total_loss": 6.0,
            "val_total_loss": 5.2,
            "train_head_losses": {head: 0.8 for head in ["readability", "specie", "behavior", "substrate", "stance"]},
            "val_head_losses": {head: 0.9 for head in ["readability", "specie", "behavior", "substrate", "stance"]},
            "val_metrics_snapshot": {"readability_f1": 0.32, "specie_f1": 0.32},
            "val_search_score": 0.12,
            "best_so_far": True,
        },
        {
            "phase": "warmup",
            "step": 20,
            "epoch_fraction": 0.6,
            "elapsed_s": 9.0,
            "lr": 0.00003,
            "train_total_loss": 5.8,
            "val_total_loss": 5.1,
            "train_head_losses": {head: 0.7 for head in ["readability", "specie", "behavior", "substrate", "stance"]},
            "val_head_losses": {head: 0.8 for head in ["readability", "specie", "behavior", "substrate", "stance"]},
            "val_metrics_snapshot": {"readability_f1": 0.33, "specie_f1": 0.33},
            "val_search_score": 0.13,
            "best_so_far": True,
        },
        {
            "phase": "finetune",
            "step": 30,
            "epoch_fraction": 1.0,
            "elapsed_s": 14.0,
            "lr": 0.00002,
            "train_total_loss": 5.4,
            "val_total_loss": 5.3,
            "train_head_losses": {head: 0.6 for head in ["readability", "specie", "behavior", "substrate", "stance"]},
            "val_head_losses": {head: 0.85 for head in ["readability", "specie", "behavior", "substrate", "stance"]},
            "val_metrics_snapshot": {"readability_f1": 0.34, "specie_f1": 0.34},
            "val_search_score": 0.125,
            "best_so_far": False,
        },
    ]
    payload = {
        "history": history,
        "summary": {"unfreeze_step": 20, "final_step": 30},
        "selection": {"step": 20, "val_total_loss": 5.1, "val_search_score": 0.13},
    }
    (candidate_dir / "training_history.json").write_text(json.dumps(payload), encoding="utf-8")
    thresholds = {
        "dynamics_acceptance": {
            "criterion": "selected_checkpoint_stability",
            "transition_spike_ratio_max": 2.0,
            "best_step_min_fraction": 0.20,
            "final_val_loss_worse_pct_max": 0.10,
            "minimum_passing_folds": 1,
        },
        "baseline_anchor": {"score_floor": 0.11407},
    }

    report = inspect_results._fold_dynamics(run_dir, thresholds)

    assert report["acceptance"]["passes"] is True
    assert report["aggregate"]["criterion"] == "selected_checkpoint_stability"
    assert report["aggregate"]["passing_selected_checkpoint_folds"] == 1
    assert report["aggregate"]["passing_checkpoint_stability_folds"] == 1
    assert report["aggregate"]["passing_transition_folds"] == 1
    assert report["aggregate"]["passing_best_step_folds"] == 1
