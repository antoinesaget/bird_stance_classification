from __future__ import annotations

import json
from pathlib import Path

import birdsys.ml_experiments.audit_served_model_b as audit_mod


def _arg_value(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _evaluation_summary(*, primary: float, balanced: float, head_macro_f1: float) -> dict:
    summary_metrics = {
        f"{head}_macro_f1": head_macro_f1
        for head in ("readability", "specie", "behavior", "substrate", "stance")
    }
    return {
        "aggregate_metrics": {
            "primary_score": primary,
            "mean_macro_f1": primary,
            "support_weighted_macro_f1": primary,
            "mean_balanced_accuracy": balanced,
        },
        "summary_metrics": summary_metrics,
        "diagnostics": {"rows_scored": 10},
    }


def test_audit_served_model_b_runs_orchestrated_workflow_and_writes_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "train_attributes_served_audit.yaml"
    config_path.write_text("backbone: convnextv2_nano.fcmae_ft_in1k\n", encoding="utf-8")

    dataset_dirs = []
    for name in ("ds_v002", "ds_v001"):
        dataset_dir = tmp_path / name
        dataset_dir.mkdir()
        dataset_dirs.append(dataset_dir)

    served_dir = tmp_path / "served_model_b" / "current"
    served_dir.mkdir(parents=True)
    promotion = {
        "source_path": "/home/antoine/_archives/bird_stance_classification/2026-04-12/repo-root/.sandboxes/model_b_autoresearch_p7_v1/runs/20260326_150758_sampler-cap-2p5-v1/candidate/current_backend/checkpoint.pt"
    }
    _write_json(served_dir / "promotion.json", promotion)

    def _train_main(argv: list[str]) -> int:
        output_dir = Path(_arg_value(argv, "--output-dir"))
        dataset_name = Path(_arg_value(argv, "--dataset-dir")).name
        smoke = "--smoke" in argv
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "checkpoint.pt").write_bytes(b"checkpoint")
        primary = 0.61 if dataset_name == "ds_v002" else 0.57
        if smoke:
            primary = 0.33
        _write_json(output_dir / "summary.json", _evaluation_summary(primary=primary, balanced=primary - 0.02, head_macro_f1=primary - 0.1))
        _write_json(
            output_dir / "report.json",
            {
                "epoch_history": [
                    {
                        "epoch_index": 1,
                        "train_primary_score": primary + 0.05,
                        "eval_primary_score": primary,
                        "train_loss": 1.2,
                        "eval_loss": 1.4,
                    }
                ]
            },
        )
        (output_dir / "report.md").write_text("# train report\n", encoding="utf-8")
        _write_json(
            output_dir / "train_eval" / "summary.json",
            _evaluation_summary(primary=primary + 0.05, balanced=primary + 0.03, head_macro_f1=primary - 0.05),
        )
        (output_dir / "train_eval" / "report.md").write_text("# train eval report\n", encoding="utf-8")
        return 0

    def _cv_main(argv: list[str]) -> int:
        output_dir = Path(_arg_value(argv, "--output-dir"))
        dataset_name = Path(_arg_value(argv, "--dataset-dir")).name
        smoke = "--smoke" in argv
        output_dir.mkdir(parents=True, exist_ok=False)
        candidate_mean = 0.6 if dataset_name == "ds_v002" else 0.56
        if smoke:
            candidate_mean = 0.3
        _write_json(
            output_dir / "summary.json",
            {
                "folds": 5,
                "candidate_metrics_summary": {
                    "primary_score": {"mean": candidate_mean, "std": 0.03},
                    "mean_balanced_accuracy": {"mean": candidate_mean - 0.02, "std": 0.02},
                },
                "baseline_metrics_summary": {
                    "primary_score": {"mean": candidate_mean - 0.07, "std": 0.01},
                },
                "delta_metrics_summary": {
                    "primary_score": {"mean": 0.07, "std": 0.01},
                    "mean_balanced_accuracy": {"mean": 0.05, "std": 0.01},
                },
            },
        )
        return 0

    def _eval_main(argv: list[str]) -> int:
        output_dir = Path(_arg_value(argv, "--output-dir"))
        dataset_name = Path(_arg_value(argv, "--dataset-dir")).name
        artifact_path = Path(_arg_value(argv, "--artifact-path"))
        output_dir.mkdir(parents=True, exist_ok=False)
        is_served = artifact_path == served_dir
        if dataset_name == "ds_v002":
            primary = 0.52 if is_served else 0.59
            balanced = 0.53 if is_served else 0.58
            head_macro_f1 = 0.5 if is_served else 0.57
        else:
            primary = 0.51 if is_served else 0.55
            balanced = 0.52 if is_served else 0.54
            head_macro_f1 = 0.49 if is_served else 0.53
        _write_json(output_dir / "summary.json", _evaluation_summary(primary=primary, balanced=balanced, head_macro_f1=head_macro_f1))
        (output_dir / "report.md").write_text("# eval report\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(audit_mod, "_train_main", _train_main)
    monkeypatch.setattr(audit_mod, "_cv_main", _cv_main)
    monkeypatch.setattr(audit_mod, "_evaluate_main", _eval_main)

    output_dir = tmp_path / "audit_out"
    assert (
        audit_mod.main(
            [
                "--dataset-dirs",
                str(dataset_dirs[0]),
                str(dataset_dirs[1]),
                "--config",
                str(config_path),
                "--data-home",
                str(tmp_path / "data_home"),
                "--served-artifact-path",
                str(served_dir),
                "--output-dir",
                str(output_dir),
                "--smoke-first",
            ]
        )
        == 0
    )

    summary_json = output_dir / "audit_summary.json"
    summary_md = output_dir / "audit_summary.md"
    assert summary_json.exists()
    assert summary_md.exists()

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["provenance"]["archived_run_dir"].endswith("20260326_150758_sampler-cap-2p5-v1")
    assert payload["smoke_result"] is not None
    assert [item["dataset_name"] for item in payload["datasets"]] == ["ds_v002", "ds_v001"]
    assert payload["datasets"][0]["test_eval"]["candidate_minus_served_primary_score"] > 0.0

    markdown = summary_md.read_text(encoding="utf-8")
    assert "Candidate minus served primary score" in markdown
    assert "CV delta primary score mean" in markdown
    assert "Audit Caveats" in markdown
