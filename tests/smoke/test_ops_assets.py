from __future__ import annotations

import py_compile
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_compose_files_have_expected_services() -> None:
    expected = {
        REPO_ROOT / "deploy/docker-compose.local.yml": {"postgres", "label-studio", "ml-backend"},
        REPO_ROOT / "deploy/docker-compose.iats-ml.yml": {"ml-backend"},
        REPO_ROOT / "deploy/docker-compose.truenas.yml": {"postgres", "label-studio"},
    }

    for compose_path, services in expected.items():
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        assert set(data["services"]) == services


def test_env_examples_exist() -> None:
    for path in (
        REPO_ROOT / "deploy/env/local.env.example",
        REPO_ROOT / "deploy/env/iats.env.example",
        REPO_ROOT / "deploy/env/truenas.env.example",
    ):
        assert path.is_file(), path


def test_ops_shell_scripts_parse() -> None:
    shell_scripts = [
        REPO_ROOT / "scripts/ops/common.sh",
        REPO_ROOT / "scripts/ops/local_compose.sh",
        REPO_ROOT / "scripts/ops/remote_repo_exec.sh",
        REPO_ROOT / "scripts/ops/git_pull_ff_only.sh",
        REPO_ROOT / "scripts/ops/iats_deploy_ml_remote.sh",
        REPO_ROOT / "scripts/ops/iats_deploy_model_b_remote.sh",
        REPO_ROOT / "scripts/ops/iats_train_remote.sh",
        REPO_ROOT / "scripts/ops/iats_normalize_annotations_remote.sh",
        REPO_ROOT / "scripts/ops/iats_build_attributes_dataset_remote.sh",
        REPO_ROOT / "scripts/ops/iats_evaluate_model_b_remote.sh",
        REPO_ROOT / "scripts/ops/iats_train_attributes_cv_remote.sh",
        REPO_ROOT / "scripts/ops/iats_train_attributes_final_remote.sh",
        REPO_ROOT / "scripts/ops/iats_promote_model_remote.sh",
        REPO_ROOT / "scripts/ops/truenas_deploy_ui_remote.sh",
        REPO_ROOT / "scripts/ops/truenas_create_project_remote.sh",
        REPO_ROOT / "scripts/ops/truenas_export_annotations_remote.sh",
        REPO_ROOT / "scripts/ops/truenas_prefill_lines_predictions_remote.sh",
        REPO_ROOT / "scripts/ops/truenas_refresh_lines_predictions_remote.sh",
        REPO_ROOT / "scripts/ops/iats_sync_data.sh",
        REPO_ROOT / "scripts/ops/iats_import_exports.sh",
        REPO_ROOT / "scripts/ops/smoke_remote.sh",
    ]

    for script in shell_scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_ops_python_scripts_compile() -> None:
    for path in (
        REPO_ROOT / "scripts/create_autoresearch_sandbox.py",
        REPO_ROOT / "scripts/export_labelstudio_snapshot.py",
        REPO_ROOT / "scripts/create_labelstudio_project.py",
        REPO_ROOT / "scripts/prefill_labelstudio_predictions.py",
        REPO_ROOT / "scripts/promote_model.py",
        REPO_ROOT / "scripts/evaluate_model_b_checkpoint.py",
        REPO_ROOT / "scripts/train_attributes.py",
        REPO_ROOT / "scripts/train_attributes_cv.py",
    ):
        py_compile.compile(str(path), doraise=True)


def test_makefile_has_core_remote_targets() -> None:
    content = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "iats-evaluate-model-b:" in content
    assert "truenas-refresh-lines-predictions:" in content
