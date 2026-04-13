"""Purpose: Verify that the core ops assets and deployment files still exist and parse"""
from __future__ import annotations

import py_compile
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_compose_files_have_expected_services() -> None:
    expected = {
        REPO_ROOT / "ops" / "compose" / "docker-compose.local.yml": {"postgres", "label-studio", "ml-backend"},
        REPO_ROOT / "projects" / "ml_backend" / "deploy" / "docker-compose.iats-ml.yml": {"ml-backend"},
        REPO_ROOT / "projects" / "labelstudio" / "deploy" / "docker-compose.truenas.yml": {"postgres", "label-studio"},
    }

    for compose_path, services in expected.items():
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        assert set(data["services"]) == services


def test_env_examples_exist() -> None:
    for path in (
        REPO_ROOT / "ops" / "env" / "local.env.example",
        REPO_ROOT / "projects" / "ml_backend" / "deploy" / "env" / "iats.env.example",
        REPO_ROOT / "projects" / "labelstudio" / "deploy" / "env" / "truenas.env.example",
    ):
        assert path.is_file(), path


def test_ops_shell_scripts_parse() -> None:
    shell_scripts = sorted((REPO_ROOT / "ops").glob("*.sh"))
    for script in shell_scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_workspace_entrypoints_compile() -> None:
    for path in (
        REPO_ROOT / "ops" / "src" / "birdsys" / "cli" / "main.py",
        REPO_ROOT / "projects" / "labelstudio" / "src" / "birdsys" / "labelstudio" / "cli.py",
        REPO_ROOT / "projects" / "datasets" / "src" / "birdsys" / "datasets" / "cli.py",
        REPO_ROOT / "projects" / "ml_backend" / "src" / "birdsys" / "ml_backend" / "cli.py",
        REPO_ROOT / "projects" / "ml_experiments" / "src" / "birdsys" / "ml_experiments" / "cli.py",
    ):
        py_compile.compile(str(path), doraise=True)


def test_makefile_has_core_remote_targets() -> None:
    content = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "iats-evaluate-model-b:" in content
    assert "truenas-refresh-lines-predictions:" in content
