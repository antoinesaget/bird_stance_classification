"""Purpose: Load the shared environment and path configuration used across the workspace"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppEnv:
    birds_data_root: Path
    label_studio_url: str
    label_studio_api_token: str
    model_a_weights: Path
    model_b_checkpoint: Path | None
    log_level: str


@dataclass(frozen=True)
class ProjectFileConfig:
    data_root: Path | None = None
    model_a_weights: Path | None = None
    model_b_checkpoint: Path | None = None


def _safe_expand_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    try:
        return path.resolve()
    except OSError:
        if path.is_absolute():
            return path
        return (Path.cwd() / path).absolute()


def _get_env_path(name: str, default: str | None = None) -> Path | None:
    value = os.getenv(name, default)
    if value is None or value.strip() == "":
        return None
    return _safe_expand_path(value)


def _load_project_file_config(config_path: Path | None) -> ProjectFileConfig:
    if config_path is None or not config_path.exists():
        return ProjectFileConfig()

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths = raw.get("paths", {}) if isinstance(raw, dict) else {}
    models = raw.get("models", {}) if isinstance(raw, dict) else {}

    def _as_path(value: Any) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return _safe_expand_path(text)

    return ProjectFileConfig(
        data_root=_as_path(paths.get("data_root")),
        model_a_weights=_as_path(models.get("model_a_weights")),
        model_b_checkpoint=_as_path(models.get("model_b_checkpoint")),
    )


def load_app_env(
    *,
    dotenv_path: Path | None = None,
    config_path: Path | None = None,
) -> AppEnv:
    if dotenv_path is not None:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()

    file_cfg = _load_project_file_config(config_path)

    birds_data_root = _get_env_path("BIRDS_DATA_ROOT") or file_cfg.data_root or Path("/data/birds_project")
    model_a_weights = _get_env_path("MODEL_A_WEIGHTS") or file_cfg.model_a_weights or Path("yolo11m.pt").resolve()
    model_b_checkpoint = _get_env_path("MODEL_B_CHECKPOINT") or file_cfg.model_b_checkpoint

    return AppEnv(
        birds_data_root=birds_data_root,
        label_studio_url=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"),
        label_studio_api_token=os.getenv("LABEL_STUDIO_API_TOKEN", ""),
        model_a_weights=model_a_weights,
        model_b_checkpoint=model_b_checkpoint,
        log_level=os.getenv("BIRDS_LOG_LEVEL", "INFO"),
    )
