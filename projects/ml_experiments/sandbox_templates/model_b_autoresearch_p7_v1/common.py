"""Purpose: Provide shared constants and helper utilities for the autoresearch sandbox scripts"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARENT_REPO_ROOT = ROOT.parents[1]
SOURCE_DATASET_DIR = (
    PARENT_REPO_ROOT / "data" / "birds_project" / "derived" / "datasets" / "ds_project7_full_v001"
)
SOURCE_OLD_MODEL_EVAL_JSON = (
    PARENT_REPO_ROOT
    / "data"
    / "birds_project"
    / "models"
    / "attributes"
    / "experiments"
    / "project7_model_b_refresh_20260326"
    / "old_model_eval"
    / "report.json"
)
SOURCE_CV_REPORT_JSON = (
    PARENT_REPO_ROOT
    / "data"
    / "birds_project"
    / "models"
    / "attributes"
    / "experiments"
    / "project7_model_b_refresh_20260326"
    / "cv_report"
    / "cv_report.json"
)
SOURCE_FINAL_REPORT_JSON = (
    PARENT_REPO_ROOT
    / "data"
    / "birds_project"
    / "models"
    / "attributes"
    / "experiments"
    / "project7_model_b_refresh_20260326"
    / "final_report"
    / "report.json"
)
IATS_ENV_PATH = PARENT_REPO_ROOT / "projects" / "ml_backend" / "deploy" / "env" / "iats.env"
TRUENAS_ENV_PATH = PARENT_REPO_ROOT / "projects" / "labelstudio" / "deploy" / "env" / "truenas.env"
WORKSPACE_SRC_DIRS = (
    PARENT_REPO_ROOT / "ops" / "src",
    PARENT_REPO_ROOT / "shared" / "birdsys_core" / "src",
    PARENT_REPO_ROOT / "projects" / "labelstudio" / "src",
    PARENT_REPO_ROOT / "projects" / "datasets" / "src",
    PARENT_REPO_ROOT / "projects" / "ml_backend" / "src",
    PARENT_REPO_ROOT / "projects" / "ml_experiments" / "src",
)

RESULTS_HEADER = [
    "run_id",
    "search_score",
    "stance_f1",
    "behavior_f1",
    "substrate_f1",
    "readability_f1",
    "specie_f1",
    "stance_acc",
    "peak_vram_gb",
    "wall_seconds",
    "status",
    "keep_discard",
    "guardrail_status",
    "export_status",
    "model_family",
    "description",
]

HEADS = ["readability", "specie", "behavior", "substrate", "stance"]
CONFUSION_HEADS = ["behavior", "substrate", "stance"]

READABILITY_TO_ID = {"readable": 0, "occluded": 1, "unreadable": 2}
SPECIE_TO_ID = {"correct": 0, "incorrect": 1, "unsure": 2}
BEHAVIOR_TO_ID = {
    "flying": 0,
    "moving": 1,
    "foraging": 2,
    "resting": 3,
    "backresting": 4,
    "bathing": 5,
    "calling": 6,
    "preening": 7,
    "display": 8,
    "breeding": 9,
    "other": 10,
    "unsure": 11,
}
SUBSTRATE_TO_ID = {"bare_ground": 0, "vegetation": 1, "water": 2, "air": 3, "unsure": 4}
STANCE_TO_ID = {"unipedal": 0, "bipedal": 1, "sitting": 2, "unsure": 3}

LABEL_MAPS = {
    "readability": READABILITY_TO_ID,
    "specie": SPECIE_TO_ID,
    "behavior": BEHAVIOR_TO_ID,
    "substrate": SUBSTRATE_TO_ID,
    "stance": STANCE_TO_ID,
}
ID_TO_LABELS = {
    head: [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
    for head, mapping in LABEL_MAPS.items()
}

LEGACY_BEHAVIOR_ALIASES = {
    "flying": "flying",
    "moving": "moving",
    "foraging": "foraging",
    "resting": "resting",
    "backresting": "backresting",
    "bathing": "bathing",
    "calling": "calling",
    "preening": "preening",
    "display": "display",
    "breeding": "breeding",
    "other": "other",
    "unsure": "unsure",
}
LEGACY_SUBSTRATE_ALIASES = {
    "ground": "bare_ground",
    "bare ground": "bare_ground",
    "bare_ground": "bare_ground",
    "vegetation": "vegetation",
    "water": "water",
    "air": "air",
    "unsure": "unsure",
}
LEGACY_STANCE_ALIASES = {
    "one": "unipedal",
    "unipedal": "unipedal",
    "two": "bipedal",
    "bipedal": "bipedal",
    "sitting": "sitting",
    "unsure": "unsure",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model_family: str
    description: str
    supports_export_now: bool
    seed_policy: str


@dataclass(frozen=True)
class FoldContext:
    train_parquet: str
    test_parquet: str
    run_dir: str
    device: str
    budget_seconds: float
    seed: int
    fold_id: str


@dataclass(frozen=True)
class FoldOutput:
    metrics: dict[str, float]
    train_seconds: float
    peak_vram_mb: float
    candidate_artifact_dir: str | None
    export_hint: dict[str, Any]
    notes: str
    best_step: int | None = None
    best_val_search_score: float | None = None
    best_val_loss: float | None = None
    final_val_loss: float | None = None
    inner_val_rows: int = 0


@dataclass(frozen=True)
class HeadMasks:
    readability: bool
    specie: bool
    behavior: bool
    substrate: bool
    stance: bool


def normalize_choice(value: str | None) -> str | None:
    if value is None:
        return None
    value_n = value.strip().lower().replace(" ", "_")
    return value_n or None


def normalize_behavior(value: str | None) -> str | None:
    value_n = normalize_choice(value)
    if value_n is None:
        return None
    return LEGACY_BEHAVIOR_ALIASES.get(value_n, value_n)


def normalize_substrate(value: str | None) -> str | None:
    value_n = normalize_choice(value)
    if value_n is None:
        return None
    return LEGACY_SUBSTRATE_ALIASES.get(value_n, value_n)


def normalize_stance(value: str | None) -> str | None:
    value_n = normalize_choice(value)
    if value_n is None:
        return None
    return LEGACY_STANCE_ALIASES.get(value_n, value_n)


def compute_head_masks(
    isbird: str | None,
    readability: str | None,
    specie: str | None,
    behavior: str | None,
    substrate: str | None,
) -> HeadMasks:
    isbird_n = normalize_choice(isbird)
    readability_n = normalize_choice(readability)
    specie_n = normalize_choice(specie)
    behavior_n = normalize_behavior(behavior)
    substrate_n = normalize_substrate(substrate)
    is_bird = isbird_n == "yes"
    usable = is_bird and readability_n in {"readable", "occluded"} and specie_n != "incorrect"
    stance_relevant = usable and behavior_n in {"resting", "backresting"} and substrate_n in {
        "bare_ground",
        "water",
        "unsure",
    }
    return HeadMasks(
        readability=is_bird,
        specie=is_bird,
        behavior=usable,
        substrate=usable,
        stance=stance_relevant,
    )


def sanitize_positive_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["crop_path"] = out["crop_path"].astype(str)
    out = out[out["crop_path"].map(lambda value: Path(value).exists())].reset_index(drop=True)
    out["isbird"] = out["isbird"].map(lambda value: normalize_choice(None if pd.isna(value) else str(value)))
    out = out[out["isbird"] == "yes"].reset_index(drop=True)
    out["readability"] = out["readability"].map(lambda value: normalize_choice(None if pd.isna(value) else str(value)))
    out["specie"] = out["specie"].map(lambda value: normalize_choice(None if pd.isna(value) else str(value)))
    out["behavior"] = out["behavior"].map(lambda value: normalize_behavior(None if pd.isna(value) else str(value)))
    out["substrate"] = out["substrate"].map(lambda value: normalize_substrate(None if pd.isna(value) else str(value)))
    out["stance"] = out["stance"].map(lambda value: normalize_stance(None if pd.isna(value) else str(value)))
    return out.reset_index(drop=True)


def stable_fold_id(group_id: str, folds: int = 5) -> int:
    digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def stable_hash_fraction(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_value, pred_value in zip(y_true, y_pred):
        if 0 <= true_value < num_classes and 0 <= pred_value < num_classes:
            matrix[int(true_value), int(pred_value)] += 1
    return matrix


def class_supports(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    supports = np.zeros((num_classes,), dtype=np.int64)
    for value in y_true:
        if 0 <= value < num_classes:
            supports[int(value)] += 1
    return supports


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, *, ignore_absent_classes: bool = True) -> float:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    supports = class_supports(y_true, num_classes)
    f1s: list[float] = []
    for idx in range(num_classes):
        if ignore_absent_classes and supports[idx] <= 0:
            continue
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        if tp <= 0.0 and (fp <= 0.0 or fn <= 0.0):
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1s.append(float(2.0 * precision * recall / (precision + recall + 1e-12)))
    return float(np.mean(f1s)) if f1s else 0.0


def collect_visible_label_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts = {head: {label: 0 for label in ID_TO_LABELS[head]} for head in HEADS}
    for _, row in frame.iterrows():
        masks = compute_head_masks(
            isbird=str(row["isbird"]),
            readability=str(row["readability"]),
            specie=str(row["specie"]),
            behavior=str(row["behavior"]) if pd.notna(row["behavior"]) else None,
            substrate=str(row["substrate"]) if pd.notna(row["substrate"]) else None,
        )
        labels = {
            "readability": normalize_choice(str(row["readability"])) if pd.notna(row["readability"]) else None,
            "specie": normalize_choice(str(row["specie"])) if pd.notna(row["specie"]) else None,
            "behavior": normalize_behavior(str(row["behavior"])) if pd.notna(row["behavior"]) else None,
            "substrate": normalize_substrate(str(row["substrate"])) if pd.notna(row["substrate"]) else None,
            "stance": normalize_stance(str(row["stance"])) if pd.notna(row["stance"]) else None,
        }
        mask_map = {
            "readability": masks.readability,
            "specie": masks.specie,
            "behavior": masks.behavior,
            "substrate": masks.substrate,
            "stance": masks.stance,
        }
        for head in HEADS:
            label = labels[head]
            if mask_map[head] and label in counts[head]:
                counts[head][label] += 1
    return counts


def summarize_dataset_labels(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for column in ["isbird", "readability", "specie", "behavior", "substrate", "stance"]:
        counts = frame[column].fillna("<null>").value_counts().sort_index().to_dict()
        out[column] = {str(key): int(value) for key, value in counts.items()}
    return out


def fallback_label(head: str) -> str:
    if "unsure" in ID_TO_LABELS[head]:
        return "unsure"
    return ID_TO_LABELS[head][0]


def normalize_head_label(head: str, value: str | None) -> str | None:
    if head == "readability":
        return normalize_choice(value)
    if head == "specie":
        return normalize_choice(value)
    if head == "behavior":
        return normalize_behavior(value)
    if head == "substrate":
        return normalize_substrate(value)
    if head == "stance":
        return normalize_stance(value)
    raise KeyError(head)


def coerce_supported_label(
    head: str,
    label: str | None,
    supported_labels: dict[str, set[str]],
    diagnostics: dict[str, dict[str, int]],
) -> str:
    normalized = normalize_head_label(head, label) or fallback_label(head)
    allowed = supported_labels.get(head) or set(ID_TO_LABELS[head])
    if normalized in allowed:
        return normalized
    fallback = fallback_label(head)
    key = f"{normalized}->{fallback}"
    diagnostics[head][key] = diagnostics[head].get(key, 0) + 1
    return fallback


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def extend_workspace_sys_path() -> None:
    for path in reversed(WORKSPACE_SRC_DIRS):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def decode_jwt_payload(token: str) -> dict[str, Any] | None:
    if token.count(".") != 2:
        return None
    _, payload_b64, _ = token.split(".", 2)
    padding = "=" * (-len(payload_b64) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8"))
    except Exception:
        return None


def auth_header(api_token: str) -> str:
    return f"Bearer {api_token}" if api_token.count(".") == 2 else f"Token {api_token}"


def request_json(
    base_url: str,
    path: str,
    api_token: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | list[Any] | None = None,
    authorization: str | None = None,
) -> dict[str, Any] | list[Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if authorization is None:
        authorization = auth_header(api_token) if api_token else ""
    if authorization:
        headers["Authorization"] = authorization
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        data=body,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def request_empty(base_url: str, path: str, api_token: str, *, method: str = "DELETE") -> None:
    headers = {"Accept": "application/json", "Authorization": auth_header(api_token)}
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        data=b"{}",
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=300):
        return None


def resolve_api_token(base_url: str, api_token: str) -> str:
    payload = decode_jwt_payload(api_token)
    if not payload or payload.get("token_type") != "refresh":
        return api_token
    response = request_json(
        base_url,
        "/api/token/refresh/",
        api_token="",
        method="POST",
        payload={"refresh": api_token},
        authorization="",
    )
    access_token = response.get("access") if isinstance(response, dict) else None
    if not access_token:
        raise RuntimeError("Label Studio token refresh did not return an access token")
    return str(access_token)
