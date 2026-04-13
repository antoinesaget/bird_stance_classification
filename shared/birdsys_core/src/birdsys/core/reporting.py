from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_version_index(name: str, prefix: str) -> int | None:
    token = f"{prefix}_v"
    if not name.startswith(token):
        return None
    suffix = name[len(token) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def find_previous_version_dir(parent: Path, prefix: str, current_name: str) -> Path | None:
    current_idx = parse_version_index(current_name, prefix)
    if current_idx is None:
        return None

    candidates: list[tuple[int, Path]] = []
    for path in parent.iterdir():
        if not path.is_dir():
            continue
        idx = parse_version_index(path.name, prefix)
        if idx is None or idx >= current_idx:
            continue
        candidates.append((idx, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def diff_numeric_dict(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, float]:
    keys = set(current) | set(previous)
    out: dict[str, float] = {}
    for key in sorted(keys):
        cur = current.get(key)
        prev = previous.get(key)
        if isinstance(cur, (int, float)) and isinstance(prev, (int, float)):
            out[key] = float(cur) - float(prev)
    return out
