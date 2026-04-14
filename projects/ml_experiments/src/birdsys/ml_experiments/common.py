"""Purpose: Share Model B label constants and dataset summaries across training and evaluation."""
from __future__ import annotations

from typing import Any

import pandas as pd

from birdsys.core.attributes import (
    BEHAVIOR_TO_ID,
    READABILITY_TO_ID,
    SPECIE_TO_ID,
    STANCE_TO_ID,
    SUBSTRATE_TO_ID,
    compute_head_masks,
    normalize_behavior,
    normalize_choice,
    normalize_stance,
    normalize_substrate,
)

HEADS = ["readability", "specie", "behavior", "substrate", "stance"]
CONFUSION_HEADS = ["behavior", "substrate", "stance"]
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
HEAD_CLASS_COUNTS = {head: len(labels) for head, labels in ID_TO_LABELS.items()}
DATASET_LABEL_COLUMNS = ["isbird", "readability", "specie", "behavior", "substrate", "stance"]


def summarize_dataset_labels(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for column in DATASET_LABEL_COLUMNS:
        if column not in frame.columns:
            out[column] = {}
            continue
        counts = frame[column].fillna("<null>").value_counts().sort_index().to_dict()
        out[column] = {str(key): int(value) for key, value in counts.items()}
    return out


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
        normalized = {
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
            label = normalized[head]
            if mask_map[head] and label in counts[head]:
                counts[head][label] += 1
    return counts


def flatten_dict(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in payload.items()}
