from __future__ import annotations

from dataclasses import dataclass


READABILITY_TO_ID = {"readable": 0, "occluded": 1, "unreadable": 2}
SPECIE_TO_ID = {"correct": 0, "incorrect": 1, "unsure": 2}
BEHAVIOR_TO_ID = {
    "flying": 0,
    "foraging": 1,
    "resting": 2,
    "backresting": 3,
    "preening": 4,
    "display": 5,
    "unsure": 6,
}
SUBSTRATE_TO_ID = {"ground": 0, "water": 1, "air": 2, "unsure": 3}
LEGS_TO_ID = {"one": 0, "two": 1, "unsure": 2, "sitting": 3}


@dataclass(frozen=True)
class HeadMasks:
    readability: bool
    specie: bool
    behavior: bool
    substrate: bool
    legs: bool


def normalize_choice(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if not v:
        return None
    return v


def compute_head_masks(
    readability: str | None,
    specie: str | None,
    behavior: str | None,
    substrate: str | None,
) -> HeadMasks:
    readability_n = normalize_choice(readability)
    specie_n = normalize_choice(specie)
    behavior_n = normalize_choice(behavior)
    substrate_n = normalize_choice(substrate)
    usable = readability_n in {"readable", "occluded"} and specie_n != "incorrect"
    legs_relevant = usable and behavior_n in {"resting", "backresting"} and substrate_n in {"ground", "water"}
    return HeadMasks(
        readability=True,
        specie=True,
        behavior=usable,
        substrate=usable,
        legs=legs_relevant,
    )


def encode_labels(
    readability: str | None,
    specie: str | None,
    behavior: str | None,
    substrate: str | None,
    legs: str | None,
) -> dict[str, int | None]:
    readability_n = normalize_choice(readability)
    specie_n = normalize_choice(specie)
    behavior_n = normalize_choice(behavior)
    substrate_n = normalize_choice(substrate)
    legs_n = normalize_choice(legs)

    return {
        "readability": READABILITY_TO_ID.get(readability_n),
        "specie": SPECIE_TO_ID.get(specie_n),
        "behavior": BEHAVIOR_TO_ID.get(behavior_n),
        "substrate": SUBSTRATE_TO_ID.get(substrate_n),
        "legs": LEGS_TO_ID.get(legs_n),
    }
