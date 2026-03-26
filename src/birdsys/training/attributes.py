from __future__ import annotations

from dataclasses import dataclass


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

LEGACY_BEHAVIOR_ALIASES = {
    "flying": "flying",
    "foraging": "foraging",
    "resting": "resting",
    "backresting": "backresting",
    "preening": "preening",
    "display": "display",
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
class HeadMasks:
    readability: bool
    specie: bool
    behavior: bool
    substrate: bool
    stance: bool


def normalize_choice(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower().replace(" ", "_")
    if not v:
        return None
    return v


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


def encode_labels(
    readability: str | None,
    specie: str | None,
    behavior: str | None,
    substrate: str | None,
    stance: str | None,
) -> dict[str, int | None]:
    readability_n = normalize_choice(readability)
    specie_n = normalize_choice(specie)
    behavior_n = normalize_behavior(behavior)
    substrate_n = normalize_substrate(substrate)
    stance_n = normalize_stance(stance)

    return {
        "readability": READABILITY_TO_ID.get(readability_n),
        "specie": SPECIE_TO_ID.get(specie_n),
        "behavior": BEHAVIOR_TO_ID.get(behavior_n),
        "substrate": SUBSTRATE_TO_ID.get(substrate_n),
        "stance": STANCE_TO_ID.get(stance_n),
    }
