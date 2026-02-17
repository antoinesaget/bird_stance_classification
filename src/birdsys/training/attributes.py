from __future__ import annotations

from dataclasses import dataclass


READABILITY_TO_ID = {"readable": 1, "unreadable": 0}
ACTIVITY_TO_ID = {"flying": 0, "foraging": 1, "standing": 2}
SUPPORT_TO_ID = {"ground": 0, "water": 1, "air": 2}
RESTING_BACK_TO_ID = {"no": 0, "yes": 1}
LEGS_TO_ID = {"one": 0, "two": 1, "unsure": 2}


@dataclass(frozen=True)
class HeadMasks:
    readability: bool
    activity: bool
    support: bool
    resting_back: bool
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
    activity: str | None,
) -> HeadMasks:
    readability_n = normalize_choice(readability)
    activity_n = normalize_choice(activity)
    is_readable = readability_n == "readable"
    is_standing = is_readable and activity_n == "standing"
    return HeadMasks(
        readability=True,
        activity=is_readable,
        support=is_readable,
        resting_back=is_standing,
        legs=is_standing,
    )


def encode_labels(
    readability: str | None,
    activity: str | None,
    support: str | None,
    resting_back: str | None,
    legs: str | None,
) -> dict[str, int | None]:
    readability_n = normalize_choice(readability)
    activity_n = normalize_choice(activity)
    support_n = normalize_choice(support)
    resting_back_n = normalize_choice(resting_back)
    legs_n = normalize_choice(legs)

    return {
        "readability": READABILITY_TO_ID.get(readability_n),
        "activity": ACTIVITY_TO_ID.get(activity_n),
        "support": SUPPORT_TO_ID.get(support_n),
        "resting_back": RESTING_BACK_TO_ID.get(resting_back_n),
        "legs": LEGS_TO_ID.get(legs_n),
    }
