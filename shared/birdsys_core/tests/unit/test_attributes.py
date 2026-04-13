from __future__ import annotations

from birdsys.core.attributes import (
    BEHAVIOR_TO_ID,
    READABILITY_TO_ID,
    SPECIE_TO_ID,
    STANCE_TO_ID,
    compute_head_masks,
    encode_labels,
)


def test_masks_for_unreadable() -> None:
    masks = compute_head_masks(isbird="yes", readability="unreadable", specie="correct", behavior="resting", substrate="bare_ground")
    assert masks.readability is True
    assert masks.specie is True
    assert masks.behavior is False
    assert masks.substrate is False
    assert masks.stance is False


def test_masks_for_readable_resting_ground() -> None:
    masks = compute_head_masks(isbird="yes", readability="readable", specie="correct", behavior="resting", substrate="bare_ground")
    assert masks.readability is True
    assert masks.specie is True
    assert masks.behavior is True
    assert masks.substrate is True
    assert masks.stance is True


def test_masks_for_specie_incorrect() -> None:
    masks = compute_head_masks(isbird="yes", readability="readable", specie="incorrect", behavior="resting", substrate="bare_ground")
    assert masks.readability is True
    assert masks.specie is True
    assert masks.behavior is False
    assert masks.substrate is False
    assert masks.stance is False


def test_masks_for_non_bird() -> None:
    masks = compute_head_masks(isbird="no", readability="readable", specie="correct", behavior="resting", substrate="bare_ground")
    assert masks.readability is False
    assert masks.specie is False
    assert masks.behavior is False
    assert masks.substrate is False
    assert masks.stance is False


def test_encode_labels() -> None:
    labels = encode_labels(
        readability="readable",
        specie="unsure",
        behavior="foraging",
        substrate="bare_ground",
        stance="bipedal",
    )
    assert labels["readability"] == READABILITY_TO_ID["readable"]
    assert labels["specie"] == SPECIE_TO_ID["unsure"]
    assert labels["behavior"] == BEHAVIOR_TO_ID["foraging"]
    assert labels["substrate"] == 0
    assert labels["stance"] == STANCE_TO_ID["bipedal"]
