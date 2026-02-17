from __future__ import annotations

from birdsys.training.attributes import BEHAVIOR_TO_ID, READABILITY_TO_ID, SPECIE_TO_ID, compute_head_masks, encode_labels


def test_masks_for_unreadable() -> None:
    masks = compute_head_masks(readability="unreadable", specie="correct", behavior="resting", substrate="ground")
    assert masks.readability is True
    assert masks.specie is True
    assert masks.behavior is False
    assert masks.substrate is False
    assert masks.legs is False


def test_masks_for_readable_resting_ground() -> None:
    masks = compute_head_masks(readability="readable", specie="correct", behavior="resting", substrate="ground")
    assert masks.readability is True
    assert masks.specie is True
    assert masks.behavior is True
    assert masks.substrate is True
    assert masks.legs is True


def test_masks_for_specie_incorrect() -> None:
    masks = compute_head_masks(readability="readable", specie="incorrect", behavior="resting", substrate="ground")
    assert masks.readability is True
    assert masks.specie is True
    assert masks.behavior is False
    assert masks.substrate is False
    assert masks.legs is False


def test_encode_labels() -> None:
    labels = encode_labels(
        readability="readable",
        specie="unsure",
        behavior="foraging",
        substrate="ground",
        legs="two",
    )
    assert labels["readability"] == READABILITY_TO_ID["readable"]
    assert labels["specie"] == SPECIE_TO_ID["unsure"]
    assert labels["behavior"] == BEHAVIOR_TO_ID["foraging"]
    assert labels["substrate"] == 0
    assert labels["legs"] == 1
