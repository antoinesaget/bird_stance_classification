from __future__ import annotations

from birdsys.training.attributes import compute_head_masks, encode_labels


def test_masks_for_unreadable() -> None:
    masks = compute_head_masks(readability="unreadable", activity="standing")
    assert masks.readability is True
    assert masks.activity is False
    assert masks.support is False
    assert masks.legs is False
    assert masks.resting_back is False


def test_masks_for_readable_standing() -> None:
    masks = compute_head_masks(readability="readable", activity="standing")
    assert masks.readability is True
    assert masks.activity is True
    assert masks.support is True
    assert masks.legs is True
    assert masks.resting_back is True


def test_encode_labels() -> None:
    labels = encode_labels(
        readability="readable",
        activity="foraging",
        support="ground",
        resting_back="no",
        legs="two",
    )
    assert labels["readability"] == 1
    assert labels["activity"] == 1
    assert labels["support"] == 0
    assert labels["resting_back"] == 0
    assert labels["legs"] == 1
