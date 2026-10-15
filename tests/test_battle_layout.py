from dataclasses import replace

import pytest

from autococ.battle_layout import describe_visible_layout
from autococ.model_vision import BuildingObservation


def describe(items, **changes):
    args = dict(classes=["town_hall", "air_defense"], frame_id="current.png", captured_at=100.,
                model_sha256="model", source_resolution=(2560, 1440),
                baseline_resolution=(1280, 720), entries={0: (100., 100.), 1: (1000., 600.)})
    args.update(changes)
    return describe_visible_layout(items, **args)


TH = BuildingObservation("town_hall", (900., 400., 1100., 600.), .98, 100., "model")
AD = BuildingObservation("air_defense", (1800., 1000., 2000., 1200.), .96, 100., "model")


def test_source_scaled_layout_distinguishes_entries_without_inventing_absence():
    result = describe([TH, AD])
    assert result["buildings"][1]["center"] == [500., 250.]
    assert result["observed_counts"] == {"air_defense": 1, "town_hall": 1}
    near = result["entries"]["1"]["nearest_observed_by_class"]["air_defense"]
    far = result["entries"]["0"]["nearest_observed_by_class"]["air_defense"]
    assert near["distance_pixels"] < far["distance_pixels"]
    assert result["missing_detection_means_absent"] is False
    assert "air_sweeper_direction" in result["unknown"]
    assert "hidden_traps" in result["unknown"]
    assert result["frame_id"] == "current.png" and result["model_sha256"] == "model"


def test_missing_supported_class_is_unknown_and_subset_model_has_no_extra_classes():
    result = describe([TH])
    assert result["entries"]["0"]["nearest_observed_by_class"]["air_defense"] is None
    assert "air_defense" not in result["observed_counts"]
    result = describe([TH], classes=["town_hall"])
    assert set(result["entries"]["0"]["nearest_observed_by_class"]) == {"town_hall"}
    assert describe([])["buildings"] == []


@pytest.mark.parametrize("changes", [
    {"frame_captured_at": 99.}, {"frame_captured_at": float("nan")},
    {"model_sha256": "another"}, {"kind": "invented"},
    {"confidence": float("nan")}, {"confidence": True}, {"confidence": 1.1},
    {"bbox": (-1., 400., 1100., 600.)}, {"bbox": (1100., 400., 900., 600.)},
    {"bbox": (900., 400., 2600., 600.)}, {"bbox": (900., 400., float("inf"), 600.)},
])
def test_wrong_frame_model_or_invalid_detection_is_rejected(changes):
    with pytest.raises(ValueError):
        describe([replace(TH, **changes)])


@pytest.mark.parametrize("changes", [
    {"classes": []}, {"classes": ["town_hall", "town_hall"]},
    {"entries": {0: (1300., 300.)}}, {"source_resolution": (0, 720)},
    {"captured_at": float("nan")},
])
def test_invalid_source_contract_is_rejected(changes):
    with pytest.raises(ValueError):
        describe([TH], **changes)
