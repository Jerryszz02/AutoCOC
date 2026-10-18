"""Visible-target grouping for the explicitly enabled multi-core preview path."""

import math

import pytest

from autococ.core_geometry import infer_multi_core_geometry, validate_multi_core_profile
from autococ.model_vision import BuildingObservation


def profile(**overrides):
    settings = {"enabled": True, "link_distance_pixels": 60,
                "separation_margin_pixels": 20, "min_targets_per_core": 1,
                "max_cores": 3, "max_selected_detections": 12, "max_candidates": 12}
    settings.update(overrides)
    return {"core_classes": {"town_hall": .9}, "multi_core": settings}


def target(x, y=250, *, confidence=.99, frame=100., model="modelhash", kind="town_hall"):
    return BuildingObservation(kind, (x - 10., y - 10., x + 10., y + 10.),
                               confidence, frame, model)


def infer(items, accepted=None):
    return infer_multi_core_geometry(items, accepted or profile(), captured_at=100.,
                                     model_sha256="modelhash", source_resolution=(1280, 720))


def test_two_three_groups_are_order_stable_and_keep_visible_target_evidence():
    two = [target(100), target(130), target(700), target(730)]
    expected = infer(two)
    assert len(expected) == 2
    assert expected == infer(list(reversed(two)))
    assert [len(group["detections"]) for group in expected] == [2, 2]
    assert expected[0]["center"][0] == 115
    assert expected[1]["center"][0] == 715
    assert all(row["frame_captured_at"] == 100. and row["model_sha256"] == "modelhash"
               for group in expected for row in group["detections"])
    three = infer(two + [target(1100)])
    assert len(three) == 3
    assert [group["center"][0] for group in three] == [115, 715, 1100]


@pytest.mark.parametrize("items", [
    [target(100), target(140), target(180), target(700)],  # bridge chain, diameter > link
    [target(100), target(170)],  # no edge, but between-group gray zone
    [target(100), target(160)],  # exactly the link boundary
    [target(100), target(180)],  # exactly the separation boundary
    [target(100), target(700), target(1100), target(1240)],  # four clear groups
])
def test_ambiguous_or_excessive_groups_are_rejected(items):
    with pytest.raises(ValueError):
        infer(items)


@pytest.mark.parametrize("bad", [
    target(700, frame=99.),
    target(700, model="other"),
    target(700, confidence=math.nan),
    target(700, confidence=.1, kind="unselected", frame=99.),
    BuildingObservation("town_hall", (-1., 240., 20., 260.), .99, 100., "modelhash"),
    BuildingObservation("town_hall", (690., 240., math.nan, 260.), .99, 100., "modelhash"),
    BuildingObservation("town_hall", None, .99, 100., "modelhash"),
])
def test_every_detection_must_share_current_frame_model_and_valid_pixels(bad):
    with pytest.raises(ValueError):
        infer([target(100), bad])


def test_helper_rejects_unbounded_output_and_boolean_capture_identity():
    with pytest.raises(ValueError, match="output bound"):
        infer([target(100)] * 301)
    observations = [BuildingObservation("town_hall", (90., 240., 110., 260.), .99,
                                        True, "modelhash"),
                    BuildingObservation("town_hall", (690., 240., 710., 260.), .99,
                                        1., "modelhash")]
    with pytest.raises(ValueError, match="current frame/model"):
        infer_multi_core_geometry(observations, profile(), captured_at=1.,
                                  model_sha256="modelhash", source_resolution=(1280, 720))


@pytest.mark.parametrize("settings", [
    None, {"enabled": 1}, {"enabled": False, "max_cores": 3},
    {"enabled": True},
    {"enabled": True, "link_distance_pixels": float("nan")},
    {"enabled": True, "max_cores": 4},
    {"enabled": True, "max_candidates": 0},
    {"enabled": True, "max_selected_detections": 25},
    {"enabled": True, "min_targets_per_core": 2, "max_selected_detections": 3},
])
def test_invalid_or_incomplete_profile_parameters_reject(settings):
    if isinstance(settings, dict) and settings.get("enabled") is True and 1 < len(settings) < 7:
        settings = {**profile()["multi_core"], **settings}
    with pytest.raises(ValueError):
        validate_multi_core_profile({"core_classes": {"town_hall": .9}, "multi_core": settings})


def test_disabled_profile_preserves_single_core_contract_and_limits():
    assert validate_multi_core_profile({"multi_core": {"enabled": False}}) is None
    assert validate_multi_core_profile({}) is None
    with pytest.raises(ValueError, match="target count"):
        infer([target(100)], profile())
    with pytest.raises(ValueError, match="target count"):
        infer([target(100), target(700), target(1100)], profile(max_selected_detections=2))
    with pytest.raises(ValueError, match="too few"):
        infer([target(100), target(700)], profile(min_targets_per_core=2))
