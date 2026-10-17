"""Anonymized current-client Warden skill transition, never training truth."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from autococ.hero_state import recognize_hero_state
from autococ import hero_state as hero_module
from autococ.post_deployment_skill import _current_warden
from autococ.scene import SceneSnapshot


FIXTURES = Path(__file__).parent / "fixtures/warden_yellow_used_061704_18_647_2"
BEFORE = FIXTURES / "00032-post-deployment-skill-before.png"
AFTER = FIXTURES / "00033-post-deployment-skill-after-1.png"
EDGE = FIXTURES / "00034-post-deployment-skill-after-2.png"
BOX = [647, 595, 736, 711]


def _target(path=AFTER, *, version="18.647.2"):
    return _current_warden(SimpleNamespace(client_version=version),
                           SimpleNamespace(screenshot_path=path))


def _image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape == (1440, 2560, 3)
    return image


def _write(path, image):
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(path)


def _mutated(image, box):
    result = image.copy()
    x0, y0, x1, y1 = box
    result[y0 * 2:y1 * 2, x0 * 2:x1 * 2] = 0
    return result


def test_anonymization_and_provenance():
    meta = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    assert meta["source_role"] == "TRAIN_candidate_attempted_unverified"
    assert meta["no_identity_or_training_truth"] is True
    for row in meta["rows"]:
        path = FIXTURES / row["fixture"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["fixture_sha256"]
        image = _image(path)
        x0, y0, x1, y1 = 1260, 1020, 1690, 1440
        assert not np.any(image[:y0])
        assert not np.any(image[:, :x0])
        assert not np.any(image[:, x1:])
        assert hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == row["retained_pixels_sha256"]


def test_anonymous_transition_requires_current_used_equipment():
    # Original-frame replay is recorded separately; this portable test uses anonymous fixtures.
    assert _target(BEFORE)["verified"] is True
    assert _target(BEFORE)["state"]["ability_ready"] is True
    used = _target(AFTER)
    assert used["verified"] is True
    assert used["state"]["ability_used"] is True
    assert used["state"]["ability_ready"] is False
    evidence = used["state"]["evidence"]
    assert evidence["yellow_health_bar"]["frame_support"] == [1., 1., 1., 1.]
    assert evidence["current_equipment_matches"]["used_book"] is not None
    assert evidence["current_equipment_matches"]["used_card"] is not None
    assert evidence["current_equipment_matches"]["ready_book"] is None
    assert evidence["current_equipment_matches"]["ready_card"] is None


def test_later_frame_meets_current_topology_without_rewriting_historical_failure():
    # The original run predates the gray-card topology fallback and remains
    # attempted_unverified. This asserts today's pixel contract, not skill truth.
    meta = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    assert meta["source_role"] == "TRAIN_candidate_attempted_unverified"
    assert meta["no_identity_or_training_truth"] is True
    target = _target(EDGE)
    assert target["verified"] is True
    assert target["locator"]["bbox"] == BOX
    topology = target["locator"]["gray_used_topology"]
    assert topology["paired_rows"] == 70
    assert topology["longest_paired_run"] == 70
    assert topology["dark_line"]["paired_dark_trough_rows"] >= 45
    state = target["state"]
    assert state["ability_used"] is True
    assert state["ability_ready"] is False
    assert state["evidence"]["yellow_health_bar"]["frame_support"] == [1., 1., 1., 1.]
    anchors = state["evidence"]["current_equipment_matches"]
    assert anchors["used_book"] is not None and anchors["used_card"] is not None
    assert anchors["ready_book"] is None and anchors["ready_card"] is None


def test_yellow_ready_cannot_authorize_input():
    ready = _image(BEFORE)
    yellow = _image(AFTER)
    # Only the native HP strip changes. The ready book, card and portrait remain.
    ready[1138:1172, 1302:1472] = yellow[1138:1172, 1302:1472]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "yellow-ready.png"
        _write(path, ready)
        target = _target(path)
        assert target["state"]["evidence"]["yellow_health_bar"] is not None
        assert target["state"]["ability_ready"] is not True
        assert target["verified"] is False


def test_missing_hp_closed_card_equipment_and_neighbor_portrait_reject():
    source = _image(AFTER)
    cases = {
        "no_hp": _mutated(source, (653, 570, 732, 586)),
        "broken_left_rim": _mutated(source, (644, 618, 651, 700)),
        "broken_right_rim": _mutated(source, (728, 618, 742, 706)),
        "missing_book": _mutated(source, (648, 525, 697, 572)),
        "missing_card": _mutated(source, (675, 621, 735, 687)),
    }
    neighbor = source.copy()
    # Move the adjacent hero's portrait into this exact closed Warden slot.
    neighbor[1230:1360, 1360:1460] = source[1230:1360, 1540:1640]
    cases["same_slot_neighbor_portrait"] = neighbor
    with TemporaryDirectory() as directory:
        for name, image in cases.items():
            path = Path(directory) / f"{name}.png"
            _write(path, image)
            target = _target(path)
            assert target["verified"] is False, name
            assert (target["state"] or {}).get("ability_used") is not True, name


def test_conflicting_equipment_wrong_hero_and_wrong_version_reject():
    source = _image(AFTER)
    ready = _image(BEFORE)
    # Keep the used card, but insert the ready book in its own search area.
    conflict = source.copy()
    conflict[1050:1144, 1296:1392] = ready[1050:1144, 1296:1392]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "conflict.png"
        _write(path, conflict)
        state = recognize_hero_state(path, BOX, unit_id="grand_warden", client_version="18.647.2")
        assert state["ability_used"] is not True
        assert _target(path)["verified"] is False
    wrong_hero = recognize_hero_state(AFTER, BOX, unit_id="archer_queen", client_version="18.647.2")
    assert wrong_hero["ability_used"] is not True
    assert _target(AFTER, version="18.600.7")["verified"] is False


def test_each_used_equipment_anchor_is_independently_required():
    original_match = hero_module._match
    for missing, present in (("used_book", "used_card"), ("used_card", "used_book")):
        template_name = f"hero_warden_18_647_2_{missing}.png"

        def match_except_one(image, template, *args, **kwargs):
            if Path(template).name == template_name:
                return None
            return original_match(image, template, *args, **kwargs)

        with patch.object(hero_module, "_match", side_effect=match_except_one):
            target = _target(AFTER)
        state = target["state"]
        evidence = state["evidence"]
        anchors = evidence["current_equipment_matches"]
        assert target["locator"]["bbox"] == BOX
        assert evidence["portrait_identity"]["unit_id"] == "grand_warden"
        assert evidence["portrait_identity"]["version_verified"] is True
        assert evidence["closed_card"]["bbox"] == BOX
        assert evidence["yellow_health_bar"]["frame_support"] == [1., 1., 1., 1.]
        assert anchors[missing] is None
        assert anchors[present] is not None
        assert state["deployed"] is True
        assert state["ability_used"] is not True
        assert target["verified"] is False


def test_old_or_repeated_frame_cannot_authorize_used_proof():
    assert _target(BEFORE)["state"]["ability_used"] is False
    from autococ.post_deployment_skill import _frame
    from autococ.errors import FlowError
    import pytest

    prior = SceneSnapshot("battle", .99, BEFORE, {
        "source_resolution": [2560, 1440], "baseline_resolution": [1280, 720],
        "capture_timestamp_basis": "capture_request_start",
        "requested_at_monotonic": 10., "observed_at_monotonic": 10.})
    repeated = SceneSnapshot("battle", .99, BEFORE, dict(prior.observations))
    session = SimpleNamespace(last_snapshot=repeated, client_version="18.647.2",
                              config=SimpleNamespace(game=SimpleNamespace(baseline_resolution=(1280, 720))))
    with pytest.raises(FlowError, match="capture order"):
        _frame(session, repeated, previous=prior)
