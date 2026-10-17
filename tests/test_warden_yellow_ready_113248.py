"""Portable regression for the masked 113248 TRAIN Warden frames.

These offline pixel checks leave the original failed live receipt unchanged.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ import hero_state as hero_module
from autococ import unit_catalog
from autococ.dynamic_tracking import read_hero_deployment_evidence
from autococ.dynamic_vision import DynamicFrame
from autococ.hero_state import locate_current_deployed_hero_card, recognize_hero_state
from autococ.post_deployment_skill import _current_warden


FIXTURES = Path(__file__).parent / "fixtures/warden_yellow_ready_113248_18_647_2"
OLD_USED = Path(__file__).parent / "fixtures/warden_yellow_used_061704_18_647_2/00033-post-deployment-skill-after-1.png"
BOX = (647, 595, 736, 711)
NATIVE_BOX = tuple(coordinate * 2 for coordinate in BOX)
NAMES = ("00162-battle-settlement.png", "00165-battle-settlement.png",
         "00166-battle-settlement.png", "00167-battle-settlement.png",
         "00168-battle-settlement.png")


def target(path: Path, version: str = "18.647.2") -> dict:
    return _current_warden(SimpleNamespace(client_version=version),
                           SimpleNamespace(screenshot_path=path))


def image(path: Path):
    result = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    assert result is not None and result.shape == (1440, 2560, 3)
    return result


def write(path: Path, pixels) -> None:
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    encoded.tofile(path)


def source_frame(path: Path, *, version: str = "18.647.2") -> DynamicFrame:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return DynamicFrame("113248-train-offline", "one-camera-epoch", path.name,
                        1.0, version, 2560, 1440, True, True, digest)


def test_anonymization_binds_original_train_sha_and_exact_retained_pixels():
    meta = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    assert meta["source_role"] == "TRAIN_development_offline_pixel_prototype"
    assert meta["human_review"] is False and meta["formal_acceptance"] is False
    assert meta["retained_native_rect_xyxy"] == [1000, 1010, 1690, 1440]
    assert [row["fixture"] for row in meta["rows"]] == list(NAMES)
    for row in meta["rows"]:
        path = FIXTURES / row["fixture"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["fixture_sha256"]
        pixels = image(path)
        assert not np.any(pixels[:1010])
        assert not np.any(pixels[:, :1000])
        assert not np.any(pixels[:, 1690:])
        assert hashlib.sha256(pixels[1010:1440, 1000:1690].tobytes()).hexdigest() == row["retained_pixels_sha256"]


@pytest.mark.parametrize("name", NAMES)
def test_anonymous_train_yellow_ready_requires_all_current_frame_evidence(name):
    path = FIXTURES / name
    located = locate_current_deployed_hero_card(path, "grand_warden", client_version="18.647.2")
    assert located is not None and located["bbox"] == list(BOX)
    state = recognize_hero_state(path, BOX, unit_id="grand_warden", client_version="18.647.2")
    evidence = state["evidence"]
    anchors = evidence["current_equipment_matches"]
    assert state["ability_ready"] is True and state["ability_used"] is False
    assert evidence["yellow_health_bar"]["method"] == "yellow_fill_enclosed_by_dark_hp_frame"
    assert min(evidence["yellow_health_bar"]["frame_support"]) >= .85
    assert evidence["portrait_identity"]["unit_id"] == "grand_warden"
    assert evidence["portrait_identity"]["confidence"] >= .95
    assert evidence["closed_card"]["bbox"] == list(BOX)
    assert anchors["ready_book"] is not None and anchors["ready_card"] is not None
    assert anchors["used_book"] is None and anchors["used_card"] is None
    assert evidence["ability_ready"]["yellow_hp_ready_contract"]["closed_card"]["frame"] == str(path)
    assert target(path)["verified"] is True


def test_independent_ready_anchor_absence_and_mixed_used_anchor_reject(monkeypatch):
    path = FIXTURES / NAMES[0]
    original = hero_module._match
    for absent in ("ready_book", "ready_card", "mixed_used_book"):
        def changed(pixels, template_path, roi, *args, **kwargs):
            name = template_path.name
            if absent == "ready_book" and name == "hero_warden_18_647_2_ready_book.png":
                return None
            if absent == "ready_card" and name.startswith("hero_warden_18_647_2_ready_card"):
                return None
            if absent == "mixed_used_book" and name == "hero_warden_18_647_2_used_book.png":
                return {"template": str(template_path), "confidence": .99, "bbox": list(roi)}
            return original(pixels, template_path, roi, *args, **kwargs)
        with monkeypatch.context() as patcher:
            patcher.setattr(hero_module, "_match", changed)
            state = recognize_hero_state(path, BOX, unit_id="grand_warden",
                                         client_version="18.647.2")
        evidence = state["evidence"]
        assert evidence["yellow_health_bar"] is not None
        assert evidence["portrait_identity"]["unit_id"] == "grand_warden"
        assert evidence["closed_card"]["bbox"] == list(BOX)
        assert state["ability_ready"] is not True


def test_missing_hp_or_closed_card_edge_rejects_anonymous_positive():
    baseline = image(FIXTURES / NAMES[0])
    for name, box in (("missing_hp", (1302, 1138, 1472, 1172)),
                      ("broken_left_card_edge", (1288, 1250, 1302, 1400))):
        x0, y0, x1, y1 = box
        changed = baseline.copy()
        changed[y0:y1, x0:x1] = 0
        with TemporaryDirectory() as directory:
            path = Path(directory) / f"{name}.png"
            write(path, changed)
            assert target(path)["verified"] is False


def test_wrong_hero_and_wrong_version_never_authorize():
    path = FIXTURES / NAMES[0]
    wrong_hero = recognize_hero_state(path, BOX, unit_id="archer_queen",
                                      client_version="18.647.2")
    assert wrong_hero["ability_ready"] is not True
    assert target(path, "18.600.7")["verified"] is False


def test_below_point_95_portrait_rejects_with_other_same_frame_evidence_intact(monkeypatch):
    path = FIXTURES / NAMES[0]
    original = unit_catalog.recognize_card_identity

    def lower_score(*args, **kwargs):
        identity = original(*args, **kwargs)
        return {**identity, "confidence": .949}

    monkeypatch.setattr(unit_catalog, "recognize_card_identity", lower_score)
    state = recognize_hero_state(path, BOX, unit_id="grand_warden",
                                 client_version="18.647.2")
    evidence = state["evidence"]
    assert evidence["yellow_health_bar"] is not None
    assert evidence["closed_card"]["bbox"] == list(BOX)
    assert evidence["current_equipment_matches"]["ready_book"] is not None
    assert evidence["current_equipment_matches"]["ready_card"] is not None
    assert state["ability_ready"] is not True


def test_dynamic_reader_binds_original_sha_exact_physical_roi_and_ready_phase():
    path = FIXTURES / NAMES[0]
    frame = source_frame(path)
    positive = read_hero_deployment_evidence(path, frame, NATIVE_BOX, "grand_warden")
    assert positive is not None and positive.confidence >= .95
    assert read_hero_deployment_evidence(path, replace(frame, source_sha256="0" * 64),
                                         NATIVE_BOX, "grand_warden") is None
    shifted = tuple(value + (1 if index in (0, 2) else 0)
                    for index, value in enumerate(NATIVE_BOX))
    assert read_hero_deployment_evidence(path, frame, shifted, "grand_warden") is None
    assert read_hero_deployment_evidence(path, replace(frame, client_version="18.600.7"),
                                         NATIVE_BOX, "grand_warden") is None
    assert read_hero_deployment_evidence(path, frame, NATIVE_BOX, "archer_queen") is None


def test_yellow_used_cannot_become_available_deployment_evidence():
    frame = source_frame(OLD_USED)
    assert read_hero_deployment_evidence(OLD_USED, frame, NATIVE_BOX,
                                         "grand_warden") is None
    used = target(OLD_USED)
    assert used["verified"] is True
    assert used["state"]["ability_used"] is True
    assert used["state"]["ability_ready"] is False
