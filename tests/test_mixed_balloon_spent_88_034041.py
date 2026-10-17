"""The 03:40 mixed probe's anonymous card pixels, never a battle success fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.errors import FlowError
from autococ.frozen_line import _closed_card_border, _file_sha256
from autococ.images import read_frame
from autococ.mixed_probe import _probe_count
from autococ.probed_anchor import _spent_balloon_dark_border
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/mixed_balloon_spent_88_034041"
LIVE = (Path(r"E:\projects\AutoCOC") / "reports/completion-20261015/live" /
        "20261017-034041-064557-2538f096")
BBOX = (335, 595, 423, 711)


def _image(path: Path) -> np.ndarray:
    return cv2.resize(read_frame(path), (1280, 720), interpolation=cv2.INTER_AREA)


def _probe(paths: dict[str, Path], *, version: str = "18.647.2") -> dict:
    source, before, spent = (paths[name] for name in ("source", "before", "spent"))
    base = SimpleNamespace(
        toolbar_before_frame_id=str(before),
        frame=SimpleNamespace(screenshot_path=source, frame_id=str(source),
                              resolution=(1280, 720)),
        frame_sha256=_file_sha256(source), client_version=version)
    plan = SimpleNamespace(base=base, balloon_point=(379, 653), balloon_bbox=BBOX)
    snapshot = SceneSnapshot("battle", .99, spent, {
        "recognition_scope": "full", "fast_fallback": True,
        "recognition_requested": "battle_hud", "ocr": [], "battle": {"slots": []}})
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    return _probe_count(session, plan, snapshot, "balloon", 0, selected=True, pending=1)


def test_anonymous_pixels_and_original_provenance():
    manifest = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "autococ.anonymous_mixed_balloon_spent_88.v1"
    assert manifest["client_version"] == "18.647.2"
    assert manifest["truth_status"] == "offline_card_evidence_only"
    assert manifest["formal_acceptance"] is False
    assert manifest["human_reviewed"] is False
    assert manifest["retained_native_xyxy"] == [656, 1170, 860, 1440]
    if LIVE.is_dir():
        assert _file_sha256(LIVE / "events.jsonl") == manifest["source_events_sha256"]
    for row in manifest["rows"]:
        fixture = FIXTURES / row["fixture"]
        assert _file_sha256(fixture) == row["fixture_sha256"]
        image = read_frame(fixture)
        assert image.shape == (1440, 2560, 3)
        mask = np.zeros(image.shape[:2], dtype=bool)
        left, top, right, bottom = manifest["retained_native_xyxy"]
        mask[top:bottom, left:right] = True
        assert np.all(image[~mask] == 45)
        original = LIVE / "frames" / row["source_frame"]
        if original.is_file():
            assert hashlib.sha256(original.read_bytes()).hexdigest() == row["source_sha256"]
            assert np.array_equal(image[mask], read_frame(original)[mask])


@pytest.mark.parametrize("source", ["anonymous", "original"])
def test_real_same_frame_x0_portrait_and_closed_outline(source):
    manifest = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    paths = {row["role"]: (FIXTURES / row["fixture"] if source == "anonymous" else
                            LIVE / "frames" / row["source_frame"])
             for row in manifest["rows"]}
    if source == "original" and not all(path.is_file() for path in paths.values()):
        pytest.skip("Optional original live frames are not present")
    for row in manifest["rows"]:
        assert _file_sha256(paths[row["role"]]) == row[
            "fixture_sha256" if source == "anonymous" else "source_sha256"]
    spent = _image(paths["spent"])
    assert _closed_card_border(spent, BBOX) is None  # The previously failed path.
    border = _spent_balloon_dark_border(spent, BBOX)
    assert border is not None
    assert border["method"] == "spent_balloon_versioned_bottom_shape_local_background"
    assert border["core_max_error"] <= 1
    assert border["low_rebound_columns"] == 7
    assert border["low_rebound_span"] == [30, 37]
    assert border["lower_rebound_min"] >= 3
    reading = _probe(paths)
    assert reading["count"] == 0 and reading["confidence"] >= .9
    assert reading["portrait_score"] >= .92
    assert reading["portrait_source"] == "current_frame_saved_battle_portrait"
    assert reading["source_identity"]["unit_id"] == "balloon"
    assert reading["current_card_border"] == border


@pytest.mark.parametrize("damage", [
    "bottom_truncated", "wide_dark_below", "left_side", "right_side",
    "left_corner", "right_corner", "no_outline", "shifted_bbox"])
def test_local_background_exception_still_rejects_broken_cards(damage):
    image = _image(FIXTURES / "spent.png").copy()
    bbox = BBOX
    if damage == "bottom_truncated":
        image[712:720, 343:415] = 0
    elif damage == "wide_dark_below":
        image[713:720, 365:385] = 0
    elif damage == "left_side":
        image[625:695, 335:337] = 45
    elif damage == "right_side":
        image[625:695, 421:423] = 45
    elif damage == "left_corner":
        image[704:713, 335:344] = 45
    elif damage == "right_corner":
        image[704:713, 414:423] = 45
    elif damage == "no_outline":
        image[590:720, 330:430] = 45
    else:
        bbox = (336, 595, 424, 711)
    assert _spent_balloon_dark_border(image, bbox) is None


@pytest.mark.parametrize("gaps", [(3,), (20, 50), tuple(range(23, 43))])
def test_local_background_exception_requires_one_short_central_interval(gaps):
    image = _image(FIXTURES / "spent.png").copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for position in gaps:
        x = 343 + position
        value = int(round(float(gray[711:713, x].mean()) + 5))
        image[713:716, x] = (value, value, value)
    assert _spent_balloon_dark_border(image, BBOX) is None


def test_local_background_exception_rejects_changed_versioned_core():
    image = _image(FIXTURES / "spent.png").copy()
    image[709, 360] = (80, 80, 80)
    assert _spent_balloon_dark_border(image, BBOX) is None


def test_local_background_exception_rejects_changed_template_hash(monkeypatch, tmp_path):
    from autococ.unit_catalog import CATALOG_ROOT

    template = CATALOG_ROOT / "units/balloon_spent_bottom_18_647_2.png"
    destination = tmp_path / "units"
    destination.mkdir()
    (destination / template.name).write_bytes(template.read_bytes() + b"modified")
    monkeypatch.setattr("autococ.unit_catalog.CATALOG_ROOT", tmp_path)
    assert _spent_balloon_dark_border(_image(FIXTURES / "spent.png"), BBOX) is None


def test_same_frame_proof_rejects_wrong_version():
    paths = {name: FIXTURES / f"{name}.png" for name in ("before", "source", "spent")}
    with pytest.raises(FlowError, match="lacks a closed card border"):
        _probe(paths, version="18.600.7")
