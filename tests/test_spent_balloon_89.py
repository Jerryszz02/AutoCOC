"""Source-bound pixel proof for the sampled 89px spent balloon card.

These failed-battle frames are development evidence, not human labels or a
verified deployment. No OCR provider or device is used in this module.
"""

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
from autococ.ocr import OCRText
from autococ.probed_anchor import _balloon_count, _spent_balloon_dark_border
from autococ.scene import SceneSnapshot
from autococ.unit_catalog import CATALOG_ROOT
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"E:\projects\AutoCOC")
METADATA = ROOT / "tests/fixtures/mixed_balloon_spent_89_18_647_2_20261016.json"
BBOX = (383, 595, 472, 711)
TEMPLATE = CATALOG_ROOT / "units/balloon_spent_bottom_89_18_647_2.png"
LOCAL_OCR = ROOT / "tests/fixtures/mixed_balloon_spent_89_local_ocr_replay_20261016.json"


def _rows():
    return json.loads(METADATA.read_text(encoding="utf-8"))["frames"]


def _path(number):
    return ROOT / next(row["path"] for row in _rows()
                       if Path(row["source_frame"]).name.startswith(f"{number:05d}"))


def _image(number):
    return cv2.resize(read_frame(_path(number)), (1280, 720),
                      interpolation=cv2.INTER_AREA)


def _retained(items, rectangles):
    return [item for item in items if item.get("bbox") and
            any(left <= item["bbox"][0] < item["bbox"][2] <= right and
                top <= item["bbox"][1] < item["bbox"][3] <= bottom
                for left, top, right, bottom in rectangles)]


def test_spent_89_fixtures_are_derived_and_ocr_is_filtered():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["client_version"] == "18.647.2"
    assert metadata["human_reviewed"] is False
    assert metadata["formal_success"] is False
    assert metadata["counts_toward_acceptance"] is False
    assert [Path(row["source_frame"]).name[:5] for row in metadata["frames"]] == [
        "00014", "00016", "00017", "00018"]
    for row in metadata["frames"]:
        path = ROOT / row["path"]
        image = read_frame(path)
        assert image.shape == (1440, 2560, 3)
        assert _file_sha256(path) == row["sha256"]
        retained = np.zeros(image.shape[:2], dtype=bool)
        for left, top, right, bottom in row["retained_native_rectangles_xyxy"]:
            retained[top:bottom, left:right] = True
        assert np.all(image[~retained] == 45)
        assert row["cached_ocr"] == _retained(
            row["cached_ocr"], row["retained_baseline_rectangles_xyxy"])
        source = SOURCE_ROOT / row["source_frame"]
        if source.is_file():
            assert _file_sha256(source) == row["source_sha256"]
            assert np.array_equal(image[retained], read_frame(source)[retained])
        events = SOURCE_ROOT / row["source_events"]
        if events.is_file():
            assert _file_sha256(events) == row["source_events_sha256"]
            original = [event for line in events.read_text(encoding="utf-8").splitlines()
                        if (event := json.loads(line)).get("kind") == "observation"
                        and Path(str(event.get("frame"))) == source]
            assert len(original) == 1
            assert row["cached_ocr"] == _retained(
                original[0]["observations"].get("ocr") or [],
                row["retained_baseline_rectangles_xyxy"])


def test_spent_89_template_is_exact_source_crop():
    provenance = json.loads(TEMPLATE.with_suffix(".provenance.json").read_text(
        encoding="utf-8"))
    assert provenance["client_version"] == "18.647.2"
    assert provenance["source_sha256"] == _rows()[-1]["source_sha256"]
    assert provenance["baseline_roi_xyxy"] == [391, 706, 464, 713]
    assert provenance["template_shape"] == [7, 73]
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == provenance["template_sha256"]
    template = cv2.imread(str(TEMPLATE), cv2.IMREAD_GRAYSCALE)
    assert template.shape == (7, 73)
    assert np.array_equal(template, cv2.cvtColor(_image(18), cv2.COLOR_BGR2GRAY)[
        706:713, 391:464])


def test_spent_89_real_pixels_require_complete_separate_gray_outline():
    for number in (14, 16, 17):
        assert _spent_balloon_dark_border(_image(number), BBOX) is None
    image = _image(18)
    assert _closed_card_border(image, BBOX) is None  # Reproduce the live stop.
    border = _spent_balloon_dark_border(image, BBOX)
    assert border["method"] == "spent_balloon_89_versioned_closed_outline"
    assert border["left_edge"] >= 200 and border["right_edge"] <= -200
    assert border["straight_left_min"] >= 200
    assert border["straight_right_max"] <= -130
    assert border["corner_trace"] == [
        [702, 384, 470], [703, 384, 470], [704, 384, 470],
        [705, 384, 470], [706, 385, 469], [707, 385, 469],
        [708, 386, 468], [709, 387, 467], [710, 389, 465]]
    assert border["core_max_error"] == 0
    assert border["core_template_sha256"] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    assert border["lower_rebound_min"] >= 7
    assert border["continuous_columns"] == 73


@pytest.mark.parametrize("damage", (
    "left_side", "right_side", "left_straight", "right_straight",
    "left_corner", "right_corner", "bottom", "below", "flat_gray",
    "shifted_bbox", "wrong_width"))
def test_spent_89_rejects_missing_side_corner_bottom_or_geometry(damage):
    image = _image(18).copy()
    bbox = BBOX
    if damage == "left_side":
        image[625:695, 381:386] = 45
    elif damage == "right_side":
        image[625:695, 469:474] = 45
    elif damage == "left_straight":
        image[696:703, 381:386] = 45
    elif damage == "right_straight":
        image[696:703, 469:474] = 45
    elif damage == "left_corner":
        image[702:706, 381:394] = 45
    elif damage == "right_corner":
        image[702:706, 461:474] = 45
    elif damage == "bottom":
        image[711:713, 391:464] = 45
    elif damage == "below":
        image[713:716, 391:464] = 0
    elif damage == "flat_gray":
        image[625:716, 383:472] = 45
    elif damage == "shifted_bbox":
        bbox = (384, 595, 473, 711)
    else:
        bbox = (383, 595, 471, 711)
    if damage in {"left_side", "right_side", "left_straight", "right_straight",
                  "left_corner", "right_corner"}:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        if damage == "left_side":
            assert float(np.median(edges[625:695, 383])) < 200
        elif damage == "right_side":
            assert float(np.median(edges[625:695, 471])) > -200
        elif damage == "left_straight":
            assert np.any(edges[696:702, 383] < 200)
        elif damage == "right_straight":
            assert np.any(edges[696:702, 471] > -200)
        elif damage == "left_corner":
            assert float(edges[704, 383:392].max()) < 40
        else:
            assert float(edges[704, 463:472].min()) > -40
    assert _spent_balloon_dark_border(image, bbox) is None


def test_spent_89_rejects_changed_template_hash(monkeypatch, tmp_path):
    copied = tmp_path / "units"
    copied.mkdir()
    (copied / TEMPLATE.name).write_bytes(TEMPLATE.read_bytes() + b"changed")
    monkeypatch.setattr("autococ.unit_catalog.CATALOG_ROOT", tmp_path)
    assert _spent_balloon_dark_border(_image(18), BBOX) is None


@pytest.mark.parametrize("caller", ("probed", "mixed"))
@pytest.mark.parametrize("failure", (None, "wrong_version", "changed_source", "wrong_identity"))
def test_spent_89_keeps_source_identity_and_version_gates(monkeypatch, caller, failure):
    """Replay the separately measured local OCR; portrait/source remain pixels."""
    source, spent = _path(14), _path(18)
    local_ocr = json.loads(LOCAL_OCR.read_text(encoding="utf-8"))
    assert local_ocr["source_frame_sha256"] == _rows()[0]["source_sha256"]
    assert local_ocr["current_frame_sha256"] == _rows()[-1]["source_sha256"]
    assert local_ocr["events_sha256"] == _rows()[-1]["source_events_sha256"]
    assert local_ocr["raw_calls"][0]["output"][0]["confidence"] < .9
    assert local_ocr["raw_calls"][1]["output"][0]["confidence"] >= .9
    report = SOURCE_ROOT / "reports/completion-20261015/diagnostics/" \
        "mixed-balloon-89-spent-local-ocr-replay-20261016.json"
    if report.is_file():
        assert _file_sha256(report) == local_ocr["source_report_sha256"]
    base = SimpleNamespace(client_version="18.647.2",
                           frame=SimpleNamespace(screenshot_path=source,
                                                 frame_id=str(source)),
                           frame_sha256=_file_sha256(source),
                           toolbar_before_frame_id=str(source))
    if failure == "wrong_version":
        base.client_version = "18.600.7"
    elif failure == "changed_source":
        base.frame_sha256 = "0" * 64
    elif failure == "wrong_identity":
        monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity",
                            lambda *_a, **_kw: {"unit_id": "dragon", "confidence": .99,
                                                  "version_verified": True,
                                                  "template_client_version": "18.647.2"})
    plan = SimpleNamespace(base=base, balloon_point=(427, 653), balloon_bbox=BBOX)
    cached_full_ocr = _rows()[-1]["cached_ocr"]
    snapshot = SceneSnapshot("battle", .99, spent, {"recognition_scope": "full",
                                                  "ocr": cached_full_ocr,
                                                  "battle": {"slots": []}})
    # The original live event did not retain x0. These exact raw/threshold
    # values were measured later on the same frames by real local CPU OCR.
    # Only replay the reader; the portrait, identity and border use pixels.
    calls = iter(local_ocr["raw_calls"])

    def reader(crop):
        call = next(calls)
        assert list(crop.shape) == call["input_shape"]
        return [OCRText(**item) for item in call["output"]]

    provider = SimpleNamespace(recognize_line_image=reader)
    recognizer = ScreenshotRecognizer(provider=provider)
    recognizer.client_version = "18.647.2"
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    generic_calls = []
    # A permissive generic result must not skip the 89px version/source gates.
    def generic_border(_image, _bbox):
        generic_calls.append(True)
        return {"generic_would_accept": True}

    module = "autococ.probed_anchor" if caller == "probed" else "autococ.mixed_probe"
    monkeypatch.setattr(f"{module}._closed_card_border", generic_border)

    def probe():
        return (_balloon_count(session, plan, snapshot, 0) if caller == "probed" else
                _probe_count(session, plan, snapshot, "balloon", 0))

    if failure is None:
        result = probe()
        assert result["count"] == 0
        assert result["confidence"] == local_ocr["post_troop_reading"]["confidence"]
        assert result["portrait_score"] >= .92
        assert result["portrait_source"] == "current_frame_saved_battle_portrait"
        source_identity = result.get("source_identity") or result.get("source_portrait_identity")
        assert source_identity["unit_id"] == "balloon"
        assert result["current_card_border"]["method"] == \
               "spent_balloon_89_versioned_closed_outline"
    else:
        with pytest.raises(FlowError):
            probe()
    assert generic_calls == []
    assert next(calls, None) is None


@pytest.mark.parametrize("caller", ("probed", "mixed"))
@pytest.mark.parametrize("asset", ("missing", "changed"))
def test_spent_89_callers_cannot_bypass_missing_or_bad_asset_with_generic_border(
        monkeypatch, caller, asset):
    source, spent = _path(14), _path(18)
    bbox, point = BBOX, (427, 653)
    base = SimpleNamespace(client_version="18.647.2",
                           frame=SimpleNamespace(screenshot_path=source, frame_id=str(source)),
                           frame_sha256=_file_sha256(source),
                           toolbar_before_frame_id=str(source))
    plan = SimpleNamespace(base=base, balloon_point=point, balloon_bbox=bbox)
    snapshot = SceneSnapshot("battle", .99, spent, {"recognition_scope": "full",
                                                  "ocr": _rows()[-1]["cached_ocr"],
                                                  "battle": {"slots": []}})
    raw_calls = iter(json.loads(LOCAL_OCR.read_text(encoding="utf-8"))["raw_calls"])

    def reader(crop):
        call = next(raw_calls)
        assert list(crop.shape) == call["input_shape"]
        return [OCRText(**item) for item in call["output"]]

    recognizer = ScreenshotRecognizer(provider=SimpleNamespace(recognize_line_image=reader))
    recognizer.client_version = "18.647.2"
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    original_read_bytes = Path.read_bytes

    def damaged_asset(path):
        if path == TEMPLATE:
            if asset == "missing":
                raise FileNotFoundError(str(path))
            return original_read_bytes(path) + b"changed"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", damaged_asset)
    generic_calls = []

    def generic_border(_image, _bbox):
        generic_calls.append(True)
        return {"generic_would_accept": True}

    module = "autococ.probed_anchor" if caller == "probed" else "autococ.mixed_probe"
    monkeypatch.setattr(f"{module}._closed_card_border", generic_border)
    with pytest.raises(FlowError, match="spent balloon.*closed.*border"):
        if caller == "probed":
            _balloon_count(session, plan, snapshot, 0)
        else:
            _probe_count(session, plan, snapshot, "balloon", 0)
    assert generic_calls == []
    assert next(raw_calls, None) is None


@pytest.mark.parametrize("caller", ("probed", "mixed"))
def test_spent_88_callers_keep_generic_border_path(monkeypatch, caller):
    source = ROOT / "tests/fixtures/probed_balloon_024858_source.png"
    spent = ROOT / "tests/fixtures/probed_balloon_024858_spent.png"
    bbox, point = (335, 595, 423, 711), (379, 653)
    base = SimpleNamespace(client_version="18.647.2",
                           frame=SimpleNamespace(screenshot_path=source, frame_id=str(source)),
                           frame_sha256=_file_sha256(source),
                           toolbar_before_frame_id=str(source))
    plan = SimpleNamespace(base=base, balloon_point=point, balloon_bbox=bbox)
    snapshot = SceneSnapshot("battle", .99, spent, {"recognition_scope": "full",
                                                  "ocr": [], "battle": {"slots": []}})
    recognizer = ScreenshotRecognizer(provider=SimpleNamespace(
        recognize_line_image=lambda _crop: [OCRText("x0", .99, (0, 0, 1, 1))]))
    recognizer.client_version = "18.647.2"
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    generic_calls = []

    def generic_border(_image, _bbox):
        generic_calls.append(True)
        return {"legacy": True}

    module = "autococ.probed_anchor" if caller == "probed" else "autococ.mixed_probe"
    monkeypatch.setattr(f"{module}._closed_card_border", generic_border)
    if caller == "probed":
        reading = _balloon_count(session, plan, snapshot, 0)
    else:
        reading = _probe_count(session, plan, snapshot, "balloon", 0)
    assert reading["current_card_border"] == {"legacy": True}
    assert generic_calls == [True]
