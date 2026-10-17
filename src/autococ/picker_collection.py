"""Bounded read-only current troop pages and first spell page, client 18.647.2.

No unit/preset/hero selection, roster mutation, search, purchase, or automatic
identity labels. This is a development evidence collector, not live acceptance.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Callable

import cv2
import numpy as np


VERSION = "18.647.2"
VERSION_CODE = 180647003
BASE = (1280, 720)
PACKAGE = "com.supercell.clashofclans"
REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = REPO_ROOT / "assets" / "picker" / VERSION
CALIBRATION_JSON_SHA256 = "89fbc6d767ccb7ff86924a3ca6e51599851280bd6e435fdb76363af2c6dc81a6"
COLLECTOR_VERSION = "project_picker_collection_v1"
SOURCE_FILES = ("src/autococ/army_navigation.py", "src/autococ/army_editor.py",
                "src/autococ/session.py", "src/autococ/vision.py")
SWIPE_START, SWIPE_END, SWIPE_MS = (980, 560), (330, 560), 550
# Rectangles are at the recognizer's 1280×720 baseline. The packaged arrow
# template is an exact crop of the hash-bound historical first troop page.
ARROW_TEMPLATE_BOX = (1237, 551, 1260, 582)
ARROW_SEARCH_BOX = (1229, 545, 1270, 589)
BOTTOM_STRIP = (20, 420, 1260, 700)
TROOP_SELECTED = (550, 190, 1250, 294)
CURRENT_SELECTED = ((550, 190, 1250, 294), (550, 366, 994, 455),
                    (1000, 366, 1250, 455))
# Two independent old read-only swipes changed at most .003352 of selected-row
# pixels by >=25 despite unchanged army counts/identities. A real card or
# count change affects substantially more of the row. This remains a strict
# auxiliary pixel gate alongside fresh structured roster/capacity proof.
MAX_SELECTED_CHANGE_25 = .006
MAX_SELECTED_MEAN_DELTA = .25


class SamplingStopped(RuntimeError):
    pass


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise SamplingStopped("image_unreadable")
    return cv2.resize(image, BASE, interpolation=cv2.INTER_AREA)


def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def _pixels_sha(image: np.ndarray, box: tuple[int, int, int, int]) -> str:
    return hashlib.sha256(_crop(image, box).tobytes()).hexdigest()


def _selected_stable(before: np.ndarray, after: np.ndarray,
                     box: tuple[int, int, int, int]) -> dict:
    delta = cv2.absdiff(_crop(before, box), _crop(after, box))
    fraction = float(np.mean(np.max(delta, axis=2) >= 25))
    mean = float(np.mean(delta))
    return {"changed_fraction_25": round(fraction, 6),
            "mean_absolute_channel_delta": round(mean, 6),
            "before_sha256": _pixels_sha(before, box),
            "after_sha256": _pixels_sha(after, box),
            "stable": fraction <= MAX_SELECTED_CHANGE_25 and mean <= MAX_SELECTED_MEAN_DELTA}


def _page_changed(before: np.ndarray, after: np.ndarray) -> dict:
    a, b = _crop(before, BOTTOM_STRIP), _crop(after, BOTTOM_STRIP)
    changed = float(np.mean(np.max(cv2.absdiff(a, b), axis=2) >= 25))
    return {"changed_fraction_25": round(changed, 6),
            "before_strip_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "after_strip_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
            "verified": changed >= .08}


def _arrow_score(image: np.ndarray, source: np.ndarray) -> float:
    template = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(_crop(image, ARROW_SEARCH_BOX), cv2.COLOR_BGR2GRAY)
    return float(cv2.minMaxLoc(cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED))[1])


def _masked_arrow_evidence(image: np.ndarray, source: np.ndarray) -> dict:
    """Match the calibrated arrow glyph without the moving cards behind it.

    The immutable source asset supplies the neutral bright arrow and its two
    pixel bevel/outline. The rest of that rectangular crop is picker content,
    which changes on every page. Keep the existing .93 strong threshold and
    require exactly one small cluster of strong positions in the fixed search
    window. A non-finite correlation is never evidence.
    """
    template = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    neutral = source.max(axis=2).astype(np.int16) - source.min(axis=2).astype(np.int16)
    bright = ((template >= 180) & (neutral <= 20)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    if count < 2:
        return {"status": "invalid_template", "score": None}
    glyph = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    mask = cv2.dilate(glyph, np.ones((5, 5), dtype=np.uint8))
    target = cv2.cvtColor(_crop(image, ARROW_SEARCH_BOX), cv2.COLOR_BGR2GRAY)
    scores = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    finite = np.isfinite(scores)
    maximum = float(scores[finite].max()) if finite.any() else None
    strong = (finite & (scores >= .93)).astype(np.uint8)
    groups, _, positions, _ = cv2.connectedComponentsWithStats(strong)
    evidence = {"status": "unverified", "score": maximum,
                "method": "calibrated_neutral_glyph_with_outline_v1",
                "mask_pixels": int(mask.sum()), "strong_position_groups": groups - 1}
    if groups != 2:
        if groups > 2:
            evidence["status"] = "ambiguous"
        return evidence
    x, y, width, height, _ = positions[1]
    # Neighbouring correlation samples may describe one subpixel-aligned
    # glyph. A broad plateau or separate match is not a unique physical arrow.
    if width > 3 or height > 3:
        return {**evidence, "status": "ambiguous"}
    best = np.where(finite, scores, -1.0)
    _, _, _, point = cv2.minMaxLoc(best)
    evidence.update(status="verified", point=[ARROW_SEARCH_BOX[0] + point[0],
                                                ARROW_SEARCH_BOX[1] + point[1]],
                    strong_position_extent=[int(x), int(y), int(width), int(height)])
    return evidence


def _calibration() -> dict:
    directory = CALIBRATION_DIR.resolve()
    if not directory.is_relative_to(REPO_ROOT.resolve()):
        raise SamplingStopped("calibration_directory_escapes_project")
    path = (directory / "calibration.json").resolve()
    if path.parent != directory:
        raise SamplingStopped("calibration_manifest_escapes_project")
    if sha(path) != CALIBRATION_JSON_SHA256:
        raise SamplingStopped("calibration_manifest_changed")
    calibration = json.loads(path.read_text(encoding="utf-8"))
    template = calibration.get("next_arrow_template", {})
    relative = template.get("path")
    if (calibration.get("schema") != "readonly_picker_calibration_v1"
            or calibration.get("client_version") != VERSION
            or calibration.get("client_version_code") != VERSION_CODE
            or calibration.get("package") != PACKAGE
            or calibration.get("baseline_resolution") != list(BASE)
            or calibration.get("swipe_start") != list(SWIPE_START)
            or calibration.get("swipe_end") != list(SWIPE_END)
            or calibration.get("swipe_ms") != SWIPE_MS
            or relative != "next-arrow.png"
            or calibration.get("arrow_search_box") != list(ARROW_SEARCH_BOX)
            or calibration.get("bottom_strip") != list(BOTTOM_STRIP)
            or calibration.get("troop_selected") != list(TROOP_SELECTED)
            or calibration.get("current_selected") != [list(box) for box in CURRENT_SELECTED]
            or calibration.get("next_arrow_absent_max") != .60
            or calibration.get("next_arrow_strong_min") != .93
            or calibration.get("min_bottom_strip_changed_fraction_25") != .08
            or calibration.get("max_selected_change_fraction_25") != MAX_SELECTED_CHANGE_25
            or calibration.get("max_selected_mean_delta") != MAX_SELECTED_MEAN_DELTA):
        raise SamplingStopped("calibration_contract_changed")
    template_path = (directory / relative).resolve()
    if template_path.parent != directory:
        raise SamplingStopped("calibration_template_escapes_project")
    if sha(template_path) != template.get("sha256"):
        raise SamplingStopped("calibration_template_changed")
    template_image = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if (template_image is None or template_image.shape != (31, 23, 3)
            or hashlib.sha256(template_image.tobytes()).hexdigest() != template.get("pixel_sha256_bgr")):
        raise SamplingStopped("calibration_template_unreadable")
    return calibration


def _control(editor: dict, action: str, kind: str | None = None) -> dict:
    controls = editor.get("controls")
    if not isinstance(controls, list):
        raise SamplingStopped("controls_unavailable")
    found = [item for item in controls if isinstance(item, dict) and item.get("action") == action
             and (kind is None or item.get("kind") == kind)
             and item.get("enabled") is True and item.get("cost_free") is True]
    if len(found) != 1:
        raise SamplingStopped(f"{action}_{kind or 'current'}_unverified")
    point = found[0].get("point")
    if (not isinstance(point, (list, tuple)) or len(point) != 2
            or any(type(x) is not int for x in point)
            or not (0 <= point[0] < BASE[0] and 0 <= point[1] < BASE[1])):
        raise SamplingStopped("control_point_unverified")
    return found[0]


def _army_counts(shot) -> tuple:
    army = shot.observations.get("army")
    if not isinstance(army, dict):
        raise SamplingStopped("army_counts_missing")
    result = []
    for name in ("heroes", "troops", "spells", "siege"):
        value = army.get(name)
        if (not isinstance(value, dict) or type(value.get("used")) is not int
                or type(value.get("capacity")) is not int or value["used"] < 0
                or value["used"] > value["capacity"]):
            raise SamplingStopped(f"army_{name}_count_unverified")
        result.append((name, value["used"], value["capacity"]))
    return tuple(result)


def _roster_signature(shot) -> tuple:
    army = shot.observations["army"]
    manifest = army.get("manifest")
    frame = str(shot.screenshot_path)
    if (not isinstance(manifest, dict) or manifest.get("frame") != frame
            or manifest.get("layout") != "my_army_rows_v1"
            or manifest.get("supported_layout") is not True
            or manifest.get("complete") is not True
            or manifest.get("identity_complete") is not True
            or manifest.get("unknowns") != []
            or manifest.get("complete_kinds") != {"troop": True, "spell": True}):
        raise SamplingStopped("current_roster_manifest_incomplete")
    coverage = army.get("identity_coverage")
    if (not isinstance(coverage, dict) or coverage.get("hero") is not True
            or coverage.get("siege") is not True):
        raise SamplingStopped("current_hero_siege_coverage_incomplete")
    for name in ("hero_evidence", "siege_evidence"):
        evidence = army.get(name)
        if (not isinstance(evidence, dict) or evidence.get("frame") != frame
                or evidence.get("complete") is not True or evidence.get("unknowns") != []):
            raise SamplingStopped(f"current_{name}_incomplete")
    result = []
    for name, kind in (("troops", "troop"), ("spells", "spell")):
        cards = manifest.get(name)
        if not isinstance(cards, list):
            raise SamplingStopped(f"current_{name}_row_missing")
        signature = []
        for card in cards:
            if (not isinstance(card, dict) or card.get("kind") != kind
                    or card.get("source") != "army"
                    or not isinstance(card.get("unit_id"), str) or not card["unit_id"]
                    or type(card.get("count")) is not int or card["count"] <= 0
                    or not isinstance(card.get("bbox"), list) or len(card["bbox"]) != 4
                    or any(type(value) is not int for value in card["bbox"])):
                raise SamplingStopped("current_roster_card_malformed")
            signature.append((tuple(card["bbox"]), card["unit_id"], card["count"]))
        if len({item[1] for item in signature}) != len(signature):
            raise SamplingStopped("current_roster_unit_duplicate")
        result.append((name, tuple(signature)))
    identities = army.get("identity_cards")
    if not isinstance(identities, list) or not identities:
        raise SamplingStopped("current_hero_siege_identity_missing")
    signature = []
    for card in identities:
        if (not isinstance(card, dict) or card.get("kind") not in {"hero", "siege"}
                or not isinstance(card.get("unit_id"), str) or not card["unit_id"]
                or type(card.get("count")) is not int or card["count"] < 1
                or not isinstance(card.get("bbox"), list) or len(card["bbox"]) != 4):
            raise SamplingStopped("current_hero_siege_identity_unverified")
        signature.append((card["kind"], card["unit_id"], card["count"], tuple(card["bbox"])))
    if len({item[1] for item in signature}) != len(signature):
        raise SamplingStopped("current_hero_siege_identity_duplicate")
    for kind, army_key in (("hero", "heroes"), ("siege", "siege")):
        if sum(item[2] for item in signature if item[0] == kind) != army[army_key]["used"]:
            raise SamplingStopped("current_hero_siege_identity_incomplete")
    result.append(("hero_and_siege", tuple(signature)))
    return tuple(result)


def _safe(session, shot, foreground, *, surface: str, kind: str | None = None) -> dict:
    session._validate_snapshot(shot)
    if (session.client_version != VERSION or session.client_version_code != VERSION_CODE
            or tuple(session.config.game.baseline_resolution) != BASE
            or shot.scene != "training" or not foreground()):
        raise SamplingStopped("version_scene_or_foreground_unverified")
    obs = shot.observations
    for name in ("account_in_use", "cloud_cover", "request_dialog", "surrender_dialog",
                 "unit_detail", "spell_detail", "hero_detail", "startup_logo"):
        if name not in obs or obs[name] is not None:
            raise SamplingStopped(f"overlay_{name}_unverified")
    editor = obs.get("army_editor")
    if (not isinstance(editor, dict) or editor.get("frame") != str(shot.screenshot_path)
            or editor.get("surface") != surface or editor.get("ready") is not True
            or (kind is not None and editor.get("editing_kind") != kind)):
        raise SamplingStopped("picker_surface_or_kind_unverified")
    session._validate_snapshot(shot)
    return editor


def _same_picker_visible(shot, baseline_counts: tuple, baseline_roster: tuple,
                         editor: dict, kind: str) -> dict:
    """Only the open picker's visible selected row is a same-frame roster proof.

    The spell picker may have army=None; hidden classes and their
    complete_kinds are deliberately not inferred while the tray is open.
    """
    if kind not in {"troop", "spell"} or editor.get("editing_kind") != kind:
        raise SamplingStopped("picker_kind_unverified")
    capacities = editor.get("capacities")
    if not isinstance(capacities, dict):
        raise SamplingStopped("picker_capacity_missing")
    name = kind + "s"
    expected = next((used, total) for group, used, total in baseline_counts if group == name)
    current = capacities.get(kind)
    if not isinstance(current, dict) or current != {"used": expected[0], "total": expected[1]}:
        raise SamplingStopped(f"picker_{kind}_capacity_missing_or_changed")
    complete_kinds = editor.get("complete_kinds")
    if (not isinstance(complete_kinds, dict) or complete_kinds.get(kind) is not True
            or editor.get("unknowns") != []):
        raise SamplingStopped(f"picker_{kind}_selected_row_incomplete")
    cards = editor.get("cards")
    if not isinstance(cards, list):
        raise SamplingStopped(f"picker_{kind}_selected_cards_missing")
    signature = []
    for card in cards:
        if (not isinstance(card, dict) or card.get("kind") != kind
                or card.get("source") != "army"
                or not isinstance(card.get("unit_id"), str) or not card["unit_id"]
                or type(card.get("count")) is not int or card["count"] <= 0
                or not isinstance(card.get("bbox"), list) or len(card["bbox"]) != 4
                or any(type(value) is not int for value in card["bbox"])):
            raise SamplingStopped(f"picker_{kind}_selected_card_malformed")
        signature.append((card["unit_id"], card["count"]))
    if len({item[0] for item in signature}) != len(signature):
        raise SamplingStopped(f"picker_{kind}_selected_identity_duplicate")
    original = next(row for row in baseline_roster if row[0] == name)[1]
    if tuple(signature) != tuple((unit_id, count) for _bbox, unit_id, count in original):
        raise SamplingStopped(f"picker_{kind}_selected_roster_changed")
    army = shot.observations.get("army")
    if army is not None:
        if not isinstance(army, dict):
            raise SamplingStopped("picker_army_observation_malformed")
        value = army.get(name)
        if (not isinstance(value, dict) or value.get("used") != expected[0]
                or value.get("capacity") != expected[1]):
            raise SamplingStopped(f"picker_{kind}_army_header_conflict")
    return {"kind": kind, "capacity": current, "selected": signature,
            "army_header_present": army is not None}


def run_flow(session, *, foreground, open_army, max_troop_pages: int,
             pause=time.sleep, clock=time.monotonic, limit_sec: int = 150,
             stop: Callable[[], bool] = lambda: False) -> dict:
    if type(max_troop_pages) is not int or not 1 <= max_troop_pages <= 5:
        raise ValueError("max_troop_pages must be 1..5")
    calibration = _calibration()
    arrow_source = cv2.imdecode(np.fromfile(CALIBRATION_DIR / calibration["next_arrow_template"]["path"],
                                           dtype=np.uint8), cv2.IMREAD_COLOR)
    started = clock()
    pages: list[dict] = []
    def check():
        if stop():
            raise SamplingStopped("stopped_by_user")
        session.check_deadline()
        if clock() - started >= limit_sec:
            raise SamplingStopped("sampling_time_limit")
    def observe(label):
        check()
        shot = session.observe(label)
        if shot.scene in {"unknown", "account_in_use"}:
            raise SamplingStopped(f"scene_{shot.scene}")
        check()
        return shot
    def tap_checked(shot, point, *, reason):
        check()
        if not foreground():
            raise SamplingStopped("game_not_foreground_before_input")
        check()
        session.tap(shot, point, reason=reason)
    home = observe("picker-v2-home")
    session._validate_snapshot(home)
    if (session.client_version != VERSION or session.client_version_code != VERSION_CODE
            or home.scene != "village" or not foreground()):
        raise SamplingStopped("fresh_village_or_version_unverified")
    check()
    current = open_army(session, home, [])
    editor = _safe(session, current, foreground, surface="current")
    baseline_counts, baseline_roster = _army_counts(current), _roster_signature(current)
    baseline_image = _image(current.screenshot_path)
    baseline_pixels = [_pixels_sha(baseline_image, box) for box in CURRENT_SELECTED]
    baseline = {"frame": str(current.screenshot_path), "sha256": sha(current.screenshot_path),
                "counts": baseline_counts, "roster": baseline_roster,
                "selected_pixels_sha256": baseline_pixels}
    _control(editor, "open_picker", "troop")
    check()
    tap_checked(current, _control(editor, "open_picker", "troop")["point"],
                reason="Development read-only open troop picker")
    pause(.8)
    picker = observe("troop-picker-page-1")
    first_editor = _safe(session, picker, foreground, surface="current_picker", kind="troop")
    _same_picker_visible(picker, baseline_counts, baseline_roster, first_editor, "troop")
    first_image = _image(picker.screenshot_path)
    first_pixels = _pixels_sha(first_image, TROOP_SELECTED)
    termination = None
    for page_index in range(max_troop_pages):
        editor = _safe(session, picker, foreground, surface="current_picker", kind="troop")
        visible = _same_picker_visible(picker, baseline_counts, baseline_roster, editor, "troop")
        image = _image(picker.screenshot_path)
        selected_stability = _selected_stable(first_image, image, TROOP_SELECTED)
        if not selected_stability["stable"]:
            raise SamplingStopped("selected_troop_row_changed")
        score = _arrow_score(image, arrow_source)
        if not math.isfinite(score):
            raise SamplingStopped("next_arrow_nonfinite")
        legacy_score = score
        arrow_evidence = {"status": "verified" if score >= .93 else "unverified",
                          "method": "full_crop_correlation", "score": score}
        if score < .93:
            arrow_evidence = _masked_arrow_evidence(image, arrow_source)
            if arrow_evidence["status"] == "verified":
                score = arrow_evidence["score"]
        record = {"index": page_index + 1, "frame": str(picker.screenshot_path),
                  "sha256": sha(picker.screenshot_path), "next_arrow_score": round(score, 6),
                  "next_arrow_full_crop_score": round(legacy_score, 6),
                  "next_arrow_evidence": arrow_evidence,
                  "selected_row_initial_sha256": first_pixels,
                  "selected_row_stability": selected_stability,
                  "visible_roster_proof": visible, "identity_labels": None}
        pages.append(record)
        session.event("development_readonly_picker_page", **record)
        if page_index + 1 >= max_troop_pages:
            termination = "configured_page_limit_not_catalog_end"
            break
        if arrow_evidence["status"] == "ambiguous":
            raise SamplingStopped("next_arrow_ambiguous")
        if score <= .60:
            termination = "no_supported_next_arrow_not_catalog_end"
            break
        if score < .93:
            raise SamplingStopped("next_arrow_ambiguous")
        check()
        session._validate_snapshot(picker)
        if not foreground():
            raise SamplingStopped("game_not_foreground_before_swipe")
        session.event("development_readonly_troop_swipe_attempt", frame=str(picker.screenshot_path),
                      frame_sha256=record["sha256"], start=list(SWIPE_START),
                      end=list(SWIPE_END), duration_ms=SWIPE_MS)
        session._input_attempt_generation += 1
        try:
            check()
            if not foreground():
                raise SamplingStopped("game_not_foreground_before_swipe")
            check()
            session.context.swipe(*SWIPE_START, *SWIPE_END, SWIPE_MS)
            session.action_count += 1
        finally:
            session.last_snapshot = None
        check()
        pause(.8)
        after = observe(f"troop-picker-page-{page_index + 2}")
        after_editor = _safe(session, after, foreground, surface="current_picker", kind="troop")
        _same_picker_visible(after, baseline_counts, baseline_roster, after_editor, "troop")
        after_image = _image(after.screenshot_path)
        if not _selected_stable(first_image, after_image, TROOP_SELECTED)["stable"]:
            raise SamplingStopped("selected_troop_row_changed_after_swipe")
        progress = _page_changed(image, after_image)
        record["swipe_result"] = {**progress, "after_frame": str(after.screenshot_path),
                                  "after_sha256": sha(after.screenshot_path)}
        session.event("development_readonly_picker_page_change", index=page_index + 1,
                      before_frame=record["frame"], before_sha256=record["sha256"],
                      **record["swipe_result"])
        if not progress["verified"]:
            raise SamplingStopped("page_advance_unverified")
        picker = after
    if termination is None:
        raise SamplingStopped("page_termination_unverified")
    session.event("development_readonly_picker_termination", reason=termination,
                  frame=str(picker.screenshot_path), frame_sha256=sha(picker.screenshot_path),
                  pages_seen=len(pages))
    editor = _safe(session, picker, foreground, surface="current_picker", kind="troop")
    _same_picker_visible(picker, baseline_counts, baseline_roster, editor, "troop")
    check()
    tap_checked(picker, _control(editor, "close_picker")["point"],
                reason="Development read-only close troop picker")
    pause(.8)
    current = observe("troop-picker-closed")
    editor = _safe(session, current, foreground, surface="current")
    _same_current(current, baseline_image, baseline_counts, baseline_roster)
    check()
    tap_checked(current, _control(editor, "open_picker", "spell")["point"],
                reason="Development read-only open spell picker")
    pause(.8)
    spell = observe("spell-picker-first-page")
    spell_editor = _safe(session, spell, foreground, surface="current_picker", kind="spell")
    spell_visible = _same_picker_visible(spell, baseline_counts, baseline_roster, spell_editor, "spell")
    spell_record = {"frame": str(spell.screenshot_path), "sha256": sha(spell.screenshot_path),
                    "identity_labels": None, "swipe_attempted": False,
                    "visible_roster_proof": spell_visible,
                    "next_page_evidence": "not_calibrated; first page only"}
    check()
    tap_checked(spell, _control(spell_editor, "close_picker")["point"],
                reason="Development read-only close spell picker")
    pause(.8)
    current = observe("spell-picker-closed")
    _safe(session, current, foreground, surface="current")
    final_deltas = _same_current(current, baseline_image, baseline_counts, baseline_roster)
    return {"army_before": baseline, "army_after": {"frame": str(current.screenshot_path),
            "sha256": sha(current.screenshot_path), "counts": _army_counts(current),
            "roster": _roster_signature(current),
            "selected_pixels_sha256": [_pixels_sha(_image(current.screenshot_path), box)
                                       for box in CURRENT_SELECTED],
            "selected_pixel_deltas": final_deltas},
            "troop_pages": pages, "troop_termination": termination,
            "spell_first_page": spell_record, "zero_roster_change_verified": True,
            "calibration": calibration}


def _same_current(shot, baseline_image: np.ndarray, counts: tuple, roster: tuple) -> list[dict]:
    if _army_counts(shot) != counts or _roster_signature(shot) != roster:
        raise SamplingStopped("current_army_structure_changed")
    image = _image(shot.screenshot_path)
    deltas = [_selected_stable(baseline_image, image, box) for box in CURRENT_SELECTED]
    if not all(item["stable"] for item in deltas):
        raise SamplingStopped("current_army_selected_pixels_changed")
    return deltas


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                   encoding="utf-8").strip()


def _open_army_checked(session, home, evidence: list, *, foreground: Callable[[], bool],
                       stop: Callable[[], bool]):
    """Navigate with a stop and foreground gate before each input."""
    if home.scene != "village" or session.client_version != VERSION:
        raise SamplingStopped("army_entrance_unverified")
    session.check_deadline()
    if stop() or not foreground():
        raise SamplingStopped("stopped_or_not_foreground_before_army_input")
    if stop():
        raise SamplingStopped("stopped_by_user")
    session.click(home, "attack")
    menu = session.wait_for({"search"}, timeout_sec=30, label="army-menu")
    evidence.append(menu.screenshot_path)
    session.check_deadline()
    if stop() or not foreground():
        raise SamplingStopped("stopped_or_not_foreground_before_army_input")
    if stop():
        raise SamplingStopped("stopped_by_user")
    session.click(menu, "edit_army")
    return session.wait_for({"training"}, timeout_sec=30, label="inspect-army")


def collect_picker(config_path: Path, output: Path, *, max_troop_pages: int = 3,
                   limit_sec: int = 150, stop: Callable[[], bool] = lambda: False,
                   repo_root: Path | None = None) -> dict:
    """Collect one read-only picker run in a fresh directory outside the checkout."""
    raw_output = output
    repo = REPO_ROOT.resolve()
    if repo_root is not None and repo_root.resolve() != repo:
        raise ValueError("repo_root must match the loaded picker collector checkout")
    config_path, output = config_path.resolve(), output.resolve()
    if (type(limit_sec) is not int or not 60 <= limit_sec <= 180
            or type(max_troop_pages) is not int or not 1 <= max_troop_pages <= 5
            or output.is_relative_to(repo) or output.exists() or raw_output.is_symlink()
            or not config_path.is_file() or not (repo / "src/autococ/session.py").is_file()):
        raise ValueError("Invalid time/pages, source/config, or existing/inside-checkout output directory")
    calibration = _calibration()
    head, source_status = _git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain")
    if source_status:
        raise SamplingStopped("Project source must be clean before device sampling")
    if stop():
        raise SamplingStopped("stopped_by_user")
    source_hashes = {name: sha(repo / name) for name in SOURCE_FILES}
    config_hash, module_hash = sha(config_path), sha(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    result = {"kind": "agent_development_readonly_picker_sampling_v2", "status": "failed",
              "reason": "not_started", "counts_toward_acceptance": False,
              "formal_acceptance": False, "human_review": False,
              "unit_identity_labels": None, "client_version": None, "action_count": 0,
              "repo_root": str(repo), "head_before": head,
              "source_sha256_before": source_hashes,
              "config_sha256_before": config_hash, "runner_sha256_before": module_hash,
              "collector_version": COLLECTOR_VERSION,
              "collector_module_sha256_before": module_hash,
              "calibration_path": "assets/picker/18.647.2/calibration.json",
              "calibration_sha256": CALIBRATION_JSON_SHA256,
              "calibration_template_sha256_before": calibration["next_arrow_template"]["sha256"],
              "events_path": str(output / "events.jsonl"), "flow": None}
    session = None
    started = time.monotonic()
    try:
        from .config import load_config
        from .device import DeviceManager, foreground_package
        from .session import GameSession

        config = load_config(config_path)
        if (tuple(config.game.baseline_resolution) != BASE or config.game.package_name != PACKAGE):
            raise SamplingStopped("config_resolution_or_package_unverified")
        manager = DeviceManager(config)
        device = manager.connect()
        def foreground() -> bool:
            return foreground_package(manager.adb, device.serial,
                                      timeout_sec=config.runtime.step_timeout_sec) == PACKAGE
        if not foreground():
            raise SamplingStopped("game_not_foreground")
        session = GameSession.connect(config, manager.adb, device.serial, output, launch=False)
        if session.client_version != VERSION or session.client_version_code != VERSION_CODE:
            raise SamplingStopped("client_version_unverified")
        session.begin_task()
        session.deadline = min(session.deadline, started + limit_sec)
        session.task_deadline = min(session.task_deadline, session.deadline)
        def open_army(current_session, home, evidence):
            return _open_army_checked(current_session, home, evidence,
                                      foreground=foreground, stop=stop)
        result["flow"] = run_flow(session, foreground=foreground, open_army=open_army,
                                  max_troop_pages=max_troop_pages,
                                  limit_sec=limit_sec - (time.monotonic() - started), stop=stop)
        if session.action_count != 6 + len(result["flow"]["troop_pages"]) - 1:
            raise SamplingStopped("unexpected_input_count")
        result["status"], result["reason"] = "completed", "read_only_capture_sequence_completed"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            result["action_count"] = session.action_count
            result["client_version"] = session.client_version
            result["client_version_code"] = session.client_version_code
            try:
                session.close()
            except Exception as exc:
                result["status"], result["reason"] = "failed", f"session_close_failed: {type(exc).__name__}: {exc}"
        result["elapsed_sec"] = round(time.monotonic() - started, 3)
        result["frames"] = [{"path": str(path), "sha256": sha(path)}
                            for path in sorted((output / "frames").glob("*.png"))]
        events = output / "events.jsonl"
        result["events_sha256"] = sha(events) if events.is_file() else None
        result["head_after"] = _git(repo, "rev-parse", "HEAD")
        result["source_status_after"] = _git(repo, "status", "--porcelain")
        result["source_sha256_after"] = {name: sha(repo / name) for name in SOURCE_FILES}
        result["config_sha256_after"] = sha(config_path)
        result["runner_sha256_after"] = sha(Path(__file__))
        result["collector_module_sha256_after"] = sha(Path(__file__))
        result["calibration_template_sha256_after"] = sha(CALIBRATION_DIR / "next-arrow.png")
        if (result["head_after"] != head or result["source_status_after"]
                or result["source_sha256_after"] != result["source_sha256_before"]
                or result["config_sha256_after"] != result["config_sha256_before"]
                or result["runner_sha256_after"] != result["runner_sha256_before"]
                or result["collector_module_sha256_after"] != result["collector_module_sha256_before"]
                or sha(CALIBRATION_DIR / "calibration.json") != CALIBRATION_JSON_SHA256
                or result["calibration_template_sha256_after"] != result["calibration_template_sha256_before"]):
            result["status"], result["reason"] = "failed", "source_or_config_changed_during_sampling"
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, default=list)
            stream.write("\n")
    return result
