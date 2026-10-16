"""Model-free, observation-free edrag_line burst from independently verified inputs.

This module does not discover terrain, prepare the army, release a model, or
enter battle. Callers must finish those observations before freezing a plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Sequence
from uuid import uuid4

from .errors import DeploymentError, FlowError
from .prepared_battle import FramePacket, PreparedInput
from .scene import SceneSnapshot
from .strategies import BattleContext, EdragLineStrategy, TwoEdgeSplitStrategy


_CLIENT_VERSION = "18.647.2"
_BASE_LINE_TROOP_IDS = frozenset({"electro_dragon", "dragon", "baby_dragon", "balloon"})
_ARCHER_FILLER_UNIT_ID = "archer"
_COUNT = re.compile(r"[xX×]\s*(\d+)")


@dataclass(frozen=True)
class FrozenLinePlan:
    plan_id: str
    battle_id: str
    frame: FramePacket
    frame_sha256: str
    toolbar_before_frame_id: str
    toolbar_before_sha256: str
    client_version: str
    frozen_at: float
    latest_start: float
    burst_limit_sec: float
    toolbar_sha256: str
    terrain_sha256: str
    actions: tuple[PreparedInput, ...]
    troop_budgets: tuple[tuple[str, str, int], ...]
    troop_cards: tuple[tuple[str, str, int, tuple[int, int, int, int], tuple[int, int, int, int]], ...]
    hero_ids: tuple[str, ...]
    hero_cards: tuple[tuple[str, tuple[int, int, int, int]], ...]
    hero_manifest_sha256: str | None
    hero_manifest_frame_id: str | None
    hero_manifest_frame_sha256: str | None
    minimum_zoom_sha256: str
    line_tactics: str
    plan_sha256: str


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             default=str).encode("utf-8")).hexdigest()


def _file_sha256(path: Any) -> str:
    digest = sha256()
    try:
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FlowError(f"Frozen line source frame is unavailable: {path}") from exc
    return digest.hexdigest()


def _plan_digest(plan: FrozenLinePlan) -> str:
    payload = asdict(plan)
    payload.pop("plan_sha256")
    return _digest(payload)


def _point(value: Any, width: int, height: int) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(type(item) is not int for item in value) or
            not (0 <= value[0] < width and 0 <= value[1] < height)):
        raise FlowError("Frozen line input point is outside the verified display")
    return tuple(value)


def _cards(snapshot: SceneSnapshot, *, client_version: str,
           expected: dict[tuple[str, str], int] | None = None) -> tuple[dict, ...]:
    battle = snapshot.observations.get("battle")
    slots = battle.get("slots") if isinstance(battle, dict) else None
    if not isinstance(slots, list) or any(not isinstance(item, dict) or item.get("kind") == "unknown"
                                               for item in slots):
        raise FlowError("Frozen line toolbar contains an unknown card")
    width, height = snapshot.observations.get("baseline_resolution", (1280, 720))
    if type(width) is not int or type(height) is not int:
        raise FlowError("Frozen line toolbar resolution is unknown")
    cards = []
    for slot in slots:
        if slot.get("kind") not in {"troop", "hero"}:
            continue
        kind, unit_id, source = slot["kind"], slot.get("unit_id"), slot.get("source")
        evidence = slot.get("evidence") if isinstance(slot.get("evidence"), dict) else {}
        identity = evidence.get("identity") if isinstance(evidence.get("identity"), dict) else {}
        confidence = slot.get("confidence")
        score = identity.get("confidence")
        bbox = slot.get("bbox")
        if (not isinstance(unit_id, str) or not unit_id or source != "army" or
                type(confidence) not in (int, float) or not math.isfinite(confidence) or confidence < .9 or
                type(score) not in (int, float) or not math.isfinite(score) or score < .9 or
                identity.get("unit_id") != unit_id or identity.get("version_verified") is not True or
                identity.get("template_client_version") != client_version or
                not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or
                any(type(value) is not int for value in bbox) or
                not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height)):
            raise FlowError("Frozen line card lacks current-version identity or geometry")
        point = _point(slot.get("point"), width, height)
        count = slot.get("count")
        if kind == "troop":
            reading = evidence.get("count") if isinstance(evidence.get("count"), dict) else {}
            anchor = evidence.get("count_roi_anchor") if isinstance(evidence.get("count_roi_anchor"), dict) else None
            read_bbox = tuple(anchor.get("source_slot_bbox") or ()) if anchor else tuple(bbox)
            if (anchor is not None and (anchor.get("verified_slot_bbox") != list(bbox) or
                                        anchor.get("verified_frame") != str(snapshot.screenshot_path) or
                                        anchor.get("source_frame") != evidence.get("identity_reference") or
                                        anchor.get("unit_id") != unit_id)):
                raise FlowError("Frozen line selected card count anchor is unbound")
            if not _count_proves(reading, count, snapshot.screenshot_path, read_bbox, unit_id, client_version):
                raise FlowError("Frozen line troop count lacks same-card OCR evidence")
            if expected is not None and expected.get((unit_id, source)) != count:
                raise FlowError("Frozen line troop count differs from independent army manifest")
        cards.append({"kind": kind, "unit_id": unit_id, "source": source,
                      "count": count, "point": point, "bbox": tuple(bbox)})
    keys = [(row["kind"], row["unit_id"], row["source"]) for row in cards]
    if len(set(keys)) != len(keys):
        raise FlowError("Frozen line toolbar has duplicate card identities")
    return tuple(cards)


def _count_proves(reading: dict, expected: int, frame: Any,
                  bbox: tuple[int, ...], unit_id: str, client_version: str) -> bool:
    confidence = reading.get("confidence")
    if (type(expected) is not int or expected <= 0 or type(confidence) not in (int, float) or
            not math.isfinite(confidence) or confidence < .9):
        return False
    if "readings" not in reading:
        match = _COUNT.fullmatch(str(reading.get("text", "")).strip())
        return match is not None and int(match[1]) == expected
    if (reading.get("count") != expected or reading.get("frame") != str(frame) or
            tuple(reading.get("slot_bbox") or ()) != bbox):
        return False
    score = reading.get("portrait_score")
    source = reading.get("portrait_source")
    if (type(score) not in (int, float) or not math.isfinite(score) or score < .92 or
            source not in {"current_frame_saved_battle_portrait", "current_frame_versioned_battle_template"}):
        return False
    if source == "current_frame_versioned_battle_template":
        identity = reading.get("portrait_identity") or {}
        if (identity.get("unit_id") != unit_id or identity.get("version_verified") is not True or
                identity.get("template_client_version") != client_version):
            return False
    reads = reading.get("readings")
    if not isinstance(reads, list) or not reads:
        return False
    counts = set()
    for item in reads:
        if not isinstance(item, dict):
            return False
        score = item.get("confidence")
        if type(score) not in (int, float) or not math.isfinite(score):
            return False
        if score >= .9:
            match = _COUNT.fullmatch(str(item.get("text", "")).strip())
            if match is not None:
                counts.add(int(match[1]))
    return counts == {expected}


def _closed_card_border(current: Any, bbox: tuple[int, int, int, int]) -> dict | None:
    """The same independent opposite edges and enclosed bottom used for spent cards."""
    import cv2
    import numpy as np

    left, _top, right, bottom = bbox
    if not (0 <= left < right < 1280 and 625 <= bottom - 4 and bottom + 3 <= 720 and
            left + 8 < right - 8 and current.shape[:2] == (720, 1280)):
        return None
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    edges = np.median(cv2.Sobel(gray[625:695], cv2.CV_32F, 1, 0, ksize=3), axis=0)
    left_edge, right_edge = float(edges[left]), float(edges[right - 1])
    bottom_contrast = float(gray[bottom - 4:bottom, left + 8:right - 8].mean() -
                            gray[bottom + 1:bottom + 3, left + 8:right - 8].mean())
    if left_edge < 250 or right_edge > -150 or bottom_contrast < 40:
        return None
    return {"left_edge": left_edge, "right_edge": right_edge,
            "bottom_contrast": bottom_contrast}


def _closed_dragon_card_border_18_647_2(current: Any,
                                         bbox: tuple[int, int, int, int]) -> dict | None:
    """Recognize the complete rounded gray dragon card in this client layout.

    This only proposes a header ROI. Fresh x0 OCR, saved portrait matching and
    versioned source identity remain separate requirements in the caller.
    """
    import cv2
    import numpy as np

    left, top, right, bottom = bbox
    if (current.shape[:2] != (720, 1280) or not (85 <= right - left <= 95) or
            not (590 <= top <= 600 and 705 <= bottom <= 715) or
            left < 2 or right >= 1278):
        return None
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    side = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    # Seven independent vertical bands must carry both opposite card edges.
    left_bands = [float(np.median(band)) for band in np.array_split(side[625:695, left], 7)]
    right_bands = [float(np.median(band)) for band in np.array_split(side[625:695, right - 1], 7)]
    if min(left_bands) < 175 or max(right_bands) > -160:
        return None
    left_edge, right_edge = float(np.median(left_bands)), float(np.median(right_bands))
    # The side edges turn inward at both rounded ends; a straight neighboring
    # seam or a rectangular overlay is insufficient.
    for y in (top + 5, bottom - 7):
        if (side[y, left] > .7 * left_edge or
                side[y, right - 1] < .75 * right_edge):
            return None
    lower_inside = gray[bottom - 4:bottom, left + 8:right - 8].mean(axis=0)
    lower_outside = gray[bottom + 1:bottom + 3, left + 8:right - 8].mean(axis=0)
    bottom_bands = [float(np.mean(band)) for band in
                    np.array_split(lower_inside - lower_outside, 5)]
    if min(bottom_bands) < 33:
        return None
    # The lower rounded edge shifts inward by a pixel between rows on this
    # layout. A single x+3 sample at bottom-7 can land beyond that edge. Require
    # a four-connected rim within fixed card bounds instead: both full side
    # traces must join the inner bottom center. This does not prove the top rim.
    from collections import deque

    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rim = np.zeros((bottom - 623, right - left), dtype=np.bool_)
    for y in range(625, bottom):
        rim[y - 624, :9] |= side[y, left:left + 9] >= 40
        rim[y - 624, -9:] |= side[y, right - 9:right] <= -40
    for y in range(bottom - 4, bottom + 1):
        rim[y - 624, 5:-5] |= vertical[y, left + 5:right - 5] <= -40
    seed = (650 - 624, 0)
    opposite = (650 - 624, right - left - 1)
    center = (bottom - 1 - 624, (right - left) // 2)
    if not (rim[seed] and rim[opposite] and rim[center]):
        return None
    connected = {seed}
    pending = deque([seed])
    while pending:
        y, x = pending.popleft()
        for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            row, col = point
            if (0 <= row < rim.shape[0] and 0 <= col < rim.shape[1] and
                    rim[point] and point not in connected):
                connected.add(point)
                pending.append(point)
    if opposite not in connected or center not in connected:
        return None
    for y in range(625, bottom):
        row = y - 624
        if (not any((row, x) in connected for x in range(9)) or
                not any((row, x) in connected for x in range(right - left - 9, right - left))):
            return None
    return {"method": "18.647.2_dragon_rounded_closed_card",
            "left_edge": left_edge, "right_edge": right_edge,
            "bottom_contrast": float(np.mean(bottom_bands)),
            "left_side_bands": left_bands, "right_side_bands": right_bands,
            "bottom_bands": bottom_bands,
            "lower_outline": {"connectivity": 4, "gradient_min": 40,
                              "bottom_gradient_max": -40,
                              "connected_pixels": len(connected)}}


def _post_troop_reading(session: Any, post_path: Any, source_path: str,
                        point: tuple[int, int], bbox: tuple[int, int, int, int], *,
                        unit_id: str | None = None) -> dict:
    """Use a frozen card portrait to bind current-frame count OCR to its identity."""
    import cv2
    import numpy as np
    from .battle_vision import BattleObserver
    from .images import read_frame

    source = cv2.resize(read_frame(source_path), (1280, 720), interpolation=cv2.INTER_AREA)
    current = cv2.resize(read_frame(post_path), (1280, 720), interpolation=cv2.INTER_AREA)
    left, top, right, bottom = bbox
    patch = source[top + 38:min(top + 78, bottom), left + 20:right - 15]
    if patch.size == 0:
        return {}
    observer = BattleObserver(session.recognizer)
    observer.cards[round(point[0])] = (bbox, cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).copy())
    reading = observer.count(post_path, current, {"point": list(point), "bbox": list(bbox)})
    def conflict(values) -> dict:
        return {"conflict": True, "reason": "strong_count_conflict",
                "counts": sorted(set(values)), "frame": str(post_path), "slot_bbox": list(bbox)}

    if isinstance(reading, dict) and reading.get("reason") == "strong_count_conflict":
        return conflict(reading.get("counts", ()))
    snapshot = getattr(session, "last_snapshot", None)
    current_ocr = (snapshot.observations.get("ocr", []) if snapshot is not None and
                   str(snapshot.screenshot_path) == str(post_path) else [])
    candidates = []
    strong_full_counts = set()
    for item in current_ocr:
        if not isinstance(item, dict) or not isinstance(item.get("bbox"), (list, tuple)):
            continue
        box = item["bbox"]
        count = _COUNT.fullmatch(str(item.get("text", "")).strip())
        if (len(box) == 4 and all(type(v) is int for v in box) and
                left <= box[0] < box[2] <= right + 3 and
                top - 5 <= box[1] < box[3] <= top + 38 and count is not None):
            candidates.append(item)
            confidence = item.get("confidence")
            if (type(confidence) in (int, float) and math.isfinite(confidence) and confidence >= .9):
                strong_full_counts.add(int(count[1]))
    if len(strong_full_counts) > 1:
        return conflict(strong_full_counts)
    if reading:
        if unit_id == "dragon":
            # The neighboring gray card can itself produce a strong x0 fast
            # read. Bind that read to this version's *source* dragon portrait
            # before accepting it as the frozen dragon slot.
            if (getattr(session.recognizer, "client_version", None) != _CLIENT_VERSION or
                    getattr(session, "client_version", None) != _CLIENT_VERSION):
                return {}
            from .unit_catalog import recognize_card_identity

            identity = recognize_card_identity(source, bbox, "troop", surface="battle",
                                               client_version=_CLIENT_VERSION)
            if (identity.get("unit_id") != unit_id or
                    identity.get("version_verified") is not True or
                    identity.get("template_client_version") != _CLIENT_VERSION):
                return {}
        if strong_full_counts and strong_full_counts != {reading["count"]}:
            return conflict(strong_full_counts | {reading["count"]})
        return reading
    # Full-frame OCR can locate a gray card's small x0 while the broad header
    # crop is uncertain. Its low-confidence number is *only* a locator; truth
    # still requires a new >=.9 line read and an independent current portrait.
    if snapshot is None or str(snapshot.screenshot_path) != str(post_path):
        return {}
    if len(candidates) > 1:
        return {}
    if candidates:
        candidate = candidates[0]
        locator_source = "current_frame_ocr_box"
    else:
        # A preceding post-burst frame may supply only *geometry*. Read the
        # quantity afresh below; the previous frame's number is never reused.
        hint = getattr(session, "_frozen_count_roi_hints", {}).get(tuple(bbox))
        if isinstance(hint, (list, tuple)) and len(hint) == 4:
            candidate = {"bbox": list(hint), "text": None, "confidence": None}
            locator_source = "prior_post_frame_roi_geometry"
        else:
            # The sampled current-client cards have a right-aligned quantity
            # header. An independently bound card may locate that header even
            # when full-frame OCR returns no numeric box at all. This geometry
            # proposes only an ROI; fresh native OCR below proves the value.
            if (getattr(session.recognizer, "client_version", None) != _CLIENT_VERSION or
                    getattr(session, "client_version", None) != _CLIENT_VERSION or
                    getattr(snapshot, "scene", None) not in {"enemy_village", "battle"} or
                    not (80 <= right - left <= 105 and 590 <= top <= 605 and 705 <= bottom <= 715)):
                return {}
            from .unit_catalog import recognize_card_identity

            identity = recognize_card_identity(source, bbox, "troop", surface="battle",
                                               client_version=_CLIENT_VERSION)
            if (not isinstance(unit_id, str) or identity.get("unit_id") != unit_id or
                    identity.get("version_verified") is not True or
                    identity.get("template_client_version") != _CLIENT_VERSION):
                return {}
            candidate = {"bbox": [right - 45, top - 3, right - 2, top + 28],
                         "text": None, "confidence": None}
            locator_source = "current_frame_closed_card_header_geometry"
    region = cv2.cvtColor(current[max(0, top + 15):min(720, bottom - 20),
                                      max(0, left - 12):min(1280, right + 12)], cv2.COLOR_BGR2GRAY)
    portrait = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    matches = []
    for scale in (1., 1.05, 1.1):
        template = cv2.resize(portrait, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if template.shape[0] <= region.shape[0] and template.shape[1] <= region.shape[1]:
            _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED))
            if math.isfinite(score):
                matches.append((score, scale, x, y))
    if not matches:
        return {}
    score, scale, x, y = max(matches)
    estimated_left = round(max(0, left - 12) + x - 20 * scale)
    estimated_top = round(max(0, top + 15) + y - 38 * scale)
    if score < .92 or abs(estimated_left - left) > 3 or abs(estimated_top - top) > 3:
        return {}
    border = None
    if locator_source == "current_frame_closed_card_header_geometry":
        border = _closed_card_border(current, bbox)
        if border is None and unit_id == "dragon":
            border = _closed_dragon_card_border_18_647_2(current, bbox)
        if border is None:
            return {}
    from .locator import scale_box

    original = read_frame(post_path)
    reader = getattr(session.recognizer.provider, "recognize_line_image", None)
    if not callable(reader):
        return {}
    if locator_source == "current_frame_closed_card_header_geometry":
        # All four renderings are of this card on this frame. Never let a
        # thresholded x0 override a contradictory strong raw or padded read.
        tight = tuple(candidate["bbox"])
        padded = (tight[0] - 2, tight[1] - 2, min(right + 3, tight[2] + 2),
                  min(top + 38, tight[3] + 2))
        strong_counts = set(strong_full_counts)
        proofs = []
        for proof_box in (tight, padded):
            a, b, c, d = scale_box(proof_box, from_resolution=(1280, 720),
                                  to_resolution=(original.shape[1], original.shape[0]))
            crop = original[b:d, a:c]
            if crop.size == 0:
                continue
            binary = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 120, 255,
                                   cv2.THRESH_BINARY)[1]
            for method, view in (("raw", crop), ("threshold", binary)):
                reads = reader(view)
                for item in reads:
                    count = _COUNT.fullmatch(str(item.text).strip())
                    confidence = item.confidence
                    if (count is not None and type(confidence) in (int, float) and
                            math.isfinite(confidence) and confidence >= .9):
                        strong_counts.add(int(count[1]))
                if len(strong_counts) > 1:
                    return conflict(strong_counts)
                if len(reads) == 1:
                    item = reads[0]
                    count = _COUNT.fullmatch(str(item.text).strip())
                    if (count is not None and type(item.confidence) in (int, float) and
                            math.isfinite(item.confidence) and item.confidence >= .9):
                        proofs.append((item.confidence, proof_box, item, method, int(count[1])))
        if not proofs:
            return {}
        confidence, proof_box, item, method, count = max(proofs, key=lambda row: row[0])
        return {"count": count, "confidence": confidence, "frame": str(post_path),
                "slot_bbox": list(bbox), "roi": list(proof_box),
                "readings": [{"text": item.text, "confidence": confidence,
                              "bbox": list(proof_box),
                              "source": f"current_closed_card_header_{method}_line"}],
                "portrait_score": score, "portrait_scale": scale,
                "portrait_source": "current_frame_saved_battle_portrait",
                "source_portrait_identity": identity, "current_card_border": border,
                "locator": {"source": locator_source, "bbox": candidate["bbox"]}}
    locator_count = _COUNT.fullmatch(str(candidate["text"]).strip()) if candidate["text"] is not None else None
    # The tight OCR box can trim the dark outline of a small gray x0. One
    # fixed two-pixel extension stays inside the independently bound header;
    # both reads use this frame and the same existing >=.9 proof threshold.
    tight = tuple(candidate["bbox"])
    padded = (max(left, tight[0] - 2), max(top - 5, tight[1] - 2),
              min(right + 3, tight[2] + 2), min(top + 38, tight[3] + 2))
    proofs = []
    strong_counts = set()
    for proof_box in (tight, padded):
        a, b, c, d = scale_box(proof_box, from_resolution=(1280, 720),
                              to_resolution=(original.shape[1], original.shape[0]))
        crop = original[b:d, a:c]
        if crop.size == 0:
            continue
        binary = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY)[1]
        reads = reader(binary)
        for item in reads:
            number = _COUNT.fullmatch(str(item.text).strip())
            confidence = item.confidence
            if (number is not None and type(confidence) in (int, float) and
                    math.isfinite(confidence) and confidence >= .9):
                strong_counts.add(int(number[1]))
        if len(strong_counts) > 1:
            return conflict(strong_counts)  # no crop may override a strong same-frame read
        if len(reads) != 1:
            continue
        narrow = reads[0]
        count = _COUNT.fullmatch(str(narrow.text).strip())
        if (count is None or type(narrow.confidence) not in (int, float) or
                not math.isfinite(narrow.confidence) or narrow.confidence < .9):
            continue
        proofs.append((proof_box, narrow, count))
    if len({item[2][1] for item in proofs}) > 1:
        return conflict(int(item[2][1]) for item in proofs)
    if (not proofs or (locator_source == "current_frame_ocr_box" and
                       (locator_count is None or proofs[0][2][1] != locator_count[1]))):
        if (proofs and locator_count is not None and strong_full_counts and
                int(proofs[0][2][1]) != int(locator_count[1])):
            return conflict(strong_full_counts | {int(proofs[0][2][1])})
        return {}
    proof_box, narrow, count = proofs[0]
    return {"count": int(count[1]), "confidence": narrow.confidence,
            "frame": str(post_path), "slot_bbox": list(bbox), "roi": list(proof_box),
            "readings": [{"text": narrow.text, "confidence": narrow.confidence,
                          "bbox": list(proof_box), "source": "current_ocr_box_threshold_line"}],
            "portrait_score": score, "portrait_scale": scale,
            "portrait_source": "current_frame_saved_battle_portrait",
            "locator": {"source": locator_source, "text": candidate["text"], "bbox": candidate["bbox"],
                        "confidence": candidate.get("confidence")}}


def freeze_model_free_line(session: Any, selected: SceneSnapshot, *,
                           toolbar_before: SceneSnapshot, terrain: Sequence[dict],
                           army_manifest: dict, minimum_zoom: dict,
                           battle_id: str, latest_start: float, burst_limit_sec: float,
                           hero_ids: Sequence[str] = (), hero_manifest: dict | None = None,
                            line_tactics: str = "single_edge",
                           clock: Callable[[], float] = time.monotonic) -> FrozenLinePlan:
    """Freeze the existing line roster, optionally with a bounded archer filler row."""
    if getattr(getattr(session.config, "vision_agent", None), "enabled", False):
        raise FlowError("Model-free line cannot replace vision-mode model release")
    if line_tactics not in {"single_edge", "two_edge_split", "two_edge_furnace_mixed_probes"}:
        raise FlowError("Frozen line tactics are unsupported")
    if getattr(session, "client_version", None) != _CLIENT_VERSION or not isinstance(battle_id, str) or not battle_id:
        raise FlowError("Frozen line requires a current-version battle identity")
    if selected is not session.last_snapshot:
        raise FlowError("Frozen line must use the last observed selected frame")
    frame = FramePacket.from_snapshot(selected, session)
    before = FramePacket.from_snapshot(toolbar_before, session)
    now = clock()
    if (frame.scene not in {"enemy_village", "battle"} or not math.isfinite(frame.confidence) or frame.confidence < .8 or
            before.scene not in {"enemy_village", "battle"} or not math.isfinite(before.confidence) or before.confidence < .8 or
            before.frame_id == frame.frame_id or before.captured_at >= frame.captured_at or
            frame.captured_at - before.captured_at > 30 or
            before.display_id != frame.display_id or before.resolution != frame.resolution or
            before.source_resolution != frame.source_resolution or
            type(latest_start) not in (int, float) or not math.isfinite(latest_start) or
            not now < latest_start <= frame.captured_at + 10 or
            type(burst_limit_sec) not in (int, float) or not math.isfinite(burst_limit_sec) or
            not 0 < burst_limit_sec <= 20):
        raise FlowError("Frozen line frame/display/time or burst budget is invalid")
    motion = minimum_zoom.get("motion") if isinstance(minimum_zoom, dict) else None
    if (not isinstance(minimum_zoom, dict) or minimum_zoom.get("verified") is not True or
            minimum_zoom.get("frame") != before.frame_id or not isinstance(motion, dict) or
            motion.get("stationary") is not True or type(motion.get("matched_points")) is not int or
            motion["matched_points"] < 40 or type(motion.get("inliers")) is not int or
            motion["inliers"] < 40 or motion["inliers"] / motion["matched_points"] < .8 or
            not isinstance(motion.get("affine"), list) or len(motion["affine"]) != 2 or
            any(not isinstance(row, list) or len(row) != 3 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in row)
                for row in motion["affine"])):
        raise FlowError("Frozen line minimum-zoom evidence is unverified")
    affine = motion["affine"]
    if (abs(affine[0][0] - 1) > .02 or abs(affine[0][1]) > .02 or
            math.hypot(affine[0][2], affine[1][2]) >= 8):
        raise FlowError("Frozen line minimum-zoom camera was moving")
    manifest = army_manifest
    troops = manifest.get("troops") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get("complete") is not True or
            manifest.get("identity_complete") is not True or manifest.get("unknowns") or
            not isinstance(troops, list) or len(troops) not in {4, 5} or
            any(not isinstance(row, dict) or not isinstance(row.get("unit_id"), str) or
                row.get("source") != "army" or type(row.get("count")) is not int or row["count"] <= 0
                for row in troops)):
        raise FlowError("Frozen line requires four troop rows or one bounded archer filler row")
    expected = {(row["unit_id"], row["source"]): row["count"] for row in troops}
    if len(expected) != len(troops):
        raise FlowError("Frozen line army manifest repeats a troop identity")
    if line_tactics == "two_edge_furnace_mixed_probes" and expected != {
            ("electro_dragon", "army"): 10, ("dragon", "army"): 1,
            ("furnace", "army"): 1, ("balloon", "army"): 1}:
        raise FlowError("Mixed furnace probes require exact current army counts")
    if len(troops) == 5:
        expected_five_ids = {
            (unit_id, "army") for unit_id in _BASE_LINE_TROOP_IDS | {_ARCHER_FILLER_UNIT_ID}
        }
        archer_count = expected.get((_ARCHER_FILLER_UNIT_ID, "army"))
        if (set(expected) != expected_five_ids or type(archer_count) is not int or
                not 1 <= archer_count <= 3):
            raise FlowError("Frozen line five-card support requires the four existing troop identities plus one to three archers")
    current_cards = _cards(selected, client_version=_CLIENT_VERSION, expected=expected)
    previous_cards = _cards(toolbar_before, client_version=_CLIENT_VERSION, expected=expected)
    current_troops = sorted((card for card in current_cards if card["kind"] == "troop"),
                            key=lambda card: card["point"][0])
    if len(current_troops) != len(expected) or {(card["unit_id"], card["source"]) for card in current_troops} != set(expected):
        raise FlowError("Frozen line toolbar does not contain exactly the manifest troops")
    # Stable card positions are established by two real, distinct source frames.
    previous = {(card["kind"], card["unit_id"], card["source"]): card for card in previous_cards}
    for card in current_troops:
        prior = previous.get((card["kind"], card["unit_id"], card["source"]))
        if (prior is None or prior["count"] != card["count"] or
                any(abs(prior["point"][axis] - card["point"][axis]) > 4 for axis in (0, 1))):
            raise FlowError("Frozen line toolbar geometry/count changed across source frames")
    if len([card for card in previous_cards if card["kind"] == "troop"]) != len(expected):
        raise FlowError("Frozen line previous toolbar lacks all manifest troops")
    from .battle_vision import selected_card_bbox
    if selected_card_bbox(selected.screenshot_path, current_troops[0]["bbox"], frame.resolution) is None:
        raise FlowError("Frozen line first troop is not selected in the current frame")
    if not isinstance(terrain, (list, tuple)) or len(terrain) != 4:
        raise FlowError("Frozen line needs four current red-boundary edge anchors")
    for edge in (0, 1):
        anchors = [row for row in terrain if isinstance(row, dict) and row.get("edge") == edge]
        if len(anchors) != 2 or anchors[0].get("point") == anchors[1].get("point"):
            raise FlowError("Frozen line lacks two distinct anchors per edge")
    for row in terrain:
        evidence = row.get("evidence") if isinstance(row, dict) else None
        if (not isinstance(evidence, dict) or evidence.get("method") != "current_red_boundary_offset" or
                evidence.get("frame") != frame.frame_id or evidence.get("horizontal_clearance") != 32 or
                any(type(evidence.get(name)) not in (int, float) or
                    not math.isfinite(evidence[name]) for name in ("slope", "intercept")) or
                not .68 <= abs(evidence["slope"]) <= .82):
            raise FlowError("Frozen line terrain is not bound to the selected source frame")
        _point(row.get("point"), *frame.resolution)
    selected_heroes = []
    if len(set(hero_ids)) != len(hero_ids) or any(not isinstance(item, str) or not item for item in hero_ids):
        raise FlowError("Frozen line hero identities are ambiguous")
    hero_frame_hash = None
    if hero_ids or hero_manifest is not None:
        rows = hero_manifest.get("cards") if isinstance(hero_manifest, dict) else None
        if (not isinstance(hero_manifest, dict) or hero_manifest.get("complete") is not True or
                hero_manifest.get("unknowns") or not isinstance(rows, list) or
                {row.get("unit_id") for row in rows if isinstance(row, dict)} != set(hero_ids) or
                len(rows) != len(hero_ids) or not isinstance(hero_manifest.get("frame"), str) or
                Path(hero_manifest["frame"]).parent != Path(toolbar_before.screenshot_path).parent):
            raise FlowError("Frozen line heroes differ from independent complete hero manifest")
        hero_frame_hash = _file_sha256(hero_manifest["frame"])
        for row in rows:
            identity = (row.get("evidence") or {}).get("identity") or {}
            score = identity.get("confidence")
            row_score = row.get("confidence")
            if (row.get("kind") != "hero" or row.get("source") != "army" or
                    type(row_score) not in (int, float) or not math.isfinite(row_score) or row_score < .9 or
                    type(score) not in (int, float) or
                    not math.isfinite(score) or score < .9 or
                    identity.get("unit_id") != row.get("unit_id") or
                    identity.get("version_verified") is not True or
                    identity.get("template_client_version") != _CLIENT_VERSION):
                raise FlowError("Frozen line independent hero identity is unverified")
    # The selected fast path can have only refreshed troop slots. Check heroes
    # directly on its current pixels at the independently located prior cards.
    if hero_ids:
        import cv2
        from .images import read_frame
        from .unit_catalog import recognize_card_identity
        image = cv2.resize(read_frame(selected.screenshot_path), (1280, 720), interpolation=cv2.INTER_AREA)
    for hero_id in hero_ids:
        prior = previous.get(("hero", hero_id, "army"))
        if prior is None:
            raise FlowError("Frozen line hero lacks current-version card identity")
        identity = recognize_card_identity(image, prior["bbox"], "hero", surface="battle",
                                           client_version=_CLIENT_VERSION)
        score = identity.get("confidence")
        if (identity.get("unit_id") != hero_id or identity.get("version_verified") is not True or
                type(score) not in (int, float) or not math.isfinite(score) or score < .94):
            raise FlowError("Frozen line hero lacks current-frame identity")
        selected_heroes.append(prior)
    if line_tactics == "single_edge":
        planner = EdragLineStrategy()
    elif line_tactics == "two_edge_split":
        planner = TwoEdgeSplitStrategy()
    else:
        from .strategies import TwoEdgeFurnaceStrategy
        planner = TwoEdgeFurnaceStrategy()
    placement = planner.build_plan(BattleContext(tuple(current_troops),
                                                 tuple(selected_heroes), tuple(terrain)))
    actions = []
    troop_steps = placement.troops
    if len(current_troops) == 5:
        # Keep the bounded filler last even when it is the preselected leftmost card.
        troop_steps = tuple(sorted(
            troop_steps, key=lambda step: step.card["unit_id"] == _ARCHER_FILLER_UNIT_ID))
    selected_troop = (current_troops[0]["unit_id"], current_troops[0]["source"])
    for index, step in enumerate(troop_steps):
        card = step.card
        card_identity = (card["unit_id"], card["source"])
        if card_identity != selected_troop:
            actions.append(PreparedInput(f"select-troop-{index}", "select", card["point"],
                                         card["unit_id"], card["source"]))
        selected_troop = card_identity
        for offset, point in enumerate(step.points):
            actions.append(PreparedInput(f"troop-{index}-{offset}", "deploy", _point(point, *frame.resolution),
                                         card["unit_id"], card["source"], .12))
    for index, step in enumerate(placement.heroes):
        card = step.card
        actions.append(PreparedInput(f"select-hero-{index}", "select", card["point"],
                                     card["unit_id"], card["source"]))
        actions.append(PreparedInput(f"hero-{index}", "deploy", _point(step.points[0], *frame.resolution),
                                     card["unit_id"], card["source"], .12))
    budgets = tuple((card["unit_id"], card["source"], card["count"]) for card in current_troops)
    troop_cards = tuple((card["unit_id"], card["source"], card["count"],
                         card["point"], card["bbox"]) for card in current_troops)
    hero_cards = tuple((card["unit_id"], card["bbox"]) for card in selected_heroes)
    # Use the same complete-batch speed gate as the model-driven path. A
    # caller's larger timeout must not silently relax the acceptance budget.
    burst_limit_sec = min(burst_limit_sec, 15., 1. + .25 * len(actions))
    if sum(action.delay_after_sec for action in actions) >= burst_limit_sec:
        raise FlowError("Frozen line fixed delays exceed the complete-batch budget")
    plan = FrozenLinePlan(uuid4().hex, battle_id, frame,
                          _file_sha256(frame.screenshot_path), before.frame_id,
                          _file_sha256(before.screenshot_path),
                          _CLIENT_VERSION, now,
                          latest_start, burst_limit_sec, _digest(current_cards), _digest(terrain),
                          tuple(actions), budgets, troop_cards, tuple(hero_ids), hero_cards,
                          _digest(hero_manifest) if hero_manifest is not None else None,
                          hero_manifest["frame"] if hero_manifest is not None else None, hero_frame_hash,
                          _digest(minimum_zoom), line_tactics, "")
    return FrozenLinePlan(**{**asdict(plan), "frame": frame, "actions": plan.actions,
                             "troop_budgets": plan.troop_budgets, "troop_cards": plan.troop_cards,
                             "hero_ids": plan.hero_ids, "hero_cards": plan.hero_cards,
                             "plan_sha256": _plan_digest(plan)})


def execute_frozen_line(session: Any, plan: FrozenLinePlan, *,
                        clock: Callable[[], float] = time.monotonic,
                        sleep: Callable[[float], None] = time.sleep) -> SceneSnapshot:
    """One attempt: no observation/recognition during input; no retry or slow fallback."""
    if getattr(getattr(session.config, "vision_agent", None), "enabled", False):
        raise FlowError("Model-free line cannot replace vision-mode model release")
    if not isinstance(plan, FrozenLinePlan) or _plan_digest(plan) != plan.plan_sha256:
        raise FlowError("Frozen line plan digest does not match")
    if (session.last_snapshot is None or str(session.last_snapshot.screenshot_path) != plan.frame.frame_id or
            session.client_version != plan.client_version or
            session.capture.input_display_id != plan.frame.display_id or
            tuple(session.config.game.baseline_resolution) != plan.frame.resolution or
            tuple(session.context.screen_resolution) != plan.frame.source_resolution or
            clock() >= plan.latest_start):
        raise FlowError("Frozen line permission expired or display/frame changed")
    if _file_sha256(plan.frame.screenshot_path) != plan.frame_sha256:
        raise FlowError("Frozen line source pixels changed after plan freeze")
    if _file_sha256(plan.toolbar_before_frame_id) != plan.toolbar_before_sha256:
        raise FlowError("Frozen line toolbar evidence changed after plan freeze")
    if (plan.hero_manifest_frame_id is not None and
            _file_sha256(plan.hero_manifest_frame_id) != plan.hero_manifest_frame_sha256):
        raise FlowError("Frozen line independent hero manifest frame changed after plan freeze")
    used = getattr(session, "_used_frozen_frames", None)
    if used is None:
        used = set()
        session._used_frozen_frames = used
    if plan.frame.frame_id in used:
        raise FlowError("Frozen line source frame was already attempted")
    used.add(plan.frame.frame_id)  # consumed before the first uncertain transport call
    start = clock()
    deadline = min(start + plan.burst_limit_sec, session.deadline, session.task_deadline)
    receipts = []
    receipt = {"mode": "model_free_line_continuous", "plan_id": plan.plan_id,
               "line_tactics": plan.line_tactics,
               "plan_sha256": plan.plan_sha256, "battle_id": plan.battle_id,
               "frame": plan.frame.frame_id, "input_count": len(plan.actions),
               "scope": "troops_and_heroes" if plan.hero_ids else "troops_only",
               "excluded_kinds": ["spell", "siege"] + ([] if plan.hero_ids else ["hero"]),
               "actions": receipts, "completed": False, "verified": False,
               "input_sent": False, "issued_placements": 0,
               "attempted_placements": 0, "deployed_units": 0,
               "burst_limit_sec": plan.burst_limit_sec, "burst_pass": False,
               "consumption_events": [],
               "evidence": [plan.frame.frame_id], "error": None}
    session.frozen_line_receipt = receipt
    try:
        for action in plan.actions:
            session.check_deadline()
            remaining = deadline - clock()
            if remaining <= 0:
                raise FlowError("Frozen line burst deadline exceeded")
            if action.kind == "deploy":
                receipt["attempted_placements"] += 1
            session._active_frozen_plan = (plan.plan_id, deadline, action.point)
            try:
                session.frozen_battle_tap(action.point,
                                          timeout_sec=min(remaining, session.config.runtime.step_timeout_sec),
                                          reason=f"model-free line {plan.plan_id} {action.action_id}",
                                          plan_id=plan.plan_id)
            except BaseException as exc:
                receipts.append({"action_id": action.action_id, "status": "uncertain", "reason": str(exc)})
                raise
            finally:
                session._active_frozen_plan = None
            receipts.append({"action_id": action.action_id, "status": "input_sent",
                             "done_at": clock(), "kind": action.kind,
                             "unit_id": action.unit_id, "point": action.point})
            receipt["issued_placements"] += int(action.kind == "deploy")
            if clock() >= deadline:
                raise FlowError("Frozen line deadline exceeded after input")
            if action.delay_after_sec:
                if action.delay_after_sec >= deadline - clock():
                    raise FlowError("Frozen line fixed delay exceeds deadline")
                sleep(action.delay_after_sec)
    except BaseException as exc:
        receipt["error"] = str(exc)
        session._active_frozen_plan = None
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise DeploymentError(str(exc), partial_receipt=receipt) from exc
    finally:
        session._active_frozen_plan = None
    receipt["burst_elapsed_sec"] = clock() - start
    receipt["input_sent"] = len(receipts) == len(plan.actions) and receipt["burst_elapsed_sec"] <= plan.burst_limit_sec
    receipt["burst_pass"] = receipt["input_sent"]
    return _verify_frozen_line_post(session, plan, receipt, receipts[-1]["done_at"], sleep=sleep)


def _verify_frozen_line_post(session: Any, plan: FrozenLinePlan, receipt: dict,
                             after_time: float, *, sleep: Callable[[float], None]) -> SceneSnapshot:
    """Read one complete original-roster post state after a separately authorized batch."""
    session._frozen_count_roi_hints = {}
    try:
        post = session.observe("frozen-line-post")
    except Exception as exc:
        receipt["error"] = f"Post-burst observation failed: {exc}"
        raise DeploymentError(receipt["error"], partial_receipt=receipt) from exc
    previous_path = None
    for attempt in range(3):
        receipt["evidence"].append(str(post.screenshot_path))
        observed_at = post.observations.get("observed_at_monotonic")
        post_source = post.observations.get("source_resolution")
        post_baseline = post.observations.get("baseline_resolution")
        if (post.scene not in {"enemy_village", "battle"} or
                type(post.confidence) not in (int, float) or not math.isfinite(post.confidence) or
                post.confidence < .8 or type(observed_at) not in (int, float) or
                not math.isfinite(observed_at) or observed_at <= after_time or
                str(post.screenshot_path) == previous_path or
                not isinstance(post_source, (list, tuple)) or tuple(post_source) != plan.frame.source_resolution or
                not isinstance(post_baseline, (list, tuple)) or tuple(post_baseline) != plan.frame.resolution):
            receipt["error"] = "Fresh post-burst battle frame unavailable"
            raise DeploymentError(receipt["error"], partial_receipt=receipt)
        after_time, previous_path = observed_at, str(post.screenshot_path)
        battle = post.observations.get("battle")
        slots = battle.get("slots") if isinstance(battle, dict) else []
        if not isinstance(slots, list):
            slots = []
        troop_results = []
        count_conflict = False
        for unit_id, source, initial, point, bbox in plan.troop_cards:
            # Spent cards can disappear from full-frame slots. Rebind their
            # saved portrait and read the quantity in this particular frame.
            try:
                reading = _post_troop_reading(session, post.screenshot_path, plan.toolbar_before_frame_id,
                                              point, bbox, unit_id=unit_id)
            except Exception as exc:
                reading = {"error": str(exc)}
            conflicting = [row for row in slots if isinstance(row, dict) and
                           (row.get("unit_id") == unit_id or row.get("bbox") == list(bbox)) and
                           type(row.get("count")) is int and row["count"] > 0]
            count_conflict |= bool(conflicting) or reading.get("conflict") is True
            score = reading.get("confidence")
            portrait = reading.get("portrait_score")
            reads = reading.get("readings")
            count_conflict |= (type(reading.get("count")) is int and reading["count"] > 0 and
                reading.get("frame") == str(post.screenshot_path) and reading.get("slot_bbox") == list(bbox) and
                type(score) in (int, float) and math.isfinite(score) and score >= .9 and
                type(portrait) in (int, float) and math.isfinite(portrait) and portrait >= .92 and
                reading.get("portrait_source") == "current_frame_saved_battle_portrait")
            explicit = [item for item in reads if isinstance(item, dict) and
                        type(item.get("confidence")) in (int, float) and item["confidence"] >= .9 and
                        (match := _COUNT.fullmatch(str(item.get("text", "")).strip())) is not None and int(match[1]) == 0] if isinstance(reads, list) else []
            valid = (reading.get("count") == 0 and reading.get("frame") == str(post.screenshot_path) and
                     reading.get("slot_bbox") == list(bbox) and not conflicting and
                     type(score) in (int, float) and math.isfinite(score) and score >= .9 and
                     type(portrait) in (int, float) and math.isfinite(portrait) and portrait >= .92 and
                     reading.get("portrait_source") == "current_frame_saved_battle_portrait" and
                     bool(explicit))
            if valid and isinstance(reading.get("roi"), list):
                session._frozen_count_roi_hints[tuple(bbox)] = reading["roi"]
            troop_results.append({"unit_id": unit_id, "initial": initial, "post_count": 0 if valid else None,
                                  "verified": valid, "reading": reading})
        receipt["troops"] = troop_results
        receipt["consumption_events"] = [
            {"unit_id": item["unit_id"], "before_count": item["initial"],
             "after_count": item["post_count"], "after_frame": str(post.screenshot_path),
             "verified": item["verified"]} for item in troop_results]
        hero_results = []
        if plan.hero_ids:
            from .hero_state import locate_current_deployed_hero_card, recognize_hero_state
            used_boxes = set()
            for hero_id, _prior_bbox in plan.hero_cards:
                locator = locate_current_deployed_hero_card(post.screenshot_path, hero_id,
                                                           client_version=plan.client_version)
                state = {"deployed": None, "frame": str(post.screenshot_path),
                         "evidence": {"reason": "current_hero_card_unlocated"}}
                box = tuple(locator["bbox"]) if locator is not None else None
                if box is not None and box not in used_boxes:
                    used_boxes.add(box)
                    state = recognize_hero_state(post.screenshot_path, box,
                                                 baseline_resolution=(1280, 720),
                                                 unit_id=hero_id, client_version=plan.client_version)
                identity = (state.get("evidence") or {}).get("portrait_identity") or {}
                verified = (locator is not None and box is not None and
                            locator.get("frame") == str(post.screenshot_path) and
                            locator.get("template_client_version") == plan.client_version and
                            state.get("deployed") is True and state.get("frame") == str(post.screenshot_path) and
                            identity.get("unit_id") == hero_id and identity.get("version_verified") is True and
                            identity.get("template_client_version") == plan.client_version)
                hero_results.append({"unit_id": hero_id, "verified": verified,
                                     "card_locator": locator, "state": state})
        receipt["heroes"] = hero_results
        receipt["verified"] = receipt["completed"] = (receipt["input_sent"] and
            all(item["verified"] for item in troop_results + hero_results))
        hero_frame_details = []
        for item in hero_results:
            state = item["state"]
            hero_evidence = state.get("evidence") or {}
            identity = hero_evidence.get("portrait_identity") or {}
            health = hero_evidence.get("health_bar") or hero_evidence.get("yellow_health_bar") or {}
            hero_frame_details.append({"unit_id": item["unit_id"], "verified": item["verified"],
                                       "locator_bbox": (item["card_locator"] or {}).get("bbox"),
                                       "state": state.get("state"), "deployed": state.get("deployed"),
                                       "identity": identity.get("unit_id"),
                                       "health_method": health.get("method"),
                                       "reason": (hero_evidence.get("reason") or hero_evidence.get("ability_phase_reason"))
                                       if not item["verified"] else None})
        receipt.setdefault("verification_frames", []).append(
            {"frame": str(post.screenshot_path), "troops": [r["verified"] for r in troop_results],
             "heroes": [r["verified"] for r in hero_results], "hero_details": hero_frame_details})
        if receipt["verified"] or attempt == 2 or count_conflict:
            break
        try:
            session.check_deadline()
            sleep(.5)  # only after the input batch; let selected-card animation settle
            session.check_deadline()
            post = session.observe("frozen-line-post-recheck")
        except Exception as exc:
            receipt["error"] = f"Post-burst reobservation failed: {exc}"
            raise DeploymentError(receipt["error"], partial_receipt=receipt) from exc
    if receipt["verified"]:
        receipt["deployed_units"] = sum(initial for _unit_id, _source, initial in plan.troop_budgets)
    post.observations["deployment"] = receipt
    if not receipt["completed"]:
        receipt["error"] = "Frozen line troop or hero consumption is unverified"
        raise DeploymentError(receipt["error"], partial_receipt=receipt)
    return post
