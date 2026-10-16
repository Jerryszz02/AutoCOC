"""Opt-in, two-stage deployment at two individually proved screen points.

The probes are game inputs, not a substitute for the independent army, card,
camera and terrain preflight in ``freeze_model_free_line``.  Their only effect
on the frozen plan is to authorize the two *already compiled* exact points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Any, Callable

from .errors import DeploymentError, FlowError, StopRequested
from .frozen_line import (FrozenLinePlan, _COUNT, _closed_card_border, _digest, _file_sha256, _plan_digest,
                          _cards, _count_proves, _post_troop_reading, _verify_frozen_line_post,
                          freeze_model_free_line)
from .prepared_battle import PreparedInput


@dataclass(frozen=True)
class ProbedAnchorPlan:
    base: FrozenLinePlan
    anchors: tuple[tuple[int, int], tuple[int, int]]
    balloon_point: tuple[int, int]
    balloon_bbox: tuple[int, int, int, int]
    checkpoint_cards: tuple[tuple[str, tuple[int, int], tuple[int, int, int, int]], ...]
    remaining: tuple[PreparedInput, ...]
    digest: str


class ExactCameraError(FlowError):
    """Only an exact map/camera proof failed; other frame gates stay separate."""


def _digest_plan(plan: ProbedAnchorPlan) -> str:
    return _digest({"base": plan.base.plan_sha256, "anchors": plan.anchors,
                    "balloon_point": plan.balloon_point, "balloon_bbox": plan.balloon_bbox,
                    "checkpoint_cards": plan.checkpoint_cards,
                    "remaining": [asdict(action) for action in plan.remaining]})


def remainder_burst_limit(plan: ProbedAnchorPlan) -> float:
    """The independently timed frozen batch budget after both probes."""
    return min(10., 1. + .25 * len(plan.remaining))


def _distance_to_segment(point: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    fraction = max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy)


def freeze_probed_anchor_line(session: Any, selected: Any, *, terrain: list[dict], **kwargs: Any) -> ProbedAnchorPlan:
    """Freeze the entire roster and both exact points *before* any probe input."""
    if tuple(session.config.game.baseline_resolution) != (1280, 720):
        raise FlowError("Probed anchors require the sampled 1280×720 card and camera layout")
    base = freeze_model_free_line(session, selected, terrain=terrain,
                                  line_tactics="two_edge_split", **kwargs)
    balloons = [card for card in base.troop_cards if card[0] == "balloon"]
    if len(balloons) != 1 or balloons[0][2] != 2 or len(base.hero_ids) != 4:
        raise FlowError("Probed anchors require exactly two identified army balloons and four heroes")
    edge_segments = []
    for edge in (0, 1):
        points = sorted((tuple(row["point"]) for row in terrain if row["edge"] == edge),
                        key=lambda point: point[1])
        if len(points) != 2:
            raise FlowError("Probed anchors require two independently verified edges")
        edge_segments.append((points[0], points[1]))
    balloon_inputs = [action for action in base.actions if action.kind == "deploy" and action.unit_id == "balloon"]
    if len(balloon_inputs) != 2:
        raise FlowError("Probed anchors cannot identify two balloon placements")
    anchors = (balloon_inputs[0].point, balloon_inputs[1].point)
    if (_distance_to_segment(anchors[0], *edge_segments[0]) > 1 or
            _distance_to_segment(anchors[1], *edge_segments[1]) > 1 or anchors[0] == anchors[1]):
        raise FlowError("Probed anchor points differ from the frozen two-edge allocation")
    remaining = []
    for action in base.actions:
        if action.unit_id == "balloon":
            continue  # Both balloon deployments and their card selection are probe inputs.
        if action.kind == "deploy":
            distances = [_distance_to_segment(action.point, *segment) for segment in edge_segments]
            if abs(distances[0] - distances[1]) < 16:
                raise FlowError("Frozen placement cannot be assigned to a unique proved edge")
            action = PreparedInput(action.action_id, action.kind, anchors[0 if distances[0] < distances[1] else 1],
                                   action.unit_id, action.source, action.delay_after_sec)
        remaining.append(action)
    deployments = [action for action in remaining if action.kind == "deploy"]
    if len(deployments) != sum(row[2] for row in base.troop_budgets) - 2 + len(base.hero_ids):
        raise FlowError("Probed anchor remainder does not preserve the frozen army")
    _, _, _, balloon_point, balloon_bbox = balloons[0]
    # The first troop is selected in base.frame, so its expanded selected-card
    # bbox is not the ordinary geometry it will have after selecting balloon.
    # base.toolbar_before was independently verified by freeze_model_free_line.
    toolbar_before = kwargs.get("toolbar_before")
    if (toolbar_before is None or
            str(toolbar_before.screenshot_path) != base.toolbar_before_frame_id or
            _file_sha256(toolbar_before.screenshot_path) != base.toolbar_before_sha256):
        raise FlowError("Probe checkpoint ordinary card source is not the frozen toolbar frame")
    previous = {card["unit_id"]: card for card in
                _cards(toolbar_before, client_version=base.client_version,
                       expected={(unit_id, source): count for unit_id, source, count in base.troop_budgets})
                if card["kind"] == "troop"}
    checkpoint_cards = []
    for unit_id, source, count, point, bbox in base.troop_cards:
        if unit_id == "balloon":
            continue
        card = previous.get(unit_id)
        if (card is None or card["source"] != source or card["count"] != count or
                any(abs(card["point"][axis] - point[axis]) > 4 for axis in (0, 1))):
            raise FlowError("Probe checkpoint lacks independently verified ordinary card geometry")
        checkpoint_cards.append((unit_id, tuple(card["point"]), tuple(card["bbox"])))
    plan = ProbedAnchorPlan(base, anchors, balloon_point, balloon_bbox,
                            tuple(checkpoint_cards), tuple(remaining), "")
    return ProbedAnchorPlan(base, anchors, balloon_point, balloon_bbox,
                            tuple(checkpoint_cards), tuple(remaining), _digest_plan(plan))


def verify_exact_anchor_camera(source_path: Any, current_path: Any,
                               anchors: tuple[tuple[int, int], tuple[int, int]]) -> dict:
    """Require one stable affine map of the battlefield and unchanged rounded anchors.

    The existing <8-pixel stationary test is intentionally not used as proof
    that a single legal screen coordinate still means the same ground point.
    """
    import cv2
    import numpy as np
    from .images import read_frame

    prior = cv2.resize(read_frame(source_path), (1280, 720), interpolation=cv2.INTER_AREA)[150:490, 240:1050]
    current = cv2.resize(read_frame(current_path), (1280, 720), interpolation=cv2.INTER_AREA)[150:490, 240:1050]
    detector = cv2.ORB_create(nfeatures=1200)
    points_a, descriptors_a = detector.detectAndCompute(cv2.cvtColor(prior, cv2.COLOR_BGR2GRAY), None)
    points_b, descriptors_b = detector.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), None)
    if descriptors_a is None or descriptors_b is None:
        raise ExactCameraError("Probe camera has too few independent map features")
    # The old camera proof used Hamming<40 to discard changed troop/particle
    # texture. In actual 18.647.2 preview→battle frames it leaves hundreds of
    # independent map features; unmatched animations cannot dilute inliers.
    pairs = [item for item in cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
        descriptors_a, descriptors_b) if item.distance < 40]
    pairs = sorted(pairs, key=lambda item: item.distance)
    if len(pairs) < 80:
        raise ExactCameraError("Probe camera has too few independent map matches")
    coordinates_a = np.float32([points_a[m.queryIdx].pt for m in pairs])
    coordinates_b = np.float32([points_b[m.trainIdx].pt for m in pairs])
    transform, mask = cv2.estimateAffinePartial2D(coordinates_a, coordinates_b,
                                                    method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if transform is None or mask is None:
        raise ExactCameraError("Probe camera transform is unverified")
    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < 80 or float(inliers.mean()) < .9:
        raise ExactCameraError("Probe camera has insufficient stationary map inliers")
    projected = coordinates_a @ transform[:, :2].T + transform[:, 2]
    residual = float(np.percentile(np.linalg.norm(projected[inliers] - coordinates_b[inliers], axis=1), 95))
    if not math.isfinite(residual) or residual > .5:
        raise ExactCameraError("Probe camera map residual is too large")
    offsets = []
    for x, y in anchors:
        mapped = transform @ np.array([x - 240, y - 150, 1.]) + np.array([240., 150.])
        shift = float(np.linalg.norm(mapped - np.array([x, y])))
        if not math.isfinite(shift) or shift > .25 or tuple(np.rint(mapped).astype(int)) != (x, y):
            raise ExactCameraError("Probe camera moved an exact deployment anchor")
        offsets.append(shift)
    return {"matches": len(pairs), "inliers": int(inliers.sum()),
            "residual_p95": residual, "anchor_offsets": offsets}


def _frame_envelope(session: Any, plan: ProbedAnchorPlan, snapshot: Any, *, after: float,
                    scene: str | None = None) -> dict:
    """Fresh battle, version, display, source pixels and foreground, without camera permission."""
    from .device import foreground_package

    if not isinstance(plan, ProbedAnchorPlan):
        raise FlowError("Probed anchor plan type is invalid")
    base = plan.base
    observed = snapshot.observations.get("observed_at_monotonic")
    if snapshot.scene == "account_in_use":
        raise FlowError("Game interruption: account_in_use")
    if (snapshot is not session.last_snapshot or snapshot.scene not in {"enemy_village", "battle"} or
            scene is not None and snapshot.scene != scene or
            type(snapshot.confidence) not in (int, float) or not math.isfinite(snapshot.confidence) or
            snapshot.confidence < .8 or type(observed) not in (int, float) or
            not math.isfinite(observed) or observed <= after or
            str(snapshot.screenshot_path) == base.frame.frame_id and after >= base.frame.captured_at or
            tuple(snapshot.observations.get("source_resolution") or ()) != base.frame.source_resolution or
            tuple(snapshot.observations.get("baseline_resolution") or ()) != base.frame.resolution or
            session.capture.input_display_id != base.frame.display_id or
            session.client_version != base.client_version or
            _file_sha256(base.frame.screenshot_path) != base.frame_sha256):
        raise FlowError("Probe has no fresh same-display current-version battle frame")
    if foreground_package(session.capture.adb, session.capture.serial,
                          timeout_sec=min(5, session.config.runtime.step_timeout_sec)) != session.config.game.package_name:
        raise FlowError("Probe game foreground is not independently verified")
    return {"frame": str(snapshot.screenshot_path), "sha256": _file_sha256(snapshot.screenshot_path),
            "observed_at": observed}


def _current_frame(session: Any, plan: ProbedAnchorPlan, snapshot: Any, *, after: float,
                   scene: str | None = None) -> dict:
    """The usual unchanged exact-point camera proof remains mandatory."""
    proof = _frame_envelope(session, plan, snapshot, after=after, scene=scene)
    try:
        proof["camera"] = verify_exact_anchor_camera(plan.base.frame.screenshot_path,
                                                       snapshot.screenshot_path, plan.anchors)
    except ExactCameraError as exc:
        exc.frame_proof = proof  # The sole recoverable failure follows all stronger frame gates.
        raise
    return proof


def _foreground_now(session: Any) -> None:
    from .device import foreground_package

    if foreground_package(session.capture.adb, session.capture.serial,
                          timeout_sec=min(5, session.config.runtime.step_timeout_sec)) != session.config.game.package_name:
        raise FlowError("Probed anchor game foreground changed before input")


def _current_card_edges(image: Any, bbox: tuple[int, int, int, int]) -> dict | None:
    """Reuse vision's independent opposite-edge and closed-bottom card contract."""
    import cv2
    import numpy as np

    left, top, right, bottom = bbox
    if (image.shape[:2] != (720, 1280) or not 3 <= left < right <= 1277 or
            not 590 <= top <= 605 or bottom != 711):
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    direction = np.median(cv2.Sobel(gray[625:695], cv2.CV_32F, 1, 0, ksize=3), axis=0)
    left_edge = float(max(direction[left - 2:left + 3]))
    right_edge = float(min(direction[right - 4:right]))
    bottom_contrast = float(gray[709:711, left + 8:right - 8].mean() -
                            gray[712, left + 8:right - 8].mean())
    if left_edge < 150 or right_edge > -150 or bottom_contrast < 40:
        return None
    return {"left_edge": left_edge, "right_edge": right_edge,
            "bottom_contrast": bottom_contrast}


def _remaining_roster_checkpoint(session: Any, plan: ProbedAnchorPlan, snapshot: Any) -> dict:
    """One frame proves all unspent cards before any unobserved remaining batch."""
    import cv2
    from .images import read_frame
    from .unit_catalog import recognize_card_identity, template_manifest, CATALOG_ROOT
    from .hero_state import recognize_hero_state

    base = plan.base
    image = cv2.resize(read_frame(snapshot.screenshot_path), (1280, 720), interpolation=cv2.INTER_AREA)
    slots = (snapshot.observations.get("battle") or {}).get("slots") or []
    if not isinstance(slots, list):
        raise FlowError("Probe checkpoint battle cards are unavailable")
    positive = []
    ordinary = {unit_id: (point, bbox) for unit_id, point, bbox in plan.checkpoint_cards}
    if len(ordinary) != len(base.troop_cards) - 1:
        raise FlowError("Probe checkpoint ordinary card geometry is incomplete")
    for unit_id, source, initial, point, bbox in base.troop_cards:
        if unit_id == "balloon":
            continue
        if unit_id not in ordinary:
            raise FlowError(f"Probe checkpoint {unit_id} ordinary card geometry is missing")
        point, bbox = ordinary[unit_id]
        border = _current_card_edges(image, bbox)
        identity = recognize_card_identity(image, bbox, "troop", surface="battle",
                                           client_version=base.client_version)
        if (border is None or identity.get("unit_id") != unit_id or
                identity.get("version_verified") is not True or
                type(identity.get("confidence")) not in (int, float) or identity["confidence"] < .94):
            raise FlowError(f"Probe checkpoint {unit_id} current card identity/border changed")
        if any(isinstance(row, dict) and
               (row.get("unit_id") == unit_id or row.get("bbox") == list(bbox)) and
               type(row.get("count")) is int and row["count"] != initial and
               type(row.get("confidence")) in (int, float) and
               math.isfinite(row["confidence"]) and row["confidence"] >= .9
               for row in slots):
            raise FlowError(f"Probe checkpoint {unit_id} strong full-card quantity conflict")
        matches = [row for row in slots if isinstance(row, dict) and row.get("unit_id") == unit_id and
                   row.get("source") == source and row.get("bbox") == list(bbox) and
                   row.get("count") == initial and
                   type(row.get("confidence")) in (int, float) and row["confidence"] >= .9 and
                   ((row.get("evidence") or {}).get("identity") or {}).get("unit_id") == unit_id and
                   ((row.get("evidence") or {}).get("identity") or {}).get("version_verified") is True and
                   ((row.get("evidence") or {}).get("identity") or {}).get("template_client_version") == base.client_version and
                   _count_proves((row.get("evidence") or {}).get("count") or {}, initial,
                                 snapshot.screenshot_path, bbox, unit_id, base.client_version)]
        local = _post_troop_reading(session, snapshot.screenshot_path, base.toolbar_before_frame_id,
                                    point, bbox, unit_id=unit_id)
        if local.get("conflict") is True:
            raise FlowError(f"Probe checkpoint {unit_id} has a strong count conflict")
        local_valid = _count_proves(local, initial, snapshot.screenshot_path, bbox, unit_id,
                                    base.client_version)
        if not matches and not local_valid:
            raise FlowError(f"Probe checkpoint {unit_id} quantity changed or is unknown")
        if (type(local.get("count")) is int and local.get("count") != initial and
                type(local.get("confidence")) in (int, float) and local["confidence"] >= .9):
            raise FlowError(f"Probe checkpoint {unit_id} local count conflicts with full card")
        positive.append({"unit_id": unit_id, "count": initial,
                         "identity": identity, "border": border,
                         "count_source": "full_card" if matches else "portrait_bound_local",
                         "local": local})
    initial_templates = {(entry.get("unit_id"), str((CATALOG_ROOT / entry["path"]).resolve()))
                         for entry in template_manifest(client_version=base.client_version)["templates"]
                         if entry.get("surface") == "battle" and entry.get("state") == "initial_scout"
                         and isinstance(entry.get("path"), str)}
    heroes = []
    for unit_id, bbox in base.hero_cards:
        border = _current_card_edges(image, bbox)
        identity = recognize_card_identity(image, bbox, "hero", surface="battle",
                                           client_version=base.client_version)
        top = (identity.get("evidence") or [{}])[0]
        state = recognize_hero_state(snapshot.screenshot_path, bbox,
                                     unit_id=unit_id, client_version=base.client_version)
        if (border is None or identity.get("unit_id") != unit_id or
                identity.get("version_verified") is not True or
                type(identity.get("confidence")) not in (int, float) or identity["confidence"] < .94 or
                (unit_id, top.get("template")) not in initial_templates or
                state.get("deployed") is True or
                ((state.get("evidence") or {}).get("portrait_identity") or {}).get("unit_id") != unit_id):
            raise FlowError(f"Probe checkpoint {unit_id} is not one current undeployed hero card")
        heroes.append({"unit_id": unit_id, "identity": identity, "border": border,
                       "state": state.get("state")})
    boxes = [bbox for _, _, bbox in plan.checkpoint_cards] + [plan.balloon_bbox] + [row[1] for row in base.hero_cards]
    if any(left[0] < right[2] and right[0] < left[2]
           for index, left in enumerate(boxes) for right in boxes[index + 1:]):
        raise FlowError("Probe checkpoint cards overlap the spent balloon")
    return {"frame": str(snapshot.screenshot_path), "sha256": _file_sha256(snapshot.screenshot_path),
            "troops": positive, "heroes": heroes}


def _spent_balloon_dark_border(current: Any, bbox: tuple[int, int, int, int]) -> dict | None:
    """18.647.2 gray-card outline, with a versioned shape for dark backgrounds."""
    import cv2
    import numpy as np

    left, top, right, bottom = bbox
    if (current.shape[:2] != (720, 1280) or not 3 <= left < right <= 1277 or
            not 590 <= top <= 605 or bottom != 711 or not 80 <= right - left <= 96):
        return None
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    horizontal_edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    direction = np.median(horizontal_edges[625:695], axis=0)
    left_edge, right_edge = float(direction[left]), float(direction[right - 1])
    # Retain the stronger opposite-side requirements of _closed_card_border.
    if left_edge < 250 or right_edge > -150:
        return None
    # The straight sides must reach the rounded lower corners. A detached
    # horizontal dark stripe is not a closed card, even if both upper sides
    # and the stripe independently look strong.
    for row in range(696, 704):
        if horizontal_edges[row, left] < 150 or horizontal_edges[row, right - 1] > -150:
            return None
    left_position, right_position = left, right - 1
    for row in range(704, 711):
        left_band = horizontal_edges[row, left:left + 9]
        right_band = horizontal_edges[row, right - 9:right]
        next_left = left + int(np.argmax(left_band))
        next_right = right - 9 + int(np.argmin(right_band))
        if (float(left_band.max()) < 40 or float(right_band.min()) > -40 or
                not left_position <= next_left <= left_position + 2 or
                not right_position - 2 <= next_right <= right_position):
            return None
        left_position, right_position = next_left, next_right
    if left_position < left + 5 or right_position > right - 6:
        return None
    interior = slice(left + 8, right - 8)
    candidates = []
    for row in range(bottom - 1, bottom + 2):
        above = gray[row - 3:row, interior].mean(axis=0)
        trough = gray[row:row + 2, interior].mean(axis=0)
        below = gray[row + 2:row + 5, interior].mean(axis=0)
        # Split the existing 40-level bottom contrast across both sides of
        # one dark stroke. Every interior column, including both rounded
        # corner bands, must show the same closed outline.
        if np.all((above - trough >= 20) & (below - trough >= 20) & (trough <= 16)):
            candidates.append((row, float(np.median(above - trough)),
                               float(np.median(below - trough))))
    if len(candidates) == 1:
        row, upper_contrast, lower_contrast = candidates[0]
        return {"method": "spent_balloon_dark_outline", "left_edge": left_edge,
                "right_edge": right_edge, "dark_row": row,
                "corner_endpoints": [left_position, right_position],
                "upper_contrast": upper_contrast, "lower_contrast": lower_contrast,
                "continuous_columns": right - left - 16}

    # The lower scene may be dark enough that a complete gray card has less
    # than 20 levels of contrast below its stroke. Compare the card's own
    # bottom shape to real 18.647.2 pixels; do not infer a border from x0.
    from .unit_catalog import CATALOG_ROOT

    template_path = CATALOG_ROOT / "units" / "balloon_spent_bottom_18_647_2.png"
    try:
        template_bytes = template_path.read_bytes()
    except OSError:
        return None
    template_sha = hashlib.sha256(template_bytes).hexdigest()
    if template_sha != "a43b2aa24a3c73ada03aa6646c2c4142a88411740c902c19bc473f6a7f606ce8":
        return None
    template = cv2.imdecode(np.frombuffer(template_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    core = gray[bottom - 5:bottom + 2, interior]
    if template is None or template.shape != core.shape or template.shape != (7, 72):
        return None
    core_max_error = int(np.abs(core.astype(np.int16) - template.astype(np.int16)).max())
    if core_max_error > 4:
        return None
    trough = gray[bottom:bottom + 2, interior].mean(axis=0)
    below = gray[bottom + 2:bottom + 5, interior].mean(axis=0)
    farther = gray[bottom + 5:bottom + 8, interior].mean(axis=0)
    rebound = below - trough
    background_jump = np.abs(below - farther)
    # Rows beyond the closed outline belong to the moving battlefield. Their
    # texture is diagnostic only: the versioned core, connected sides/corners
    # and contrast immediately below the stroke already prove the card edge.
    if np.any(rebound < 7):
        return None
    return {"method": "spent_balloon_versioned_bottom_shape", "left_edge": left_edge,
            "right_edge": right_edge, "dark_row": bottom,
            "corner_endpoints": [left_position, right_position],
            "core_template_sha256": template_sha, "core_max_error": core_max_error,
            "lower_rebound_min": float(rebound.min()),
            "background_jump_p95": float(np.percentile(background_jump, 95)),
            "continuous_columns": right - left - 16}


def _balloon_count(session: Any, plan: ProbedAnchorPlan, snapshot: Any, expected: int, *,
                   source_bound: bool = False, selected: bool = False,
                   pending: int | None = None) -> dict:
    base = plan.base
    reading = _post_troop_reading(session, snapshot.screenshot_path, base.toolbar_before_frame_id,
                                  plan.balloon_point, plan.balloon_bbox, unit_id="balloon")
    score, portrait = reading.get("confidence"), reading.get("portrait_score")
    reads = reading.get("readings")
    actual = reading.get("count")
    if reading.get("conflict") is True:
        raise FlowError("Probe strong_count_conflict on the current balloon card")
    if (type(actual) is not int or
            type(score) not in (int, float) or not math.isfinite(score) or score < .9 or
            type(portrait) not in (int, float) or not math.isfinite(portrait) or portrait < .92 or
            reading.get("portrait_source") != "current_frame_saved_battle_portrait" or
            reading.get("frame") != str(snapshot.screenshot_path) or
            reading.get("slot_bbox") != list(plan.balloon_bbox) or
            not isinstance(reads, list) or not reads or
            {int(match[1]) for item in reads if isinstance(item, dict) and
             type(item.get("confidence")) in (int, float) and item["confidence"] >= .9 and
             (match := _COUNT.fullmatch(str(item.get("text", "")).strip())) is not None} != {actual}):
        raise FlowError("Probe balloon count lacks one strong same-frame portrait-bound value")
    if actual not in ({expected, pending} if pending is not None else {expected}):
        raise FlowError(f"Probe balloon consumption differs from frozen {expected} count")
    if snapshot.observations.get("recognition_scope") == "troop_count":
        fast_reads = snapshot.observations.get("deployment_count_reads")
        if (not isinstance(fast_reads, list) or len(fast_reads) != 1 or
                not isinstance(fast_reads[0], dict) or
                fast_reads[0].get("frame") != str(snapshot.screenshot_path) or
                fast_reads[0].get("slot_bbox") != list(plan.balloon_bbox) or
                type(fast_reads[0].get("count")) is not int or
                type(fast_reads[0].get("confidence")) not in (int, float) or
                not math.isfinite(fast_reads[0]["confidence"]) or
                fast_reads[0]["confidence"] < .9):
            raise FlowError("Probe fast current-card count lacks one strong bound value")
        if fast_reads[0]["count"] != actual:
            raise FlowError("Probe strong_count_conflict between fast and portrait-bound OCR")
    slots = (snapshot.observations.get("battle") or {}).get("slots") or []
    competing = [row for row in slots if isinstance(row, dict) and
                 (row.get("unit_id") == "balloon" or row.get("bbox") == list(plan.balloon_bbox))]
    if any(type(row.get("count")) is int and row["count"] != actual and
           type(row.get("confidence")) in (int, float) and row["confidence"] >= .9 for row in competing):
        raise FlowError("Probe strong_count_conflict between full card and local OCR")
    for row in competing:
        score = row.get("confidence")
        if type(score) not in (int, float) or not math.isfinite(score) or score < .9:
            continue
        identity = ((row.get("evidence") or {}).get("identity") or {})
        if (row.get("unit_id") not in (None, "balloon") or
                row.get("unit_id") == "balloon" and
                (identity.get("unit_id") not in (None, "balloon") or
                 identity.get("version_verified") is False or
                 identity.get("template_client_version") not in (None, base.client_version))):
            raise FlowError("Probe strong current-card identity conflicts with frozen balloon")
    if actual > 0:
        identity = [row for row in competing if row.get("unit_id") == "balloon" and
                    row.get("bbox") == list(plan.balloon_bbox) and row.get("count") == actual and
                    row.get("source") == "army" and
                    type(row.get("confidence")) in (int, float) and row["confidence"] >= .9 and
                    ((row.get("evidence") or {}).get("identity") or {}).get("unit_id") == "balloon" and
                    ((row.get("evidence") or {}).get("identity") or {}).get("version_verified") is True and
                    ((row.get("evidence") or {}).get("identity") or {}).get("template_client_version") == base.client_version]
        if len(identity) != 1 and not source_bound:
            raise FlowError("Probe positive balloon card has no current-frame versioned identity")
        if source_bound:
            from .images import read_frame
            from .unit_catalog import recognize_card_identity
            from .battle_vision import selected_card_bbox
            import cv2

            if _file_sha256(base.frame.screenshot_path) != base.frame_sha256:
                raise FlowError("Probe versioned balloon source pixels changed")
            source = cv2.resize(read_frame(base.frame.screenshot_path), (1280, 720),
                                interpolation=cv2.INTER_AREA)
            source_identity = recognize_card_identity(source, plan.balloon_bbox, "troop",
                                                      surface="battle", client_version=base.client_version)
            if (source_identity.get("unit_id") != "balloon" or
                    source_identity.get("version_verified") is not True or
                    source_identity.get("template_client_version") != base.client_version or
                    type(source_identity.get("confidence")) not in (int, float) or
                    not math.isfinite(source_identity["confidence"]) or
                    source_identity["confidence"] < .94):
                raise FlowError("Probe source balloon versioned identity is unverified")
            reading["source_portrait_identity"] = source_identity
            reading["source_identity_frame"] = base.frame.frame_id
            if selected:
                border = selected_card_bbox(snapshot.screenshot_path, plan.balloon_bbox,
                                            base.frame.resolution)
                if border is None:
                    raise FlowError("Probe selected balloon lacks the closed current card border")
                reading["selected_card_border"] = list(border)
    else:
        from .images import read_frame
        import cv2
        current = cv2.resize(read_frame(snapshot.screenshot_path), (1280, 720), interpolation=cv2.INTER_AREA)
        border = _closed_card_border(current, plan.balloon_bbox)
        if border is None:
            # This fallback is specific to the observed current-version gray
            # x0 card. It does not change the legacy border contract.
            if base.client_version != "18.647.2":
                raise FlowError("Probe spent balloon lacks the independent closed-card border")
            from .unit_catalog import recognize_card_identity
            if _file_sha256(base.frame.screenshot_path) != base.frame_sha256:
                raise FlowError("Probe versioned balloon source pixels changed")
            source = cv2.resize(read_frame(base.frame.screenshot_path), (1280, 720),
                                interpolation=cv2.INTER_AREA)
            identity = recognize_card_identity(source, plan.balloon_bbox, "troop",
                                               surface="battle", client_version=base.client_version)
            if (identity.get("unit_id") != "balloon" or
                    identity.get("version_verified") is not True or
                    identity.get("template_client_version") != base.client_version or
                    type(identity.get("confidence")) not in (int, float) or
                    not math.isfinite(identity["confidence"]) or identity["confidence"] < .94):
                raise FlowError("Probe source balloon versioned identity is unverified")
            border = _spent_balloon_dark_border(current, plan.balloon_bbox)
            if border is None:
                raise FlowError("Probe spent balloon lacks the independent closed-card border")
            reading["source_portrait_identity"] = identity
            reading["source_identity_frame"] = base.frame.frame_id
        reading["current_card_border"] = border
    return reading


def _recover_original_anchors(session: Any, plan: ProbedAnchorPlan, receipt: dict, *,
                              after: float, clock: Callable[[], float]) -> tuple:
    """Observe only: two fresh full frames must return to the *old* exact points."""
    started = clock()
    burst = remainder_burst_limit(plan)
    latest = min(started + 12., session.deadline - 90. - burst,
                 session.task_deadline - 90. - burst)
    recovery = {"status": "observing", "started_at": started, "limit_sec": 12.,
                "deadline": latest, "frames": [], "elapsed_sec": None,
                "authorized_at": None, "old_points_only": True, "input_count": 0}
    receipt["recovery"] = recovery
    candidate = None
    try:
        if started >= latest:
            raise FlowError("Probe recovery lacks a twelve-second window and settlement reserve")
        for index in range(3):
            session.check_deadline()
            if clock() >= latest:
                raise FlowError("Probe original-anchor recovery deadline expired")
            current = session.observe(f"probed-anchor-recovery-full-{index}")
            session.check_deadline()
            receipt["evidence"].append(str(current.screenshot_path))
            row = {"observation_index": index, "frame": str(current.screenshot_path),
                   "sha256": None, "status": "checking"}
            recovery["frames"].append(row)
            try:
                row["sha256"] = _file_sha256(current.screenshot_path)
                if current.scene == "account_in_use":
                    raise FlowError("Game interruption: account_in_use")
                if (current.scene != "battle" or
                        current.observations.get("recognition_scope") not in (None, "full") or
                        current.observations.get("fast_fallback") is True):
                    raise FlowError("Probe recovery requires one independent full battle frame")
                proof = _frame_envelope(session, plan, current, after=after, scene="battle")
                after = proof["observed_at"]
                row["sha256"] = proof["sha256"]
                balloon = _balloon_count(session, plan, current, 0)
                checkpoint = _remaining_roster_checkpoint(session, plan, current)
                row.update(balloon_count=balloon["count"], roster_sha256=checkpoint["sha256"])
                if (_file_sha256(current.screenshot_path) != proof["sha256"] or
                        _file_sha256(checkpoint["frame"]) != checkpoint["sha256"] or
                        _file_sha256(plan.base.frame.screenshot_path) != plan.base.frame_sha256):
                    raise FlowError("Probe recovery frame pixels changed during the full roster proof")
                if clock() >= latest:
                    raise FlowError("Probe recovery proof exceeded its bounded window")
                try:
                    proof["camera"] = verify_exact_anchor_camera(plan.base.frame.screenshot_path,
                                                                   current.screenshot_path, plan.anchors)
                except ExactCameraError as exc:
                    if _file_sha256(plan.base.frame.screenshot_path) != plan.base.frame_sha256:
                        raise FlowError("Probe recovery frozen camera source pixels changed") from exc
                    row.update(status="camera_rejected", reason=str(exc))
                    candidate = None
                    continue
                row["source_camera"] = proof["camera"]
                if (_file_sha256(plan.base.frame.screenshot_path) != plan.base.frame_sha256 or
                        _file_sha256(current.screenshot_path) != proof["sha256"]):
                    raise FlowError("Probe recovery exact-camera proof pixels changed")
                if candidate is None:
                    row["status"] = "first_exact_frame"
                    candidate = (current, proof)
                    continue
                previous, previous_proof = candidate
                try:
                    row["adjacent_camera"] = verify_exact_anchor_camera(previous.screenshot_path,
                                                                          current.screenshot_path, plan.anchors)
                except ExactCameraError as exc:
                    if (_file_sha256(previous.screenshot_path) != previous_proof["sha256"] or
                            _file_sha256(current.screenshot_path) != proof["sha256"]):
                        raise FlowError("Probe recovery adjacent camera pixels changed") from exc
                    row.update(status="camera_rejected", reason=str(exc))
                    candidate = None
                    continue
                if (_file_sha256(previous.screenshot_path) != previous_proof["sha256"] or
                        _file_sha256(current.screenshot_path) != proof["sha256"] or
                        _file_sha256(plan.base.frame.screenshot_path) != plan.base.frame_sha256):
                    raise FlowError("Probe recovery camera source pixels changed")
                if clock() >= latest:
                    raise FlowError("Probe recovery two-frame proof exceeded its deadline")
                row["status"] = "second_exact_frame"
                recovery["status"] = "two_frame_proved"
                return current, proof, balloon, checkpoint, latest
            except BaseException as exc:
                if row["status"] == "checking":
                    row.update(status="rejected", reason=str(exc))
                raise
        raise FlowError("Probe original-anchor recovery lacks two consecutive exact full frames")
    except BaseException as exc:
        recovery.update(status="failed", reason=str(exc))
        raise
    finally:
        recovery["elapsed_sec"] = clock() - started


def execute_probed_anchor_line(session: Any, plan: ProbedAnchorPlan, *,
                               clock: Callable[[], float] = time.monotonic,
                               sleep: Callable[[float], None] = time.sleep) -> Any:
    """One input per probe, then one frozen remainder; no failed input is repeated."""
    if not isinstance(plan, ProbedAnchorPlan):
        raise FlowError("Probed anchor plan type is invalid")
    base = plan.base
    if (_digest_plan(plan) != plan.digest or
            _plan_digest(base) != base.plan_sha256 or
            base.frame.resolution != (1280, 720) or
            session.last_snapshot is None or str(session.last_snapshot.screenshot_path) != base.frame.frame_id or
            clock() >= base.latest_start or session.client_version != base.client_version or
            session.capture.input_display_id != base.frame.display_id or
            tuple(session.context.screen_resolution) != base.frame.source_resolution or
            _file_sha256(base.frame.screenshot_path) != base.frame_sha256 or
            _file_sha256(base.toolbar_before_frame_id) != base.toolbar_before_sha256 or
            base.hero_manifest_frame_id is not None and
            _file_sha256(base.hero_manifest_frame_id) != base.hero_manifest_frame_sha256):
        raise FlowError("Probed anchor frozen source permission is invalid")
    used = getattr(session, "_used_frozen_frames", None)
    if used is None:
        used = set()
        session._used_frozen_frames = used
    if base.frame.frame_id in used:
        raise FlowError("Probed anchor source frame was already attempted")
    used.add(base.frame.frame_id)
    receipt = {"mode": "model_free_line_probed_anchors", "line_tactics": "two_edge_probed_anchors",
               "plan_id": base.plan_id, "plan_sha256": plan.digest, "battle_id": base.battle_id,
               "frame": base.frame.frame_id, "scope": "troops_and_heroes",
               "excluded_kinds": ["spell", "siege"], "anchors": plan.anchors,
               "probe_limit_sec": 12., "probe_events": [], "actions": [],
               "input_count": len(plan.remaining) + 3, "attempted_placements": 0,
               "issued_placements": 0, "deployed_units": 0, "verified": False,
               "completed": False, "input_sent": False, "any_input_sent": False,
               "burst_pass": False,
               "consumption_events": [], "evidence": [base.frame.frame_id], "error": None}
    session.frozen_line_receipt = receipt
    started = clock()
    receipt.update(probe_started_at=started, probe_elapsed_sec=None, probe_authorized_at=None)
    try:
        current = session.last_snapshot
        recovered_checkpoint = None
        authorization_deadline = min(started + 12, session.deadline, session.task_deadline)
        balloon_select = clock()
        if balloon_select >= base.latest_start:
            raise FlowError("First probe selection missed the verified preview deadline")
        first_source = _current_frame(session, plan, current,
                                      after=base.frame.captured_at - 1, scene="enemy_village")
        # The selected preflight card is the first troop, never infer that the
        # balloon remained selected after any intervening UI or capture.
        if _file_sha256(current.screenshot_path) != first_source["sha256"]:
            raise FlowError("Probe selected source pixels changed before input")
        _foreground_now(session)
        if _file_sha256(current.screenshot_path) != first_source["sha256"] or clock() >= base.latest_start:
            raise FlowError("Probe selection proof consumed the preview deadline")
        session.check_deadline()
        if clock() >= base.latest_start:
            raise FlowError("Probe selection stopped before the preview input deadline")
        selection = {"action_id": "select-probe-balloon", "status": "attempted",
                     "kind": "select", "point": plan.balloon_point}
        receipt["actions"].append(selection)
        try:
            session.tap(current, plan.balloon_point, reason="Select frozen balloon probe card")
        except BaseException:
            selection["status"] = "uncertain"
            raise
        selection.update(status="input_sent", done_at=clock())
        receipt["any_input_sent"] = True
        select_done = receipt["actions"][-1]["done_at"]
        from .battle_vision import selected_card_bbox
        selection_deadline = min(select_done + 3, started + 12, base.latest_start,
                                 session.deadline, session.task_deadline)
        selected_proof = None
        for observation in range(3):
            if clock() >= selection_deadline:
                break
            current = session.observe("probed-anchor-balloon-selected", purpose="troop_count",
                                      slot={"point": list(plan.balloon_point),
                                            "bbox": list(plan.balloon_bbox)})
            receipt["evidence"].append(str(current.screenshot_path))
            if current.scene == "account_in_use":
                raise FlowError("Game interruption: account_in_use")
            if current.observations.get("deployment_count_conflict") is not None:
                raise FlowError("Probe strong_count_conflict in the fast selected-card frame")
            # Fast recognition must prove both HUD anchors on this image. It
            # must not inherit an enemy-preview scene from the prior full OCR.
            if (current.observations.get("recognition_scope") != "troop_count" or
                    current.observations.get("fast_fallback") is not False or
                    current.observations.get("matched_reasons") != ["two_current_hud_anchors"]):
                raise FlowError("Probe selected frame lacks independent fast-scene guard")
            proof = _current_frame(session, plan, current, after=select_done, scene="enemy_village")
            selected_border = selected_card_bbox(current.screenshot_path, plan.balloon_bbox,
                                                 base.frame.resolution)
            reading = _balloon_count(session, plan, current, 2, source_bound=True,
                                     selected=selected_border is not None)
            if clock() >= selection_deadline:
                raise FlowError("Probe selected-card proof exceeded its preview budget")
            if selected_border is not None:
                selected_proof = proof
                receipt["selection_reading"] = reading
                break
        if selected_proof is None:
            raise FlowError("Probe balloon card is not selected on a fresh guarded preview")
        for index, anchor in enumerate(plan.anchors):
            stage_start = clock()
            stage_deadline = min(stage_start + 6, started + 12, session.deadline, session.task_deadline)
            if stage_start >= stage_deadline or index == 0 and stage_start >= base.latest_start:
                raise FlowError("Probe input deadline expired")
            pre = selected_proof if index == 0 else confirmed[0]
            if (current is not session.last_snapshot or
                    _file_sha256(current.screenshot_path) != pre["sha256"] or
                    session.capture.input_display_id != base.frame.display_id or
                    session.client_version != base.client_version):
                raise FlowError("Probe current frame/display changed before input")
            _foreground_now(session)
            if (_file_sha256(current.screenshot_path) != pre["sha256"] or
                    clock() >= stage_deadline or index == 0 and clock() >= base.latest_start):
                raise FlowError("Probe proof consumed its input deadline")
            session.check_deadline()
            if clock() >= stage_deadline or index == 0 and clock() >= base.latest_start:
                raise FlowError("Probe stopped before its input deadline")
            receipt["attempted_placements"] += 1
            receipt["probe_events"].append({"edge": index, "point": anchor,
                                            "before_count": 2 - index, "before": pre,
                                            "started_at": stage_start, "deadline": stage_deadline,
                                            "status": "attempted"})
            try:
                session.tap(current, anchor, reason=f"One frozen balloon anchor probe edge {index}")
            except BaseException:
                receipt["probe_events"][-1]["status"] = "uncertain"
                receipt["actions"].append({"action_id": f"probe-{index}", "kind": "deploy",
                                           "unit_id": "balloon", "point": anchor, "status": "uncertain"})
                raise
            done = clock()
            receipt["actions"].append({"action_id": f"probe-{index}", "kind": "deploy",
                                       "unit_id": "balloon", "point": anchor,
                                       "status": "input_sent", "done_at": done})
            receipt["probe_events"][-1]["status"] = "input_sent"
            receipt["probe_events"][-1]["input_done_at"] = done
            receipt["issued_placements"] += 1
            receipt["any_input_sent"] = True
            if done >= stage_deadline:
                raise FlowError("Probe input exceeded its six-second deadline")
            confirmed = None
            for observation in range(2):
                if clock() >= stage_deadline:
                    break
                # The two HUD anchors establish this frame's scene quickly;
                # _balloon_count independently binds portrait and number below.
                current = session.observe(f"probed-anchor-{index}-post-{observation}",
                                          purpose="battle_hud")
                receipt["evidence"].append(str(current.screenshot_path))
                if current.scene == "account_in_use":
                    raise FlowError("Game interruption: account_in_use")
                if current.observations.get("deployment_count_conflict") is not None:
                    raise FlowError("Probe strong_count_conflict in the post-input frame")
                scope = current.observations.get("recognition_scope")
                if scope == "battle_hud":
                    if (current.observations.get("fast_fallback") is not False or
                            current.observations.get("matched_reasons") != ["two_current_hud_anchors"]):
                        raise FlowError("Probe fast post frame lacks independent HUD scene proof")
                elif not (scope == "full" and current.observations.get("fast_fallback") is True and
                          current.observations.get("recognition_requested") == "battle_hud"):
                    raise FlowError("Probe post frame lacks a fresh full or guarded fast recognition")
                if current.scene not in {"enemy_village", "battle"}:
                    continue
                camera_error = None
                try:
                    post = _current_frame(session, plan, current, after=done, scene=current.scene)
                except ExactCameraError as exc:
                    if index != 1 or current.scene != "battle":
                        raise
                    # This is only a fresh same-display/count candidate. The
                    # camera failure never authorizes either old point.
                    post = getattr(exc, "frame_proof", None)
                    if (not isinstance(post, dict) or post.get("frame") != str(current.screenshot_path) or
                            _file_sha256(current.screenshot_path) != post.get("sha256") or
                            _file_sha256(base.frame.screenshot_path) != base.frame_sha256):
                        raise FlowError("Probe camera rejection lacks unchanged current frame proof") from exc
                    camera_error = exc
                try:
                    reading = _balloon_count(session, plan, current, 1 - index,
                                             source_bound=True, selected=True,
                                             pending=2 - index)
                except FlowError as count_error:
                    receipt["probe_events"][-1].setdefault("count_proof_rejections", []).append({
                        "observation_index": observation,
                        "frame": post["frame"], "sha256": post["sha256"],
                        "message": str(count_error),
                    })
                    if camera_error is not None:
                        raise  # Count failure is not an exact-camera-only recovery case.
                    if ("strong_count_conflict" in str(count_error) or
                            "differs from frozen" in str(count_error) or
                            "source balloon versioned identity" in str(count_error) or
                            "strong current-card identity conflicts" in str(count_error)):
                        raise
                    continue  # One more new frame, never one more input.
                if camera_error is not None:
                    receipt["probe_events"][-1]["camera_rejection"] = {
                        "observation_index": observation, "frame": post["frame"],
                        "sha256": post["sha256"], "message": str(camera_error),
                        "same_frame_count": reading["count"]}
                    if (reading["count"] != 0 or clock() >= stage_deadline or
                            clock() >= started + 12):
                        raise camera_error
                    receipt["probe_elapsed_sec"] = clock() - started
                    recovered_checkpoint = _recover_original_anchors(
                        session, plan, receipt, after=post["observed_at"], clock=clock)
                    current, recovered_proof, recovered_balloon, recovered_roster, authorization_deadline = recovered_checkpoint
                    verified_at = clock()
                    confirmed = (recovered_proof, recovered_balloon)
                    break
                if current.scene == "enemy_village":
                    if reading["count"] != 2 - index:
                        raise FlowError("Probe preview changed count without a positive battle scene")
                    continue  # A preview cannot authorize the next game input.
                if reading["count"] == 2 - index:
                    continue  # Game has not consumed this one tap yet; never replay it.
                verified_at = clock()
                if verified_at >= stage_deadline and recovered_checkpoint is None:
                    raise FlowError("Probe complete same-frame proof exceeded its deadline")
                confirmed = (post, reading)
                break
            if confirmed is None:
                raise FlowError("Probe lacks one fresh complete battle/count/camera proof")
            receipt["probe_events"][-1].update(status="verified_after_recovery" if recovered_checkpoint is not None
                                                else "verified", after=confirmed[0],
                                                after_count=1 - index, reading=confirmed[1],
                                                verified_at=verified_at)
            receipt["deployed_units"] = index + 1
        # The fast second-probe frame proves only that one balloon was consumed.
        # A separate, newer full frame must prove the entire remaining roster.
        session.check_deadline()
        if clock() >= authorization_deadline:
            raise FlowError("Probed anchor final full checkpoint missed the total budget")
        if recovered_checkpoint is None:
            current = session.observe("probed-anchor-final-checkpoint")
            receipt["evidence"].append(str(current.screenshot_path))
        # Ordinary full observation does not currently publish recognition_scope;
        # explicit fast scope/fallback is never enough for the roster checkpoint.
        if current.scene != "battle" or current.observations.get("recognition_scope") not in (None, "full") or \
                current.observations.get("fast_fallback") is True:
            if current.scene == "account_in_use":
                raise FlowError("Game interruption: account_in_use")
            raise FlowError("Probed anchor final checkpoint lacks an independent full battle frame")
        if recovered_checkpoint is None:
            final_proof = _current_frame(session, plan, current, after=verified_at, scene="battle")
            final_balloon = _balloon_count(session, plan, current, 0)
            receipt["checkpoint"] = _remaining_roster_checkpoint(session, plan, current)
        else:
            final_proof, final_balloon = recovered_proof, recovered_balloon
            receipt["checkpoint"] = recovered_roster
        receipt["final_checkpoint"] = {"frame": final_proof["frame"],
                                        "sha256": final_proof["sha256"],
                                        "observed_at": final_proof["observed_at"],
                                        "camera": final_proof["camera"],
                                        "balloon": final_balloon,
                                        "roster_sha256": receipt["checkpoint"]["sha256"],
                                        "verified_at": clock()}
        if (clock() >= authorization_deadline or clock() >= min(session.deadline, session.task_deadline) or
                current is not session.last_snapshot or
                _digest_plan(plan) != plan.digest or _plan_digest(base) != base.plan_sha256 or
                _file_sha256(current.screenshot_path) != final_proof["sha256"] or
                _file_sha256(base.frame.screenshot_path) != base.frame_sha256 or
                _file_sha256(base.toolbar_before_frame_id) != base.toolbar_before_sha256 or
                base.hero_manifest_frame_id is not None and
                _file_sha256(base.hero_manifest_frame_id) != base.hero_manifest_frame_sha256 or
                session.capture.input_display_id != base.frame.display_id or
                session.client_version != base.client_version):
            raise FlowError("Probed anchor final authorization expired or source changed")
        for event in receipt["probe_events"]:
            for key in ("before", "after"):
                if _file_sha256(event[key]["frame"]) != event[key]["sha256"]:
                    raise FlowError("Probed anchor source proof pixels changed")
        if receipt.get("recovery") is not None:
            for row in receipt["recovery"]["frames"]:
                if row["sha256"] is not None and _file_sha256(row["frame"]) != row["sha256"]:
                    raise FlowError("Probed anchor recovery proof pixels changed")
        session.check_deadline()
        if clock() >= authorization_deadline:
            raise FlowError("Probed anchor final proof hashing exceeded its deadline")
        _foreground_now(session)
        session.check_deadline()
        if (current is not session.last_snapshot or
                session.capture.input_display_id != base.frame.display_id or
                session.client_version != base.client_version or
                _digest_plan(plan) != plan.digest or _plan_digest(base) != base.plan_sha256 or
                _file_sha256(current.screenshot_path) != final_proof["sha256"] or
                _file_sha256(base.frame.screenshot_path) != base.frame_sha256 or
                _file_sha256(base.toolbar_before_frame_id) != base.toolbar_before_sha256 or
                base.hero_manifest_frame_id is not None and
                _file_sha256(base.hero_manifest_frame_id) != base.hero_manifest_frame_sha256):
            raise FlowError("Probed anchor final foreground proof or source changed")
        authorized_at = clock()  # All hashes and foreground proof have now finished.
        if authorized_at >= authorization_deadline:
            raise FlowError("Probed anchor foreground proof exceeded total budget")
        receipt["probe_authorized_at"] = authorized_at
        if recovered_checkpoint is None:
            receipt["probe_elapsed_sec"] = authorized_at - started
        else:
            receipt["recovery"].update(status="authorized", authorized_at=authorized_at,
                                        total_authorization_elapsed_sec=authorized_at - started)
        # No scene, identity or count recognition occurs inside this remainder.
        start = clock()
        limit = remainder_burst_limit(plan)
        deadline = min(start + limit, session.deadline, session.task_deadline)
        receipt["burst_limit_sec"] = limit
        for action in plan.remaining:
            session.check_deadline()
            remaining = deadline - clock()
            if remaining <= 0:
                raise FlowError("Probed anchor remaining batch deadline exceeded")
            if action.kind == "deploy":
                receipt["attempted_placements"] += 1
            session._active_frozen_plan = (base.plan_id, deadline, action.point)
            try:
                session.frozen_battle_tap(action.point,
                    timeout_sec=min(remaining, session.config.runtime.step_timeout_sec),
                    reason=f"probed anchor {base.plan_id} {action.action_id}", plan_id=base.plan_id)
            except BaseException:
                receipt["actions"].append({"action_id": action.action_id, "kind": action.kind,
                                           "unit_id": action.unit_id, "point": action.point,
                                           "status": "uncertain"})
                raise
            finally:
                session._active_frozen_plan = None
            done = clock()
            receipt["actions"].append({"action_id": action.action_id, "kind": action.kind,
                                       "unit_id": action.unit_id, "point": action.point,
                                       "status": "input_sent", "done_at": done})
            receipt["issued_placements"] += int(action.kind == "deploy")
            if done >= deadline:
                raise FlowError("Probed anchor remaining batch exceeded its deadline")
            if action.delay_after_sec:
                if action.delay_after_sec >= deadline - clock():
                    raise FlowError("Probed anchor fixed delay exceeds batch deadline")
                sleep(action.delay_after_sec)
        receipt["burst_elapsed_sec"] = clock() - start
        receipt["burst_pass"] = receipt["burst_elapsed_sec"] <= limit
        receipt["input_sent"] = receipt["burst_pass"] and len(receipt["actions"]) == receipt["input_count"]
        if not receipt["input_sent"]:
            raise FlowError("Probed anchor full action receipt is incomplete")
        return _verify_frozen_line_post(session, base, receipt, receipt["actions"][-1]["done_at"], sleep=sleep)
    except StopRequested:
        receipt["probe_elapsed_sec"] = receipt["probe_elapsed_sec"] or clock() - started
        raise
    except DeploymentError as exc:
        receipt["probe_elapsed_sec"] = receipt["probe_elapsed_sec"] or clock() - started
        receipt["error"] = str(exc)
        raise
    except Exception as exc:
        receipt["probe_elapsed_sec"] = receipt["probe_elapsed_sec"] or clock() - started
        receipt["error"] = str(exc)
        if "account_in_use" in str(exc):
            receipt["interruption"] = "account_in_use"
        raise DeploymentError(str(exc), partial_receipt=receipt) from exc
    finally:
        session._active_frozen_plan = None
