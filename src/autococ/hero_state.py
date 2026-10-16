"""Local evidence for previously located support cards, without issuing actions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .images import read_frame, read_template
from .errors import SceneError
from .locator import scale_box


TEMPLATES = Path(__file__).resolve().parents[2] / "assets/templates"
# Both Warden ability templates come from the stable, deployed frame below.
# They describe this equipped gold-book layout, not arbitrary heroes/equipment.
WARDEN_TEMPLATE_SOURCE = "reports/live-20260922/hero-state-023106/frames/00004-hero-placement-after.png"
WARDEN_TEMPLATE_CROPS = {
    "hero_warden_active_book.png": (600, 526, 642, 568),
    "hero_warden_ready_card.png": (603, 631, 685, 680),
}
WARDEN_READY_PHASE_SOURCE = "reports/20260924-004347-675110-2e2b80bd/frames/00027-deploy-hero-verify.png"
WARDEN_READY_PHASE_CROP = (506, 631, 588, 680)
WARDEN_USED_TEMPLATE_SOURCE = "reports/live-20260922/warden-activate-023930/frames/00004-after-ability.png"
WARDEN_USED_TEMPLATE_CROPS = {
    "hero_warden_used_book.png": (600, 526, 642, 568),
    "hero_warden_used_card.png": (603, 631, 685, 680),
}
WARDEN_PORTRAIT_CROP = (639, 594, 681, 646)
SIEGE_TEMPLATE_SOURCE = "reports/20260923-021847-763016-3632dc81/frames/00015-deploy-support-selected.png"
SIEGE_TEMPLATE_CROP = (531, 658, 576, 701)
HERO_ABILITY_SAMPLE_DIRECTORY = "reports/hero-ability-samples-20260924-015325-792650/frames"
HERO_ABILITY_SAMPLES = {
    "battle_hero_pet_1.png": ("barbarian_king", 599, "00014-hero-deployed.png", "00016-ability-after.png"),
    "battle_hero_pet_2.png": ("minion_prince", 696, "00020-hero-deployed.png", "00022-ability-after.png"),
    "battle_hero_pet_3.png": ("archer_queen", 792, "00026-hero-deployed.png", "00028-ability-after.png"),
}
# Crop x coordinates are relative to the sampled card's left edge; y is baseline.
HERO_ABILITY_SAMPLE_CROPS = {
    "ready_equipment": (0, 528, 42, 569), "ready_portrait": (40, 594, 82, 646),
    "used_portrait": (40, 594, 82, 646), "used_weapon": (35, 647, 82, 679),
}
HERO_PHASE_TEMPLATE_IDS = {
    "grand_warden": "warden", "barbarian_king": "barbarian_king",
    "minion_prince": "minion_prince", "archer_queen": "archer_queen",
}
CURRENT_WARDEN_SKILL_SOURCE = "reports/completion-20261015/probes/20261016-035742-manual-probed-skill/frames"
CURRENT_WARDEN_READY_CARD_PHASE_SOURCE = (
    "reports/completion-20261015/live/20261016-053200-457276-365d1d63/"
    "frames/00028-post-deployment-skill-before.png"
)


def recognize_siege_state(
    screenshot_path: str | Path, slot_bbox: tuple[int, int, int, int] | list[int], *,
    baseline_resolution: tuple[int, int] = (1280, 720),
    client_version: str | None = None,
) -> dict[str, object]:
    """An observed release control plus independent HP verifies deployment."""
    import cv2
    import numpy as np

    path = Path(screenshot_path)
    source = read_frame(path, cv2.IMREAD_COLOR)
    if source is None:
        raise SceneError(f"Unable to decode siege screenshot: {path}")
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    left, top, right, bottom = scale_box(tuple(slot_bbox), from_resolution=baseline_resolution,
                                       to_resolution=(1280, 720))
    result = {"state": "unknown", "deployed": None, "frame": str(path), "slot_bbox": list(slot_bbox),
              "evidence": {"coordinate_resolution": [1280, 720], "template_source": SIEGE_TEMPLATE_SOURCE}}
    if not (0 <= left < right <= 1280 and 80 <= right - left <= 110
            and 550 <= top <= 605 and 695 <= bottom <= 720):
        return result
    if client_version != "18.600.7":
        result["evidence"]["reason"] = "current_client_siege_deployment_templates_unavailable"
        return result
    release = _match(image, TEMPLATES / "siege_deployed_release.png",
                     (max(0, left - 10), 645, min(1280, right + 10), 720), mask_kind="release")
    hsv, gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    health = _health_bar(hsv, gray, [left + 7, 591])
    if health is None:
        # The observed smaller deployed card has a 72-pixel enclosed HP bar.
        health = _health_bar(hsv, gray, [left + 10, 591], frame_width=72)
    result["evidence"].update(release_control=release, health_bar=health)
    if release is not None and health is not None:
        result.update(state="deployed", deployed=True)
    return result


def recognize_hero_state(
    screenshot_path: str | Path,
    slot_bbox: tuple[int, int, int, int] | list[int],
    *,
    baseline_resolution: tuple[int, int] = (1280, 720),
    unit_id: str | None = None,
    client_version: str | None = None,
) -> dict[str, object]:
    """Inspect a known hero slot; absent evidence is unknown, never consumption.

    Ability evidence covers only the sampled hero portraits and equipment layouts.
    Defeated has no positive fixture: visible HP proves alive; otherwise unknown.
    A selected white border alone never proves deployment or ability readiness.
    When unit_id is supplied, its portrait must match again in this frame.
    """
    import cv2
    import numpy as np

    path = Path(screenshot_path)
    source = read_frame(path, cv2.IMREAD_COLOR)
    if source is None:
        raise SceneError(f"Unable to decode hero screenshot: {path}")
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    box = scale_box(tuple(slot_bbox), from_resolution=baseline_resolution, to_resolution=(1280, 720))
    left, top, right, bottom = box
    result = {
        "state": "unknown", "selected": None, "deployed": None, "ability_ready": None,
        "ability_used": None, "defeated": None, "frame": str(path), "slot_bbox": list(slot_bbox),
        "evidence": {"coordinate_resolution": [1280, 720], "unsupported_positive_states": ["defeated"]},
    }
    evidence = result["evidence"]
    if not (0 <= left < right <= 1280 and 80 <= right - left <= 110
            and 550 <= top <= 605 and 695 <= bottom <= 720):
        evidence["reason"] = "unrecognized_hero_slot_geometry"
        return result
    if not isinstance(unit_id, str) or not unit_id or client_version not in {"18.600.7", "18.647.2"}:
        evidence["reason"] = "current_frame_hero_identity_unverified"
        return result
    if unit_id is not None:
        from .unit_catalog import recognize_card_identity

        identity = recognize_card_identity(image, box, "hero", surface="battle",
                                           client_version=client_version)
        evidence["portrait_identity"] = identity
        if not identity["version_verified"] or (identity["unit_id"] is not None
                                                and identity["unit_id"] != unit_id):
            evidence["reason"] = "current_frame_hero_identity_unverified"
            return result
        if identity["unit_id"] is None:
            # Phase portraits were sampled on the older client only. A known
            # army portrait in another version does not validate a battle card.
            if client_version != "18.600.7":
                evidence["reason"] = "current_frame_hero_identity_unverified"
                return result
            phase_portrait = _hero_phase_portrait(image, box, unit_id)
            evidence["phase_portrait"] = phase_portrait
            if phase_portrait is None:
                evidence["reason"] = "current_frame_hero_identity_unverified"
                return result
    roi = (max(0, left - 15), 560, min(1280, left + 50), 650)
    pets = [_match(image, TEMPLATES / f"battle_hero_pet_{index}.png", roi, (1.0, 1.05, 1.1), mask_kind="pet")
            for index in range(4)]
    pet = max((item for item in pets if item is not None), key=lambda item: item["confidence"], default=None)
    if pet is None and unit_id is None:
        evidence["reason"] = "hero_card_identity_anchor_missing"
        return result
    if pet is not None:
        evidence["pet_anchor"] = pet
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    x0, x1 = max(0, left - 12), min(1280, right + 12)
    white = cv2.inRange(hsv[565:720, x0:x1], np.array([0, 0, 220]), np.array([180, 35, 255]))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected = None
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        border = (x + x0, y + 565, x + x0 + width, y + 565 + height)
        if (80 <= width <= 108 and 105 <= height <= 135
                and abs((border[0] + border[2] - left - right) / 2) <= 7
                and cv2.contourArea(contour) / (width * height) >= .85):
            selected = {"bbox": list(border), "method": "closed_white_selected_card_border"}
            break
    result["selected"] = selected is not None
    evidence["selected_border"] = selected
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    health = _health_bar(hsv, gray, pet["bbox"]) if pet is not None else None
    if health is None and unit_id is not None:
        # A deployed card's enclosed HP frame is anchored to the card, while
        # the pet icon may be absent or reassigned. Search only inside this card.
        for py in range(592, 606):
            for offset, width in ((5, 78), (8, 72)):
                if left + offset + 1 + width > right + 2:
                    continue
                health = _health_bar(hsv, gray, [left + offset, py], frame_width=width)
                if health is not None:
                    break
            if health is not None:
                break
    evidence["health_bar"] = health
    yellow_health = None
    if client_version == "18.647.2" and unit_id == "grand_warden" and health is None:
        # This client's sampled Warden card can show yellow HP after its
        # initial green phase. Recheck the portrait and closed card in this
        # exact frame before accepting colour inside the card as live HP.
        located = locate_current_deployed_hero_card(path, unit_id, client_version=client_version)
        evidence["closed_card"] = located
        if located is not None and tuple(located["bbox"]) == box:
            for py in range(592, 606):
                for offset, width in ((5, 78), (8, 72)):
                    if left + offset + 1 + width > right + 2:
                        continue
                    yellow_health = _yellow_health_bar(hsv, gray, [left + offset, py], frame_width=width)
                    if yellow_health is not None:
                        break
                if yellow_health is not None:
                    break
    evidence["yellow_health_bar"] = yellow_health
    if health is not None or yellow_health is not None:
        result.update(state="deployed", deployed=True, defeated=False)
    elif selected is not None:
        result["state"] = "selected"
    if client_version == "18.647.2":
        # The current equipped book has its own paired ready/used samples.
        # A gray face by itself is only an identity candidate, never an ability
        # transition. Require the live HP frame and the independently closed
        # card located again in this same screenshot.
        if unit_id != "grand_warden" or health is None:
            evidence["ability_phase_reason"] = "current_warden_equipment_or_live_hp_unverified"
            return result
        located = evidence.get("closed_card") or locate_current_deployed_hero_card(
            path, unit_id, client_version=client_version)
        evidence["closed_card"] = located
        if located is None or tuple(located["bbox"]) != box:
            evidence["ability_phase_reason"] = "current_warden_card_not_independently_closed"
            return result
        ready_card_roi = (left + 28, 621, right - 2, 685)
        ready_cards = [_match(image, TEMPLATES / name, ready_card_roi)
                       for name in ("hero_warden_18_647_2_ready_card.png",
                                    "hero_warden_18_647_2_ready_card_phase.png")]
        anchors = {
            "ready_book": _match(image, TEMPLATES / "hero_warden_18_647_2_ready_book.png",
                                 (left + 1, 525, left + 48, 572)),
            "ready_card": max((match for match in ready_cards if match is not None),
                              key=lambda match: match["confidence"], default=None),
            "used_book": _match(image, TEMPLATES / "hero_warden_18_647_2_used_book.png",
                                (left + 1, 525, left + 48, 572)),
            "used_card": _match(image, TEMPLATES / "hero_warden_18_647_2_used_card.png",
                                (left + 28, 621, right - 2, 685)),
        }
        evidence["current_equipment_matches"] = anchors
        ready = anchors["ready_book"] is not None and anchors["ready_card"] is not None
        used = anchors["used_book"] is not None and anchors["used_card"] is not None
        mixed = ((anchors["ready_book"] is not None or anchors["ready_card"] is not None)
                 and (anchors["used_book"] is not None or anchors["used_card"] is not None))
        if ready and not mixed:
            result.update(state="ability_ready", ability_ready=True, ability_used=False)
            ready_template = Path(anchors["ready_card"]["template"]).name
            ready_source = (CURRENT_WARDEN_READY_CARD_PHASE_SOURCE
                            if ready_template == "hero_warden_18_647_2_ready_card_phase.png"
                            else f"{CURRENT_WARDEN_SKILL_SOURCE}/00025-manual-skill-fresh-action-check.png")
            evidence["ability_ready"] = {"hero": unit_id, "equipment_layout": "current_gold_book",
                                         "active_equipment": anchors["ready_book"],
                                         "ready_card": anchors["ready_card"],
                                         "ready_book_source": f"{CURRENT_WARDEN_SKILL_SOURCE}/00025-manual-skill-fresh-action-check.png",
                                         "ready_card_source": ready_source,
                                         "template_source": ready_source,
                                         "source_kind": "agent_visual_prototype"}
        elif used and not mixed and "battle_used_035742" in str(identity["evidence"][0]["template"]):
            result.update(state="ability_used", ability_ready=False, ability_used=True)
            evidence["ability_used"] = {"hero": unit_id, "equipment_layout": "current_gold_book",
                                        "gray_equipment": anchors["used_book"],
                                        "gray_card": anchors["used_card"],
                                        "template_source": f"{CURRENT_WARDEN_SKILL_SOURCE}/00027-manual-skill-after-2.png",
                                        "source_kind": "agent_visual_prototype"}
        else:
            evidence["ability_phase_reason"] = "current_warden_book_card_or_identity_conflict"
        return result
    if client_version != "18.600.7":
        # The ability portraits and equipment below were sampled only on the
        # older client. Current-card identity/HP may be known independently,
        # but neither authorizes a ready or used ability phase on 18.647.2.
        evidence["ability_phase_reason"] = "current_client_ability_templates_unavailable"
        return result
    # A pet can be reassigned: the Warden-specific lower-card and book images must
    # both match before interpreting this pet-equipped card as this ready layout.
    if pet is not None and Path(pet["template"]).name == "battle_hero_pet_0.png" and (unit_id is None or unit_id == "grand_warden"):
        active = _match(image, TEMPLATES / "hero_warden_active_book.png", (x0, 510, x1, 573))
        glows = [_match(image, TEMPLATES / name, (x0, 620, x1, 690))
                 for name in ("hero_warden_ready_card.png", "hero_warden_ready_card_phase.png")]
        glow = max((item for item in glows if item is not None), key=lambda item: item["confidence"], default=None)
        used_book = _match(image, TEMPLATES / "hero_warden_used_book.png", (x0, 510, x1, 573))
        used_card = _match(image, TEMPLATES / "hero_warden_used_card.png", (x0, 620, x1, 690))
        if health is not None and active is not None and glow is not None:
            result.update(state="ability_ready", ability_ready=True, ability_used=False)
            evidence["ability_ready"] = {
                "hero": "grand_warden", "equipment_layout": "observed_gold_book",
                "active_equipment": active, "card_book_glow": glow,
                "template_source": WARDEN_TEMPLATE_SOURCE,
                "card_template_source": WARDEN_READY_PHASE_SOURCE if Path(glow["template"]).name.endswith("_phase.png") else WARDEN_TEMPLATE_SOURCE,
            }
        elif health is not None and used_book is not None and used_card is not None:
            result.update(state="ability_used", ability_ready=False, ability_used=True)
            evidence["ability_used"] = {
                "hero": "grand_warden", "equipment_layout": "observed_gold_book",
                "gray_equipment": used_book, "gray_card_book": used_card,
                "template_source": WARDEN_USED_TEMPLATE_SOURCE,
            }
        elif health is None and unit_id == "grand_warden":
            # Low health can remove the green fill before the next observation.
            # Three independent current-card anchors must still prove this exact
            # ready layout; a red bar or a selected border alone proves nothing.
            phase_portrait = evidence.get("phase_portrait") or _hero_phase_portrait(image, box, unit_id)
            evidence["phase_portrait"] = phase_portrait
            if (phase_portrait is not None and phase_portrait["phase"] == "ready"
                    and active is not None and glow is not None
                    and used_book is None and used_card is None):
                result.update(state="ability_ready", deployed=True, ability_ready=True, ability_used=False)
                evidence["ability_ready"] = {
                    "hero": "grand_warden", "equipment_layout": "observed_gold_book_low_health",
                    "phase_portrait": phase_portrait, "active_equipment": active, "card_book_glow": glow,
                    "gray_equipment": used_book, "gray_card_book": used_card,
                    "template_source": WARDEN_TEMPLATE_SOURCE,
                    "card_template_source": WARDEN_READY_PHASE_SOURCE if Path(glow["template"]).name.endswith("_phase.png") else WARDEN_TEMPLATE_SOURCE,
                }
    elif health is not None and pet is not None and Path(pet["template"]).name in HERO_ABILITY_SAMPLES:
        hero, _, ready_frame, used_frame = HERO_ABILITY_SAMPLES[Path(pet["template"]).name]
        if unit_id is not None and unit_id != hero:
            return result
        # Pets are only a search hint. Independent hero-specific portrait and
        # equipment/weapon evidence must agree; gray cards without HP stay unknown.
        regions = {"ready_equipment": (x0, 510, x1, 573), "ready_portrait": (x0, 585, x1, 657),
                   "used_portrait": (x0, 585, x1, 657), "used_weapon": (x0, 637, x1, 690)}
        matches = {name: _match(image, TEMPLATES / f"hero_{hero}_{name}.png", roi)
                   for name, roi in regions.items()}
        ready = matches["ready_equipment"] is not None and matches["ready_portrait"] is not None
        used = matches["used_portrait"] is not None and matches["used_weapon"] is not None
        if ready and used:
            evidence["ability_conflict"] = matches
        elif ready:
            result.update(state="ability_ready", ability_ready=True, ability_used=False)
            evidence["ability_ready"] = {
                "hero": hero, "equipment_layout": "observed_sample",
                "active_equipment": matches["ready_equipment"], "portrait": matches["ready_portrait"],
                "template_source": f"{HERO_ABILITY_SAMPLE_DIRECTORY}/{ready_frame}",
            }
        elif used:
            result.update(state="ability_used", ability_ready=False, ability_used=True)
            evidence["ability_used"] = {
                "hero": hero, "equipment_layout": "observed_sample",
                "gray_portrait": matches["used_portrait"], "gray_weapon": matches["used_weapon"],
                "template_source": f"{HERO_ABILITY_SAMPLE_DIRECTORY}/{used_frame}",
            }
    return result


def locate_current_deployed_hero_card(
    screenshot_path: str | Path, unit_id: str, *, client_version: str | None,
) -> dict[str, object] | None:
    """Locate a deployed card from this frame's versioned face and card edges.

    A pre-burst slot is deliberately not an input: post-burst animation may
    change the card art or outline, so its old coordinates are only history.
    This locates a card; ``recognize_hero_state`` must still prove HP/deployment.
    """
    import cv2
    import math
    import numpy as np

    if client_version != "18.647.2":
        return None
    from .unit_catalog import CATALOG_ROOT, template_manifest

    image = cv2.resize(read_frame(screenshot_path), (1280, 720), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([75, 65, 60]), np.array([170, 255, 255]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = []
    for entry in template_manifest(client_version=client_version)["templates"]:
        if (entry.get("unit_id") != unit_id or entry.get("surface") != "battle" or
                entry.get("state") != "deployed" or entry.get("client_version") != client_version):
            continue
        portrait_box, card_box = entry.get("source_bbox_1280"), entry.get("source_card_bbox_1280")
        if (not isinstance(portrait_box, list) or len(portrait_box) != 4 or
                not isinstance(card_box, list) or len(card_box) != 4 or
                any(type(v) is not int for v in portrait_box + card_box)):
            continue
        template_path = (CATALOG_ROOT / entry["path"]).resolve()
        if not template_path.is_relative_to(CATALOG_ROOT.resolve()) or not template_path.is_file():
            continue
        template = cv2.imdecode(np.fromfile(template_path, np.uint8), cv2.IMREAD_COLOR)
        search = image[620:690, 550:1010]
        if template is None or template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
            continue
        _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED))
        if not math.isfinite(score) or score < .94:
            continue
        left = 550 + x - (portrait_box[0] - card_box[0])
        top = 620 + y - (portrait_box[1] - card_box[1])
        right, bottom = left + card_box[2] - card_box[0], top + card_box[3] - card_box[1]
        if not (3 <= left < right <= 1277 and 590 <= top <= 605 and 705 <= bottom <= 715):
            continue
        # Independent same-frame card sides and closed bottom prevent a face
        # in battlefield art, a transient fade, or a stale slot from locating it.
        sides = [max(float((mask[625:695, edge + shift:edge + shift + 3] > 0).mean())
                     for shift in (-2, -1, 0, 1, 2)) for edge in (left, right - 3)]
        bottom_contrast = float(gray[709:711, left + 8:right - 8].mean() -
                                gray[712, left + 8:right - 8].mean())
        gray_used_border = None
        if (unit_id == "grand_warden" and "battle_used_035742" in template_path.name):
            # The used card loses its purple side colour. Its opposite dark
            # outline gradients and closed bottom remain visible in 27-29.
            direction = np.median(cv2.Sobel(gray[625:695], cv2.CV_32F, 1, 0, ksize=3), axis=0)
            gray_used_border = [float(max(direction[left - 2:left + 3])),
                                float(min(direction[right - 4:right]))]
            closed = gray_used_border[0] >= 250 and gray_used_border[1] <= -250
        else:
            closed = min(sides) >= .5
        if not closed or bottom_contrast < 40:
            continue
        candidates.append((score, (left, top, right, bottom), sides, bottom_contrast,
                           str(template_path), gray_used_border))
    if not candidates:
        return None
    score, box, sides, bottom_contrast, template_path, gray_used_border = max(candidates, key=lambda row: row[0])
    return {"bbox": list(box), "portrait_score": round(score, 5), "template": template_path,
            "template_client_version": client_version, "frame": str(screenshot_path),
            "card_sides": sides, "gray_used_border": gray_used_border,
            "bottom_contrast": bottom_contrast,
            "method": "current_versioned_deployed_portrait_and_closed_card"}


def _hero_phase_portrait(image, box: tuple[int, int, int, int], unit_id: str) -> dict | None:
    """Match a hero-specific deployed/used face in this card, never its pet."""
    if unit_id not in HERO_PHASE_TEMPLATE_IDS:
        return None
    left, _, right, _ = box
    roi = (left + 35, 590, right - 5, 650)
    template_hero = HERO_PHASE_TEMPLATE_IDS[unit_id]
    for phase in ("ready", "used"):
        match = _match(image, TEMPLATES / f"hero_{template_hero}_{phase}_portrait.png", roi)
        if match is not None:
            return {"hero": unit_id, "phase": phase, **match}
    return None


def _match(image, template_path: Path, roi: tuple[int, int, int, int], scales=(1.0,), *,
           mask_kind: Literal["release", "pet"] | None = None) -> dict | None:
    import cv2
    import numpy as np

    if not template_path.is_file():
        return None
    template = read_template(template_path)
    if template is None:
        return None
    mask = None
    if mask_kind == "release":
        hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        # The green release arrow and white passenger remain opaque over the map.
        mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([35, 60, 130]), np.array([90, 255, 255])),
                             cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 45, 255])))
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
    elif mask_kind == "pet":
        height, width = template.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (width // 2, height // 2), (width // 2 - 3, height // 2 - 3), 0, 0, 360, 255, -1)
    left, top, right, bottom = roi
    region = image[top:bottom, left:right]
    best = None
    for scale in scales:
        candidate = read_template(template_path, scale)
        height, width = candidate.shape[:2]
        if height > region.shape[0] or width > region.shape[1]:
            continue
        candidate_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) if mask is not None else None
        scores = cv2.matchTemplate(region, candidate, cv2.TM_CCOEFF_NORMED, mask=candidate_mask)
        scores[~np.isfinite(scores)] = -1
        _, score, _, location = cv2.minMaxLoc(scores)
        if score >= .9 and (best is None or score > best["confidence"]):
            x, y = location
            best = {"template": str(template_path), "confidence": round(score, 5), "scale": scale,
                    "bbox": [left + x, top + y, left + x + width, top + y + height]}
            if mask_kind is not None:
                best["comparison"] = f"opaque_{mask_kind}_foreground"
    return best


def _health_bar(hsv, gray, pet_bbox: list[int], *, frame_width: int = 78) -> dict | None:
    """Require a horizontal green fill enclosed by a dark card HP-bar frame."""
    import cv2
    import numpy as np

    px, py = pet_bbox[:2]
    left, top, right, bottom = px + 1, py - 22, px + 1 + frame_width, py - 1
    if left < 1 or right >= hsv.shape[1] or top < 3:
        return None
    green = cv2.inRange(hsv[top:bottom, left:right], np.array([35, 140, 180]), np.array([85, 255, 255]))
    _, _, stats, _ = cv2.connectedComponentsWithStats(green)
    for x, y, width, height, area in stats[1:]:
        if not (15 <= width <= 76 and 4 <= height <= 11 and area / (width * height) >= .85 and x <= 6):
            continue
        gx, gy = left + int(x), top + int(y)
        top_support = max(float((gray[row, left + 1:right - 2] < 65).mean()) for row in range(gy - 3, gy))
        bottom_support = max(float((gray[row, left + 1:right - 2] < 65).mean()) for row in range(gy + height, gy + height + 3))
        sides = [max(float((gray[gy:gy + height, column] < 65).mean()) for column in columns)
                 for columns in (range(left - 1, left + 3), range(right - 5, right + 1))]
        if min(top_support, bottom_support, *sides) < .85:
            continue
        return {"method": "green_fill_enclosed_by_dark_hp_frame", "fill_bbox": [gx, gy, gx + int(width), gy + int(height)],
                "frame_span": [left, gy - 3, right, gy + int(height) + 3],
                "frame_support": [top_support, bottom_support, *sides]}
    return None


def _yellow_health_bar(hsv, gray, pet_bbox: list[int], *, frame_width: int = 78) -> dict | None:
    """The sampled current Warden's yellow fill inside its own dark HP frame."""
    import cv2
    import numpy as np

    px, py = pet_bbox[:2]
    left, top, right, bottom = px + 1, py - 22, px + 1 + frame_width, py - 1
    if left < 1 or right >= hsv.shape[1] or top < 3:
        return None
    yellow = cv2.inRange(hsv[top:bottom, left:right], np.array([13, 170, 130]),
                         np.array([29, 255, 255]))
    _, _, stats, _ = cv2.connectedComponentsWithStats(yellow)
    for x, y, width, height, area in stats[1:]:
        if not (15 <= width <= 76 and 4 <= height <= 11 and area / (width * height) >= .85 and x <= 6):
            continue
        gx, gy = left + int(x), top + int(y)
        top_support = max(float((gray[row, left + 1:right - 2] < 65).mean()) for row in range(gy - 3, gy))
        bottom_support = max(float((gray[row, left + 1:right - 2] < 65).mean())
                             for row in range(gy + height, gy + height + 3))
        sides = [max(float((gray[gy:gy + height, column] < 65).mean()) for column in columns)
                 for columns in (range(left - 1, left + 3), range(right - 5, right + 1))]
        if min(top_support, bottom_support, *sides) < .85:
            continue
        return {"method": "yellow_fill_enclosed_by_dark_hp_frame",
                "fill_bbox": [gx, gy, gx + int(width), gy + int(height)],
                "frame_span": [left, gy - 3, right, gy + int(height) + 3],
                "frame_support": [top_support, bottom_support, *sides],
                "fill_density": round(float(area / (width * height)), 5)}
    return None
