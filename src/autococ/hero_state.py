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


def recognize_siege_state(
    screenshot_path: str | Path, slot_bbox: tuple[int, int, int, int] | list[int], *,
    baseline_resolution: tuple[int, int] = (1280, 720),
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
    if health is not None:
        result.update(state="deployed", deployed=True, defeated=False)
    elif selected is not None:
        result["state"] = "selected"
    # A pet can be reassigned: the Warden-specific lower-card and book images must
    # both match before interpreting this pet-equipped card as this ready layout.
    if health is not None and pet is not None and Path(pet["template"]).name == "battle_hero_pet_0.png" and (unit_id is None or unit_id == "grand_warden"):
        active = _match(image, TEMPLATES / "hero_warden_active_book.png", (x0, 510, x1, 573))
        glows = [_match(image, TEMPLATES / name, (x0, 620, x1, 690))
                 for name in ("hero_warden_ready_card.png", "hero_warden_ready_card_phase.png")]
        glow = max((item for item in glows if item is not None), key=lambda item: item["confidence"], default=None)
        used_book = _match(image, TEMPLATES / "hero_warden_used_book.png", (x0, 510, x1, 573))
        used_card = _match(image, TEMPLATES / "hero_warden_used_card.png", (x0, 620, x1, 690))
        if active is not None and glow is not None:
            result.update(state="ability_ready", ability_ready=True, ability_used=False)
            evidence["ability_ready"] = {
                "hero": "grand_warden", "equipment_layout": "observed_gold_book",
                "active_equipment": active, "card_book_glow": glow,
                "template_source": WARDEN_TEMPLATE_SOURCE,
                "card_template_source": WARDEN_READY_PHASE_SOURCE if Path(glow["template"]).name.endswith("_phase.png") else WARDEN_TEMPLATE_SOURCE,
            }
        elif used_book is not None and used_card is not None:
            result.update(state="ability_used", ability_ready=False, ability_used=True)
            evidence["ability_used"] = {
                "hero": "grand_warden", "equipment_layout": "observed_gold_book",
                "gray_equipment": used_book, "gray_card_book": used_card,
                "template_source": WARDEN_USED_TEMPLATE_SOURCE,
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
