"""Positive image evidence for earned stars and the observed reward panel."""

from pathlib import Path


TEMPLATES = Path(__file__).resolve().parents[2] / "assets/templates"
BONUS_TEMPLATE_SOURCE = "reports/20260924-004347-675110-2e2b80bd/frames/00061-battle-settlement-reread.png"
BONUS_TEMPLATE_CROPS = {
    "gold": (1007, 349, 1036, 381),
    "elixir": (1007, 385, 1036, 418),
    "dark_elixir": (1007, 421, 1036, 454),
}


def locate_number_line(image, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Tighten an icon-anchored white numeric row without guessing its value."""
    import cv2
    import numpy as np

    left, top, right, bottom = roi
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 80, 255]))
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    # Long one-pixel separators are not glyphs; other substantial components
    # must all fit the same row, rather than silently dropping an extra digit.
    glyphs = [row for row in components[1:] if row[4] >= 8 and row[3] > 3]
    if not glyphs or any(not (2 <= row[2] <= 42 and 12 <= row[3] <= 36) for row in glyphs):
        return None
    if max(row[1] for row in glyphs) - min(row[1] for row in glyphs) > 5:
        return None
    if max(row[3] for row in glyphs) - min(row[3] for row in glyphs) > 5:
        return None
    x1, y1 = min(row[0] for row in glyphs), min(row[1] for row in glyphs)
    x2 = max(row[0] + row[2] for row in glyphs)
    y2 = max(row[1] + row[3] for row in glyphs)
    if x1 < 2 or y1 < 2 or x2 > right - left - 2 or y2 > bottom - top - 2 or right - left - x2 > 20:
        return None
    return tuple(map(int, (left + x1 - 2, top + y1 - 2, left + x2 + 2, top + y2 + 2)))


def recognize_earned_stars(image) -> dict:
    """Count complete bright five-point silhouettes at the 1280x720 baseline."""
    import cv2
    import numpy as np

    roi = (430, 65, 845, 220)
    left, top, right, bottom = roi
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 75, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    evidence = {"method": "complete_five_concave_star_silhouettes", "roi_at_1280x720": list(roi), "shapes": []}
    valid = True
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 600:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        shape = {"bbox_at_1280x720": [left + x, top + y, left + x + width, top + y + height],
                 "area": area, "verified": False}
        evidence["shapes"].append(shape)
        if not (80 <= width <= 170 and 80 <= height <= 150 and .75 <= width / height <= 1.35):
            valid = False
            continue
        solidity = area / cv2.contourArea(cv2.convexHull(contour))
        # Antialiased highlights can add small vertices to a complete star.
        # A slightly coarser outline keeps its five deep concavities intact.
        polygon = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        if not 10 <= len(polygon) <= 12:
            shape.update(solidity=solidity, polygon_vertices=len(polygon))
            valid = False
            continue
        defects = cv2.convexityDefects(polygon, cv2.convexHull(polygon, returnPoints=False))
        depths = [] if defects is None else [float(item[0][3]) / 256 for item in defects]
        deep_concavities = sum(depth >= min(width, height) * .09 for depth in depths)
        complete = (.35 <= area / (width * height) <= .6 and .5 <= solidity <= .8
                    and 10 <= len(polygon) <= 12 and deep_concavities == 5)
        shape.update(verified=complete, solidity=solidity, polygon_vertices=len(polygon), concavity_depths=depths)
        valid = valid and complete
    shapes = evidence["shapes"]
    # An unexplained large fragment may be a second, occluded star. Never silently
    # discard it and report only the easier, fully visible stars.
    count = len(shapes) if valid and 1 <= len(shapes) <= 3 else None
    return {"count": count, "evidence": evidence}


def recognize_bonus_icons(image) -> dict:
    """Locate the three distinct resource icons in the observed right panel."""
    import cv2
    import numpy as np

    left, top, right, bottom = 1002, 345, 1043, 459
    icons = {}
    for name in BONUS_TEMPLATE_CROPS:
        path = TEMPLATES / f"settlement_bonus_{name}.png"
        if not path.is_file():
            continue
        template = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if template is None:
            continue
        scores = cv2.matchTemplate(image[top:bottom, left:right], template, cv2.TM_CCOEFF_NORMED)
        scores[~np.isfinite(scores)] = -1
        _, confidence, _, point = cv2.minMaxLoc(scores)
        if confidence < .9:
            continue
        x, y = point
        others = scores.copy()
        others[max(0, y - 15):y + 16, :] = -1
        if float(others.max()) >= .9:
            continue
        icons[name] = {"bbox_at_1280x720": [left + x, top + y, left + x + template.shape[1], top + y + template.shape[0]],
                       "confidence": float(confidence), "template": str(path), "template_source": BONUS_TEMPLATE_SOURCE}
    return icons
