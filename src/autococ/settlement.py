"""Positive image evidence for earned stars and the observed reward panel."""

import math
from pathlib import Path


TEMPLATES = Path(__file__).resolve().parents[2] / "assets/templates"
BONUS_TEMPLATE_SOURCE = "reports/20260924-004347-675110-2e2b80bd/frames/00061-battle-settlement-reread.png"
BONUS_TEMPLATE_CROPS = {
    "gold": (1007, 349, 1036, 381),
    "elixir": (1007, 385, 1036, 418),
    "dark_elixir": (1007, 421, 1036, 454),
}


def _split_star_notch(contour, polygon, defects, width: int, height: int,
                      origin: tuple[int, int]) -> dict | None:
    """Identify one short extra tip that divides a single deep star notch.

    The current victory animation can put six convex tips and six concavities
    on a five-point silhouette. Both close pairs and the remaining angular
    distribution must agree; a six-point star or a weak/missing tip does not.
    """
    import cv2

    # Only the smaller side-star silhouette showed this animation. An
    # incomplete enlarged centre star can also grow six notches after a tip
    # is covered, but it is not evidence for a second earned star.
    if (len(polygon) != 12 or defects is None or len(defects) != 6 or
            not 110 <= width <= 140 or not 105 <= height <= 135):
        return None
    minimum = min(width, height)
    if any(float(row[3]) / 256 < minimum * .09 for row in defects[:, 0]):
        return None
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    center = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    def angular_points(points):
        return sorted((math.atan2(float(point[1]) - center[1],
                                  float(point[0]) - center[0]) % (2 * math.pi),
                       (int(point[0]), int(point[1]))) for point in points)

    valleys = angular_points([polygon[int(row[2])][0] for row in defects[:, 0]])
    tips = angular_points(cv2.convexHull(polygon)[:, 0])
    if len(tips) != 6:
        return None

    def close_pairs(points, max_angle: float, max_distance: float):
        return [(index, gap) for index, (angle, point) in enumerate(points)
                if (gap := (points[(index + 1) % len(points)][0] - angle) % (2 * math.pi)) <= max_angle
                and math.dist(point, points[(index + 1) % len(points)][1]) <= max_distance]

    valley_pairs = close_pairs(valleys, math.radians(30), minimum * .18)
    tip_pairs = close_pairs(tips, math.radians(35), minimum * .27)
    if len(valley_pairs) != 1 or len(tip_pairs) != 1:
        return None
    valley_index, valley_gap = valley_pairs[0]
    tip_index, _ = tip_pairs[0]
    # The short tip must lie in the same angular sector as the doubled notch.
    valley_mid = (valleys[valley_index][0] + valley_gap / 2) % (2 * math.pi)
    tip_gap = (tips[(tip_index + 1) % 6][0] - tips[tip_index][0]) % (2 * math.pi)
    tip_mid = (tips[tip_index][0] + tip_gap / 2) % (2 * math.pi)
    if abs((valley_mid - tip_mid + math.pi) % (2 * math.pi) - math.pi) > math.radians(30):
        return None
    paired = valleys[valley_index][1], valleys[(valley_index + 1) % 6][1]
    if (min(point[0] for point in paired) <= center[0] + minimum * .1 or
            max(point[1] for point in paired) >= center[1] - minimum * .1):
        return None  # The observed side-star split is confined to its upper-right notch.
    grouped = [(angle, point) for index, (angle, point) in enumerate(valleys)
               if index not in {valley_index, (valley_index + 1) % 6}]
    grouped.append((valley_mid, (round((paired[0][0] + paired[1][0]) / 2),
                                 round((paired[0][1] + paired[1][1]) / 2))))
    grouped.sort()
    gaps = [(grouped[(index + 1) % 5][0] - angle) % (2 * math.pi)
            for index, (angle, _) in enumerate(grouped)]
    if any(not math.radians(40) <= gap <= math.radians(105) for gap in gaps):
        return None

    def placed(point):
        return [point[0] + origin[0], point[1] + origin[1]]

    return {"raw_valleys_at_1280x720": [placed(point) for _, point in valleys],
            "grouped_valleys_at_1280x720": [placed(point) for _, point in grouped],
            "merged_valleys_at_1280x720": [placed(point) for point in paired],
            "short_tip_pair_at_1280x720": [placed(tips[tip_index][1]),
                                           placed(tips[(tip_index + 1) % 6][1])],
            "valley_gap_degrees": round(math.degrees(valley_gap), 3),
            "valley_gap_pixels": round(math.dist(*paired), 3),
            "remaining_gaps_degrees": [round(math.degrees(gap), 3) for gap in gaps]}


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

    # The enlarged centre victory star reaches y=27 in the current client;
    # y=65 cuts off its tip before the five-concavity check can run.
    roi = (430, 20, 845, 220)
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
        if not (0 < x and 0 < y and x + width < right - left and y + height < bottom - top
                and 80 <= width <= 185 and 80 <= height <= 165
                and .75 <= width / height <= 1.35):
            valid = False
            continue
        solidity = area / cv2.contourArea(cv2.convexHull(contour))
        # Antialiased highlights can add small vertices to a complete star.
        # A slightly coarser outline keeps its five deep concavities intact.
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, .02 * perimeter, True)
        approximation = .02
        # A complete current-client star can gain two short outline vertices
        # during its shine animation. Only an otherwise valid silhouette may
        # use a slightly coarser outline; missing tips/concavities still fail.
        if (len(polygon) > 12 and .35 <= area / (width * height) <= .6
                and .5 <= solidity <= .8):
            coarser = cv2.approxPolyDP(contour, .025 * perimeter, True)
            if 10 <= len(coarser) <= 12:
                polygon, approximation = coarser, .025
        if not 10 <= len(polygon) <= 12:
            shape.update(solidity=solidity, polygon_vertices=len(polygon),
                         polygon_approximation=approximation)
            valid = False
            continue
        defects = cv2.convexityDefects(polygon, cv2.convexHull(polygon, returnPoints=False))
        depths = [] if defects is None else [float(item[0][3]) / 256 for item in defects]
        deep_concavities = sum(depth >= min(width, height) * .09 for depth in depths)
        merged_notch = (_split_star_notch(contour, polygon, defects, width, height,
                                          (left, top))
                        if deep_concavities == 6 else None)
        complete = (.35 <= area / (width * height) <= .6 and .5 <= solidity <= .8
                    and 10 <= len(polygon) <= 12 and
                    (deep_concavities == 5 or merged_notch is not None))
        shape.update(verified=complete, solidity=solidity, polygon_vertices=len(polygon),
                     polygon_approximation=approximation, concavity_depths=depths,
                     merged_concavity_pair=merged_notch)
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
