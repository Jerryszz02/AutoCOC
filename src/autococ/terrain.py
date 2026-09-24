"""Visual deployment candidates whose validity must be proved by troop consumption."""

from __future__ import annotations

from pathlib import Path
import math

from .errors import SceneError
from .locator import scale_box, scale_point


def measure_cloud_cover(screenshot_path: str | Path) -> dict:
    """Reject the broad pale veil seen during enemy-village arrival."""
    import cv2
    import numpy as np

    source = cv2.imdecode(np.fromfile(screenshot_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise SceneError(f"Unable to decode scout screenshot: {screenshot_path}")
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    region = hsv[150:490, 240:1050]
    fraction = float(((region[:, :, 1] < 45) & (region[:, :, 2] > 210)).mean())
    controls = hsv[500:590, :180]
    control_fraction = float(((controls[:, :, 1] < 60) & (controls[:, :, 2] > 210)).mean())
    # A thin cloud can leave the map readable while washing out its controls.
    # Pale map tiles alone do not satisfy this second condition.
    obscured = fraction >= .25 or (fraction >= .1 and control_fraction >= .5)
    return {"obscured": obscured, "pale_fraction": round(fraction, 6),
            "roi_at_1280x720": [240, 150, 1050, 490], "threshold": .25,
            "control_pale_fraction": round(control_fraction, 6),
            "control_roi_at_1280x720": [0, 500, 180, 590],
            "joint_map_threshold": .1, "control_threshold": .5}


def measure_camera_motion(before_path: str | Path, after_path: str | Path) -> dict:
    """Compare map features only; unknown correspondence never authorizes a retry."""
    import cv2
    import numpy as np

    result = {"stationary": None, "matched_points": 0, "inliers": 0, "affine": None,
              "roi_at_1280x720": [240, 150, 1050, 490]}
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[150:490, 240:1050] = 255
    detector = cv2.ORB_create(nfeatures=1200)
    features = []
    for path in (before_path, after_path):
        source = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise SceneError(f"Unable to decode camera screenshot: {path}")
        image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
        points, descriptors = detector.detectAndCompute(image, mask)
        if descriptors is None:
            return result
        features.append((points, descriptors))
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(features[0][1], features[1][1])
    matches = [match for match in matches if match.distance < 40]
    result["matched_points"] = len(matches)
    if len(matches) < 40:
        return result
    before = np.float32([features[0][0][match.queryIdx].pt for match in matches])
    after = np.float32([features[1][0][match.trainIdx].pt for match in matches])
    affine, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC, ransacReprojThreshold=3)
    if affine is None or inliers is None or not np.isfinite(affine).all():
        return result
    count = int(inliers.sum())
    result.update(inliers=count, affine=affine.tolist())
    # A zoom/rotation or inconsistent matches are not evidence of a stationary map.
    if (count < 40 or count / len(matches) < .8 or abs(affine[0, 0] - 1) > .02
            or abs(affine[0, 1]) > .02):
        return result
    result["stationary"] = math.hypot(float(affine[0, 2]), float(affine[1, 2])) < 8
    return result


def find_west_deployment_points(
    screenshot_path: str | Path,
    *,
    baseline_resolution: tuple[int, int] = (1280, 720),
    max_points: int = 7,
) -> list[dict[str, object]]:
    """Prefer clear ground west of a verified red boundary or open polyline.

    This is a visual candidate generator, not proof that a tap deploys a troop.
    The caller must confirm the selected troop count decreases after each attempt.
    If ground appearance is unknown, return at most three geometry-only probes.
    Missing/occluded boundaries return no candidates instead of guessed coordinates.
    """
    import cv2
    import numpy as np

    source = cv2.imdecode(np.fromfile(screenshot_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise SceneError(f"Unable to decode terrain screenshot: {screenshot_path}")
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red = cv2.inRange(hsv, np.array([0, 145, 90]), np.array([17, 255, 215]))
    # Exclude loot, timer and resource HUDs, and all bottom buttons/card controls.
    red[:100] = 0
    red[505:] = 0
    red[:, :150] = 0
    red[:, 1000:] = 0
    # A short or disconnected outer edge can fail the stronger corner/polyline
    # classifier. It must still veto candidates beside a red roof farther inside.
    western_frontier = _western_line_frontier(red)
    red[:, :200] = 0
    lines = cv2.HoughLinesP(red, 1, np.pi / 180, 50, minLineLength=120, maxLineGap=15)
    if lines is None:
        return _find_polyline_grass_points(hsv, gray, red, western_frontier, baseline_resolution, max_points)
    support_mask = cv2.dilate(red, np.ones((3, 3), dtype=np.uint8))
    segments = []
    for raw in lines[:, 0]:
        x1, y1, x2, y2 = map(int, raw)
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if x2 - x1 < 100:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not 0.5 <= abs(slope) <= 1.05:
            continue
        xs = np.linspace(x1, x2, 100).round().astype(int)
        ys = np.linspace(y1, y2, 100).round().astype(int)
        support = float(np.mean(support_mask[ys, xs] > 0))
        if support < 0.85:
            continue
        segments.append({"line": [x1, y1, x2, y2], "slope": slope,
                         "intercept": y1 - slope * x1, "support": support,
                         "length": math.hypot(x2 - x1, y2 - y1)})
    corners = []
    for upper in (line for line in segments if line["slope"] < 0):
        for lower in (line for line in segments if line["slope"] > 0):
            x = (lower["intercept"] - upper["intercept"]) / (upper["slope"] - lower["slope"])
            y = upper["slope"] * x + upper["intercept"]
            if not (225 <= x <= 720 and 165 <= y <= 455):
                continue
            if any(math.hypot(x - line["line"][0], y - line["line"][1]) > 14 for line in (upper, lower)):
                continue
            corners.append((upper["length"] + lower["length"], x, y, upper, lower))
    if not corners:
        dirt = _find_dirt_flank_points(hsv, gray, support_mask, segments, western_frontier, baseline_resolution, max_points)
        if dirt and not dirt[0]["evidence"].get("ground_unverified"):
            return dirt
        polyline = _find_polyline_grass_points(hsv, gray, red, western_frontier, baseline_resolution, max_points)
        if polyline and not polyline[0]["evidence"].get("ground_unverified"):
            return polyline
        return _geometry_probes(dirt + polyline, max_points)
    _, corner_x, corner_y, upper, lower = max(corners, key=lambda candidate: candidate[0])
    endpoints = [upper["line"], lower["line"]]
    boundary_box = (min(line[0] for line in endpoints), min(min(line[1], line[3]) for line in endpoints),
                    max(line[2] for line in endpoints), max(max(line[1], line[3]) for line in endpoints))
    candidates = []
    for delta_y in (0, -36, 36, -72, 72, -108, 108):
        y = round(corner_y + delta_y)
        segment = upper if delta_y < 0 else lower
        boundary_x = (y - segment["intercept"]) / segment["slope"]
        for clearance in (38, 60):
            x = round(boundary_x - clearance)
            if not (225 <= x <= 760 and 165 <= y <= 470):
                continue
            visible_edge = float(western_frontier[y - 18:y + 19].min())
            if x + 13 >= visible_edge:
                continue
            patch = hsv[y - 12:y + 13, x - 12:x + 13]
            grass = ((patch[:, :, 0] >= 35) & (patch[:, :, 0] <= 85)
                     & (patch[:, :, 1] >= 65) & (patch[:, :, 1] <= 235)
                     & (patch[:, :, 2] >= 70))
            grass_fraction = float(grass.mean())
            texture = gray[y - 18:y + 19, x - 18:x + 19]
            texture_std = float(texture.std())
            # Scenery skins change grass brightness; retain fine grass texture while
            # rejecting large light/dark structures from trees, cakes and logs.
            mean_brightness = max(float(texture.mean()), 1.0)
            texture_contrast = texture_std / mean_brightness
            coarse_contrast = float(cv2.GaussianBlur(texture, (11, 11), 0).std()) / mean_brightness
            if grass_fraction < 0.97 or texture_contrast > 0.15 or coarse_contrast > 0.07:
                continue
            if any(math.dist((x, y), candidate["evidence"]["point_at_1280x720"]) < 27 for candidate in candidates):
                continue
            candidates.append({
                "point": list(scale_point((x, y), from_resolution=(1280, 720), to_resolution=baseline_resolution)),
                "bbox": list(scale_box((x - 12, y - 12, x + 13, y + 13), from_resolution=(1280, 720), to_resolution=baseline_resolution)),
                "confidence": round(min(upper["support"], lower["support"], grass_fraction), 4),
                "evidence": {
                    "method": "two_red_boundary_segments_and_clear_grass",
                    "point_at_1280x720": [x, y],
                    "boundary_bbox": list(scale_box(boundary_box, from_resolution=(1280, 720), to_resolution=baseline_resolution)),
                    "corner_at_1280x720": [round(corner_x, 2), round(corner_y, 2)],
                    "segments_at_1280x720": endpoints,
                    "line_support": [upper["support"], lower["support"]],
                    "grass_fraction": round(grass_fraction, 4), "texture_std": round(texture_std, 3),
                    "texture_contrast": round(texture_contrast, 4), "coarse_contrast": round(coarse_contrast, 4),
                    "horizontal_clearance_at_1280x720": clearance,
                    "visible_western_edge_at_1280x720": visible_edge if math.isfinite(visible_edge) else None,
                    "deployment_confirmed": False,
                },
            })
            if len(candidates) >= max_points:
                return candidates
    return candidates or _find_polyline_grass_points(hsv, gray, red, western_frontier, baseline_resolution, max_points)


def find_clear_ground_probes(
    screenshot_path: str | Path,
    *,
    baseline_resolution: tuple[int, int] = (1280, 720),
    max_points: int = 3,
) -> list[dict[str, object]]:
    """Propose up to three textured grass patches when the red border is absent.

    A grass patch does not prove deployability. The caller must first select a
    verified troop, try one placement, and confirm consumption before any burst.
    """
    import cv2
    import numpy as np

    source = cv2.imdecode(np.fromfile(screenshot_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise SceneError(f"Unable to decode terrain screenshot: {screenshot_path}")
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    grass = ((hue >= 35) & (hue <= 85) & (saturation >= 65) & (saturation <= 235) & (value >= 70)).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    coverage = cv2.boxFilter(grass.astype(np.float32), -1, (25, 25))
    mean = cv2.boxFilter(gray, -1, (37, 37))
    deviation = np.sqrt(np.maximum(0, cv2.boxFilter(gray * gray, -1, (37, 37)) - mean * mean))
    coarse = cv2.GaussianBlur(gray, (11, 11), 0)
    coarse_mean = cv2.boxFilter(coarse, -1, (37, 37))
    coarse_deviation = np.sqrt(np.maximum(0, cv2.boxFilter(coarse * coarse, -1, (37, 37)) - coarse_mean * coarse_mean))
    contrast = deviation / np.maximum(mean, 1)
    coarse_contrast = coarse_deviation / np.maximum(mean, 1)
    valid = ((coverage >= .97) & (deviation >= 1.5) & (contrast <= .15) & (coarse_contrast <= .07))
    # Keep the entire probe patch away from resource HUDs, timer and controls.
    region = np.zeros_like(grass)
    region[165:471, 225:761] = 1
    valid &= region.astype(bool)
    distance = cv2.distanceTransform(grass, cv2.DIST_L2, 5)
    ys, xs = np.indices(grass.shape)
    # Favor spacious western patches, without using a fixed placement coordinate.
    score = np.minimum(distance, 80) - (xs - 225) * .08 - np.abs(ys - 315) * .01
    candidates = []
    for _ in range(min(3, max_points)):
        scored = np.where(valid, score, -np.inf)
        if not np.isfinite(scored).any():
            break
        y, x = map(int, np.unravel_index(scored.argmax(), scored.shape))
        candidates.append({
            "point": list(scale_point((x, y), from_resolution=(1280, 720), to_resolution=baseline_resolution)),
            "bbox": list(scale_box((x - 12, y - 12, x + 13, y + 13), from_resolution=(1280, 720), to_resolution=baseline_resolution)),
            "confidence": round(float(coverage[y, x]), 4),
            "evidence": {
                "method": "clear_textured_grass_probe", "point_at_1280x720": [x, y],
                "confidence_meaning": "ground_appearance_only", "boundary_verified": False,
                "grass_fraction": round(float(coverage[y, x]), 4),
                "texture_contrast": round(float(contrast[y, x]), 4),
                "coarse_contrast": round(float(coarse_contrast[y, x]), 4),
                "grass_clearance_at_1280x720": round(float(distance[y, x]), 3),
                "deployment_confirmed": False,
            },
        })
        valid &= (xs - x) ** 2 + (ys - y) ** 2 >= 70 ** 2
    return candidates


def _western_line_frontier(red):
    """Use even disconnected diagonal edges as vetoes, never as tap proposals."""
    import cv2
    import numpy as np

    frontier = np.full(red.shape[0], np.inf)
    lines = cv2.HoughLinesP(red, 1, np.pi / 180, 15, minLineLength=28, maxLineGap=5)
    if lines is None:
        return frontier
    support = cv2.dilate(red, np.ones((3, 3), dtype=np.uint8))
    for x1, y1, x2, y2 in lines[:, 0]:
        if x1 == x2 or not .5 <= abs((y2 - y1) / (x2 - x1)) <= 1.05:
            continue
        xs = np.linspace(x1, x2, 100).round().astype(int)
        ys = np.linspace(y1, y2, 100).round().astype(int)
        if float((support[ys, xs] > 0).mean()) < .9:
            continue
        np.minimum.at(frontier, ys, xs)
    return frontier


def _find_polyline_grass_points(
    hsv, gray, red, visible_frontier, baseline_resolution: tuple[int, int], max_points: int,
) -> list[dict[str, object]]:
    """Follow thin, open red zigzags instead of requiring a long straight corner."""
    import cv2
    import numpy as np

    count, labels, stats, _ = cv2.connectedComponentsWithStats(red)
    frontiers = []
    for label in range(1, count):
        left, top, width, height, area = map(int, stats[label])
        if area < 160 or width < 80 or height < 50 or area / (width * height) > .10:
            continue
        local = (labels[top:top + height, left:left + width] == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fill = sum(cv2.contourArea(contour) for contour in contours) / (width * height)
        if fill > .15:  # Closed red ornaments/roofs enclose area; an open border does not.
            continue
        lines = cv2.HoughLinesP(local, 1, np.pi / 180, 10, minLineLength=14, maxLineGap=4)
        support = cv2.dilate(local, np.ones((3, 3), dtype=np.uint8))
        covered = np.zeros_like(local)
        segments, directions = [], set()
        for raw in ([] if lines is None else lines[:, 0]):
            x1, y1, x2, y2 = map(int, raw)
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            if x2 - x1 < 10:
                continue
            slope = (y2 - y1) / (x2 - x1)
            xs = np.linspace(x1, x2, 40).round().astype(int)
            ys = np.linspace(y1, y2, 40).round().astype(int)
            if not .5 <= abs(slope) <= 1.05 or float((support[ys, xs] > 0).mean()) < .9:
                continue
            segments.append([x1 + left, y1 + top, x2 + left, y2 + top])
            directions.add(1 if slope > 0 else -1)
            cv2.line(covered, (x1, y1), (x2, y2), 255, 3)
        coverage = float(((local > 0) & (covered > 0)).sum() / area)
        if len(segments) < 3 or len(directions) != 2 or coverage < .8:
            continue
        frontier = np.full(720, np.inf)
        for y in range(height):
            columns = np.flatnonzero((local[y] > 0) & (covered[y] > 0))
            if columns.size:
                frontier[top + y] = left + int(columns[0])
        frontiers.append({"x": frontier, "bbox": (left, top, left + width, top + height),
                          "segments": segments, "coverage": coverage, "fill": fill})
    if not frontiers:
        return []
    western_edge = np.min([item["x"] for item in frontiers], axis=0)
    candidates = []
    unverified_ground = []
    for item in sorted(frontiers, key=lambda value: value["bbox"][0]):
        for y in range(max(165, item["bbox"][1] + 18), min(470, item["bbox"][3] - 18), 12):
            nearby = item["x"][y - 18:y + 19]
            if not np.isfinite(item["x"][y]) or np.isfinite(nearby).mean() < .9:
                continue
            # Stay west of every verified line in the entire candidate patch height.
            edge = float(western_edge[y - 18:y + 19].min())
            for clearance in (30, 48):
                x = round(edge - clearance)
                if not 225 <= x <= 760:
                    continue
                visible_edge = float(visible_frontier[y - 18:y + 19].min())
                if x + 13 >= visible_edge:
                    continue
                patch = hsv[y - 12:y + 13, x - 12:x + 13]
                grass = ((patch[:, :, 0] >= 35) & (patch[:, :, 0] <= 85)
                         & (patch[:, :, 1] >= 65) & (patch[:, :, 1] <= 235) & (patch[:, :, 2] >= 70))
                fraction = float(grass.mean())
                texture = gray[y - 18:y + 19, x - 18:x + 19]
                brightness = max(float(texture.mean()), 1.0)
                contrast = float(texture.std()) / brightness
                coarse = float(cv2.GaussianBlur(texture, (11, 11), 0).std()) / brightness
                ground_verified = fraction >= .97 and contrast <= .15 and coarse <= .07
                if not ground_verified and not np.isfinite(nearby).all():
                    continue
                if any(math.dist((x, y), point["evidence"]["point_at_1280x720"]) < 27 for point in candidates):
                    continue
                candidate = {
                    "point": list(scale_point((x, y), from_resolution=(1280, 720), to_resolution=baseline_resolution)),
                    "bbox": list(scale_box((x - 12, y - 12, x + 13, y + 13), from_resolution=(1280, 720), to_resolution=baseline_resolution)),
                    "confidence": round(min(item["coverage"], fraction) if ground_verified else item["coverage"], 4),
                    "evidence": {
                        "method": ("open_red_polyline_western_frontier_and_clear_grass" if ground_verified
                                   else "open_red_polyline_western_frontier_ground_unverified"),
                        "point_at_1280x720": [x, y], "segments_at_1280x720": item["segments"],
                        "boundary_bbox": list(scale_box(item["bbox"], from_resolution=(1280, 720), to_resolution=baseline_resolution)),
                        "diagonal_coverage": round(item["coverage"], 4), "enclosed_area_fraction": round(item["fill"], 4),
                        "western_edge_at_1280x720": edge, "grass_fraction": round(fraction, 4),
                        "visible_western_edge_at_1280x720": visible_edge if math.isfinite(visible_edge) else None,
                        "texture_contrast": round(contrast, 4), "coarse_contrast": round(coarse, 4),
                        "horizontal_clearance_at_1280x720": clearance, "deployment_confirmed": False,
                        "ground_unverified": not ground_verified,
                        "confidence_meaning": "geometry_and_ground" if ground_verified else "boundary_geometry_only",
                        "local_boundary_row_support": float(np.isfinite(nearby).mean()),
                    },
                }
                (candidates if ground_verified else unverified_ground).append(candidate)
                if len(candidates) >= max_points:
                    return candidates
    return candidates or _geometry_probes(unverified_ground, max_points)


def _geometry_probes(candidates: list[dict[str, object]], max_points: int) -> list[dict[str, object]]:
    """Rank unknown ground without claiming appearance proves deployability."""
    selected = []
    for candidate in sorted(candidates, key=lambda item: (
        item["evidence"]["texture_contrast"] + item["evidence"]["coarse_contrast"], -item["confidence"],
    )):
        point = candidate["evidence"]["point_at_1280x720"]
        if any(math.dist(point, item["evidence"]["point_at_1280x720"]) < 27 for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= min(3, max_points):
            break
    return selected


def _find_dirt_flank_points(
    hsv, gray, red_support, segments: list[dict[str, object]], visible_frontier,
    baseline_resolution: tuple[int, int], max_points: int,
) -> list[dict[str, object]]:
    """Handle a partly occluded western corner using a verified notched NW flank."""
    import cv2
    import numpy as np

    pairs = []
    for lower in segments:
        if lower["slope"] >= 0 or lower["length"] < 180 or lower["support"] < .9:
            continue
        ax, ay, bx, by = lower["line"]
        if not (225 <= ax <= 720 and 245 <= ay <= 455):
            continue
        for upper in segments:
            cx, cy, dx, dy = upper["line"]
            if upper["support"] < .9 or abs(upper["slope"] - lower["slope"]) > .035:
                continue
            offset = abs(upper["intercept"] - lower["intercept"]) / math.hypot(1, lower["slope"])
            # A real inset produces distinct, adjacent parallel segments. Duplicate
            # Hough detections of one line and unrelated building edges do not qualify.
            if not (12 <= offset <= 35 and cx >= ax + 100 and dy <= by - 60
                    and -60 <= cx - bx <= 20 and math.dist((bx, by), (cx, cy)) <= 50
                    and dx - ax >= 250 and ay - dy >= 200):
                continue
            pairs.append((lower["length"] + upper["length"], lower, upper))
    if not pairs:
        return []
    _, lower, upper = max(pairs, key=lambda pair: pair[0])
    x1, y1, x2, y2 = lower["line"]
    candidates = []
    unverified_ground = []
    for y in range(y2 + 15, y1 - 14, 5):
        boundary_x = (y - lower["intercept"]) / lower["slope"]
        # Require local visible boundary, never extrapolate through its occluded end.
        ys = np.arange(y - 6, y + 7)
        xs = ((ys - lower["intercept"]) / lower["slope"]).round().astype(int)
        if float((red_support[ys, xs] > 0).mean()) < .9:
            continue
        for clearance in (30, 36, 42):
            x = round(boundary_x - clearance)
            if not (225 <= x <= 760 and 165 <= y <= 470):
                continue
            visible_edge = float(visible_frontier[y - 14:y + 15].min())
            if x + 11 >= visible_edge:
                continue
            patch = hsv[y - 10:y + 11, x - 10:x + 11]
            dirt = ((patch[:, :, 0] >= 18) & (patch[:, :, 0] <= 45)
                    & (patch[:, :, 1] >= 25) & (patch[:, :, 1] <= 125)
                    & (patch[:, :, 2] >= 45) & (patch[:, :, 2] <= 175))
            ground_fraction = float(dirt.mean())
            texture = gray[y - 14:y + 15, x - 14:x + 15]
            brightness = max(float(texture.mean()), 1.0)
            contrast = float(texture.std()) / brightness
            coarse = float(cv2.GaussianBlur(texture, (9, 9), 0).std()) / brightness
            ground_verified = ground_fraction >= .98 and contrast <= .15 and coarse <= .07
            local_ys = np.arange(y - 14, y + 15)
            local_xs = ((local_ys - lower["intercept"]) / lower["slope"]).round().astype(int)
            local_support = float((red_support[local_ys, local_xs] > 0).mean())
            western_edge = float(local_xs.min())
            if not ground_verified and (local_support < 1 or x + 11 > western_edge - 6):
                continue
            if any(math.dist((x, y), item["evidence"]["point_at_1280x720"]) < 27 for item in candidates):
                continue
            candidate = {
                "point": list(scale_point((x, y), from_resolution=(1280, 720), to_resolution=baseline_resolution)),
                "bbox": list(scale_box((x - 10, y - 10, x + 11, y + 11), from_resolution=(1280, 720), to_resolution=baseline_resolution)),
                "confidence": round(min(lower["support"], upper["support"], ground_fraction if ground_verified else local_support), 4),
                "evidence": {
                    "method": ("two_notched_red_flank_segments_and_clear_dirt" if ground_verified
                               else "two_notched_red_flank_segments_ground_unverified"),
                    "point_at_1280x720": [x, y], "boundary_side": "northwest",
                    "segments_at_1280x720": [lower["line"], upper["line"]],
                    "boundary_bbox": list(scale_box((x1, upper["line"][3], upper["line"][2], y1), from_resolution=(1280, 720), to_resolution=baseline_resolution)),
                    "line_support": [lower["support"], upper["support"]],
                    "ground_fraction": round(ground_fraction, 4), "ground_type": "low_saturation_dirt",
                    "texture_contrast": round(contrast, 4), "coarse_contrast": round(coarse, 4),
                    "horizontal_clearance_at_1280x720": clearance, "deployment_confirmed": False,
                    "visible_western_edge_at_1280x720": visible_edge if math.isfinite(visible_edge) else None,
                    "ground_unverified": not ground_verified,
                    "confidence_meaning": "geometry_and_ground" if ground_verified else "boundary_geometry_only",
                    "local_boundary_row_support": local_support, "western_edge_at_1280x720": western_edge,
                },
            }
            (candidates if ground_verified else unverified_ground).append(candidate)
            if len(candidates) >= max_points:
                return candidates
    return candidates or _geometry_probes(unverified_ground, max_points)
