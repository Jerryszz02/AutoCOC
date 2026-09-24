"""UI XML and optional OpenCV template locators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .errors import LocatorError


BOUNDS_PATTERN = re.compile(r"\[(?P<left>-?\d+),(?P<top>-?\d+)\]\[(?P<right>-?\d+),(?P<bottom>-?\d+)\]")


@dataclass(frozen=True)
class Bounds:
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def parse(cls, value: str) -> "Bounds":
        match = BOUNDS_PATTERN.fullmatch(value.strip())
        if not match:
            raise LocatorError(f"Invalid bounds format: {value!r}")
        bounds = cls(
            left=int(match.group("left")),
            top=int(match.group("top")),
            right=int(match.group("right")),
            bottom=int(match.group("bottom")),
        )
        if bounds.right <= bounds.left or bounds.bottom <= bounds.top:
            raise LocatorError(f"Invalid bounds dimensions: {value!r}")
        return bounds

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    def contains(self, other: "Bounds") -> bool:
        return (
            other.left >= self.left
            and other.top >= self.top
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    def offset(self, x: int, y: int) -> "Bounds":
        return Bounds(self.left + x, self.top + y, self.right + x, self.bottom + y)

    def scale(self, from_resolution: tuple[int, int], to_resolution: tuple[int, int]) -> "Bounds":
        left, top, right, bottom = scale_box(
            (self.left, self.top, self.right, self.bottom),
            from_resolution=from_resolution,
            to_resolution=to_resolution,
        )
        return Bounds(left, top, right, bottom)


@dataclass(frozen=True)
class UiNode:
    text: str
    resource_id: str
    class_name: str
    content_desc: str
    clickable: bool
    enabled: bool
    bounds: Bounds | None


@dataclass(frozen=True)
class LocatorResult:
    node: UiNode | None
    method: str
    confidence: float
    point: tuple[int, int]
    bounds: Bounds | None


@dataclass(frozen=True)
class RecognitionResult:
    name: str
    method: str
    confidence: float
    bbox: tuple[int, int, int, int]
    text: str | None = None
    value: int | float | str | None = None


class XMLLocator:
    def __init__(self, nodes: list[UiNode]) -> None:
        self.nodes = nodes

    @classmethod
    def from_file(cls, path: str | Path) -> "XMLLocator":
        return cls.from_string(Path(path).read_text(encoding="utf-8", errors="replace"))

    @classmethod
    def from_string(cls, xml_text: str) -> "XMLLocator":
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise LocatorError(f"Unable to parse UI XML: {exc}") from exc

        nodes: list[UiNode] = []
        for element in root.iter("node"):
            bounds_text = element.attrib.get("bounds", "")
            bounds = Bounds.parse(bounds_text) if bounds_text else None
            nodes.append(
                UiNode(
                    text=element.attrib.get("text", ""),
                    resource_id=element.attrib.get("resource-id", ""),
                    class_name=element.attrib.get("class", ""),
                    content_desc=element.attrib.get("content-desc", ""),
                    clickable=_xml_bool(element.attrib.get("clickable", "false")),
                    enabled=_xml_bool(element.attrib.get("enabled", "true")),
                    bounds=bounds,
                )
            )
        return cls(nodes)

    def find(self, locator: dict[str, object], *, require_enabled: bool = True) -> LocatorResult | None:
        candidates = self._filtered_nodes(locator, require_enabled=require_enabled)
        strategies = [
            ("resource-id", "resource_id"),
            ("content-desc", "content_desc"),
            ("text", "text"),
        ]
        for locator_key, node_attr in strategies:
            if locator_key in locator:
                value = str(locator[locator_key])
                result = _first_with_attr(candidates, node_attr, value)
                if result:
                    return _result(result, locator_key)

        class_value = locator.get("class")
        text_value = locator.get("text")
        content_desc_value = locator.get("content-desc")
        if class_value and text_value:
            result = _first_combo(candidates, class_name=str(class_value), text=str(text_value))
            if result:
                return _result(result, "class+text")
        if class_value and content_desc_value:
            result = _first_combo(candidates, class_name=str(class_value), content_desc=str(content_desc_value))
            if result:
                return _result(result, "class+content-desc")

        if "bounds" in locator:
            result = next((node for node in candidates if node.bounds is not None), None)
            if result:
                return _result(result, "bounds")

        if class_value:
            result = _first_with_attr(candidates, "class_name", str(class_value))
            if result:
                return _result(result, "class")

        return None

    def _filtered_nodes(self, locator: dict[str, object], *, require_enabled: bool) -> list[UiNode]:
        region = _locator_bounds(locator.get("bounds"))
        clickable = locator.get("clickable")
        nodes: list[UiNode] = []
        for node in self.nodes:
            if require_enabled and not node.enabled:
                continue
            if isinstance(clickable, bool) and node.clickable != clickable:
                continue
            if region and (node.bounds is None or not region.contains(node.bounds)):
                continue
            nodes.append(node)
        return nodes


def find_template(
    screenshot_path: str | Path,
    template_path: str | Path,
    *,
    threshold: float = 0.8,
    roi: Bounds | tuple[int, int, int, int] | None = None,
    roi_base_resolution: tuple[int, int] | None = None,
    template_base_resolution: tuple[int, int] | None = None,
) -> LocatorResult:
    screenshot = Path(screenshot_path)
    template = Path(template_path)
    if not screenshot.exists():
        raise LocatorError(f"Screenshot does not exist: {screenshot}")
    if not template.exists():
        raise LocatorError(f"Template image does not exist: {template}")

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LocatorError("OpenCV is not installed. Install with: python -m pip install -e .[image]") from exc

    screenshot_img = cv2.imread(str(screenshot))
    template_img = cv2.imread(str(template))
    if screenshot_img is None:
        raise LocatorError(f"Unable to read screenshot image: {screenshot}")
    if template_img is None:
        raise LocatorError(f"Unable to read template image: {template}")
    base = template_base_resolution or roi_base_resolution
    if base is not None:
        _validate_resolution(base, "template_base_resolution")
        target_size = (
            max(1, round(template_img.shape[1] * screenshot_img.shape[1] / base[0])),
            max(1, round(template_img.shape[0] * screenshot_img.shape[0] / base[1])),
        )
        template_img = cv2.resize(template_img, target_size, interpolation=cv2.INTER_LINEAR)

    offset_x = 0
    offset_y = 0
    search_img = screenshot_img
    if roi is not None:
        bounds = _coerce_bounds(roi)
        if roi_base_resolution is not None:
            height, width = screenshot_img.shape[:2]
            bounds = bounds.scale(roi_base_resolution, (width, height))
        bounds = _clip_bounds(bounds, screenshot_img.shape[1], screenshot_img.shape[0])
        offset_x = bounds.left
        offset_y = bounds.top
        search_img = screenshot_img[bounds.top : bounds.bottom, bounds.left : bounds.right]
        if search_img.size == 0:
            raise LocatorError("Template ROI is empty after clipping")

    if template_img.shape[0] > search_img.shape[0] or template_img.shape[1] > search_img.shape[1]:
        raise LocatorError("Template image is larger than the search region")

    result = cv2.matchTemplate(search_img, template_img, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_location = cv2.minMaxLoc(result)
    if max_value < threshold:
        raise LocatorError(f"Template confidence {max_value:.3f} is below threshold {threshold:.3f}")

    height, width = template_img.shape[:2]
    left, top = max_location
    left += offset_x
    top += offset_y
    bounds = Bounds(left=left, top=top, right=left + width, bottom=top + height)
    return LocatorResult(
        node=None,
        method="opencv_template",
        confidence=float(max_value),
        point=bounds.center,
        bounds=bounds,
    )


def find_color_feature(
    screenshot_path: str | Path,
    *,
    name: str,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    threshold: float = 0.7,
    roi: Bounds | tuple[int, int, int, int] | None = None,
    roi_base_resolution: tuple[int, int] | None = None,
    color_space: str = "HSV",
) -> RecognitionResult:
    screenshot = Path(screenshot_path)
    if not screenshot.exists():
        raise LocatorError(f"Screenshot does not exist: {screenshot}")
    if threshold < 0 or threshold > 1:
        raise LocatorError("threshold must be between 0 and 1")

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LocatorError("OpenCV is not installed. Install with: python -m pip install -e .[image]") from exc

    image = cv2.imread(str(screenshot))
    if image is None:
        raise LocatorError(f"Unable to read screenshot image: {screenshot}")

    offset_x = 0
    offset_y = 0
    region = image
    if roi is not None:
        bounds = _coerce_bounds(roi)
        if roi_base_resolution is not None:
            height, width = image.shape[:2]
            bounds = bounds.scale(roi_base_resolution, (width, height))
        bounds = _clip_bounds(bounds, image.shape[1], image.shape[0])
        offset_x = bounds.left
        offset_y = bounds.top
        region = image[bounds.top : bounds.bottom, bounds.left : bounds.right]
        if region.size == 0:
            raise LocatorError("Color ROI is empty after clipping")

    normalized_space = color_space.upper()
    if normalized_space == "HSV":
        region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    elif normalized_space == "RGB":
        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    elif normalized_space != "BGR":
        raise LocatorError("color_space must be HSV, RGB, or BGR")

    mask = cv2.inRange(region, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    confidence = float(cv2.countNonZero(mask)) / float(mask.size)
    if confidence < threshold:
        raise LocatorError(f"Color feature confidence {confidence:.3f} is below threshold {threshold:.3f}")

    points = cv2.findNonZero(mask)
    if points is None:
        raise LocatorError("Color feature produced an empty mask")
    left, top, width, height = cv2.boundingRect(points)
    bbox = (left + offset_x, top + offset_y, left + width + offset_x, top + height + offset_y)
    return RecognitionResult(
        name=name,
        method="color",
        confidence=confidence,
        bbox=bbox,
        value=confidence,
    )


def scale_point(
    point: tuple[int, int],
    *,
    from_resolution: tuple[int, int],
    to_resolution: tuple[int, int],
) -> tuple[int, int]:
    _validate_resolution(from_resolution, "from_resolution")
    _validate_resolution(to_resolution, "to_resolution")
    scale_x = to_resolution[0] / from_resolution[0]
    scale_y = to_resolution[1] / from_resolution[1]
    return (round(point[0] * scale_x), round(point[1] * scale_y))


def scale_box(
    box: tuple[int, int, int, int],
    *,
    from_resolution: tuple[int, int],
    to_resolution: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top = scale_point((box[0], box[1]), from_resolution=from_resolution, to_resolution=to_resolution)
    right, bottom = scale_point((box[2], box[3]), from_resolution=from_resolution, to_resolution=to_resolution)
    if right <= left or bottom <= top:
        raise LocatorError("scaled box must have positive dimensions")
    return (left, top, right, bottom)


def scale_path(
    points: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    from_resolution: tuple[int, int],
    to_resolution: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    return tuple(scale_point(point, from_resolution=from_resolution, to_resolution=to_resolution) for point in points)


def _result(node: UiNode, method: str) -> LocatorResult:
    if node.bounds is None:
        raise LocatorError(f"Matched node by {method}, but it has no bounds")
    return LocatorResult(node=node, method=method, confidence=1.0, point=node.bounds.center, bounds=node.bounds)


def _first_with_attr(nodes: list[UiNode], attr: str, value: str) -> UiNode | None:
    return next((node for node in nodes if getattr(node, attr) == value), None)


def _first_combo(
    nodes: list[UiNode],
    *,
    class_name: str,
    text: str | None = None,
    content_desc: str | None = None,
) -> UiNode | None:
    for node in nodes:
        if node.class_name != class_name:
            continue
        if text is not None and node.text == text:
            return node
        if content_desc is not None and node.content_desc == content_desc:
            return node
    return None


def _xml_bool(value: str) -> bool:
    return value.lower() == "true"


def _locator_bounds(value: object) -> Bounds | None:
    if value is None:
        return None
    if isinstance(value, Bounds):
        return value
    if isinstance(value, str):
        return Bounds.parse(value)
    raise LocatorError("locator bounds must be a Bounds instance or '[l,t][r,b]' string")


def _coerce_bounds(value: Bounds | tuple[int, int, int, int]) -> Bounds:
    if isinstance(value, Bounds):
        return value
    if isinstance(value, tuple) and len(value) == 4 and all(isinstance(item, int) for item in value):
        return Bounds(*value)
    raise LocatorError("ROI must be Bounds or a four-item integer tuple")


def _clip_bounds(bounds: Bounds, width: int, height: int) -> Bounds:
    clipped = Bounds(
        left=max(0, min(bounds.left, width)),
        top=max(0, min(bounds.top, height)),
        right=max(0, min(bounds.right, width)),
        bottom=max(0, min(bounds.bottom, height)),
    )
    if clipped.right <= clipped.left or clipped.bottom <= clipped.top:
        raise LocatorError("ROI is outside the screenshot")
    return clipped


def _validate_resolution(value: tuple[int, int], name: str) -> None:
    if len(value) != 2 or value[0] <= 0 or value[1] <= 0:
        raise LocatorError(f"{name} must contain positive width and height")
