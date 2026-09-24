import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autococ.errors import LocatorError
from autococ.locator import Bounds, XMLLocator, find_template, scale_box, scale_path, scale_point


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy>
  <node text="Start" resource-id="app:id/start" class="android.widget.Button"
        content-desc="primary action" clickable="true" enabled="true" bounds="[10,20][110,70]" />
  <node text="Done" resource-id="app:id/done" class="android.widget.TextView"
        content-desc="" clickable="false" enabled="true" bounds="[5,100][80,140]" />
  <node text="Disabled" resource-id="app:id/disabled" class="android.widget.Button"
        content-desc="" clickable="true" enabled="false" bounds="[0,0][20,20]" />
</hierarchy>
"""


class BoundsTests(unittest.TestCase):
    def test_parse_bounds_and_center(self) -> None:
        bounds = Bounds.parse("[10,20][110,70]")
        self.assertEqual(bounds.center, (60, 45))

    def test_rejects_invalid_bounds(self) -> None:
        with self.assertRaises(LocatorError):
            Bounds.parse("[10,20][10,30]")

    def test_scales_bounds_between_resolutions(self) -> None:
        bounds = Bounds.parse("[10,20][110,70]")

        scaled = bounds.scale((1280, 720), (1920, 1080))

        self.assertEqual(scaled, Bounds(15, 30, 165, 105))


class ScaleTests(unittest.TestCase):
    def test_template_scales_with_baseline(self) -> None:
        import cv2
        import numpy as np

        template = np.random.default_rng(4).integers(0, 255, (12, 16, 3), dtype=np.uint8)
        screen = np.zeros((200, 200, 3), dtype=np.uint8)
        screen[60:84, 80:112] = cv2.resize(template, (32, 24), interpolation=cv2.INTER_LINEAR)
        with TemporaryDirectory() as tmp:
            screenshot_path = Path(tmp) / "screen.png"
            template_path = Path(tmp) / "template.png"
            cv2.imwrite(str(screenshot_path), screen)
            cv2.imwrite(str(template_path), template)
            result = find_template(screenshot_path, template_path, template_base_resolution=(100, 100))
        self.assertEqual(result.bounds, Bounds(80, 60, 112, 84))

    def test_scales_point_box_and_path(self) -> None:
        self.assertEqual(scale_point((100, 50), from_resolution=(1000, 500), to_resolution=(2000, 1000)), (200, 100))
        self.assertEqual(
            scale_box((10, 20, 30, 40), from_resolution=(100, 100), to_resolution=(200, 300)),
            (20, 60, 60, 120),
        )
        self.assertEqual(
            scale_path(((10, 10), (20, 20)), from_resolution=(100, 100), to_resolution=(200, 300)),
            ((20, 30), (40, 60)),
        )


class XMLLocatorTests(unittest.TestCase):
    def test_finds_by_resource_id(self) -> None:
        result = XMLLocator.from_string(SAMPLE_XML).find({"resource-id": "app:id/start"})

        self.assertIsNotNone(result)
        self.assertEqual(result.point, (60, 45))
        self.assertEqual(result.method, "resource-id")

    def test_finds_by_content_desc(self) -> None:
        result = XMLLocator.from_string(SAMPLE_XML).find({"content-desc": "primary action"})

        self.assertIsNotNone(result)
        self.assertEqual(result.method, "content-desc")

    def test_finds_by_text(self) -> None:
        result = XMLLocator.from_string(SAMPLE_XML).find({"text": "Done"})

        self.assertIsNotNone(result)
        self.assertEqual(result.point, (42, 120))

    def test_skips_disabled_nodes_by_default(self) -> None:
        result = XMLLocator.from_string(SAMPLE_XML).find({"text": "Disabled"})

        self.assertIsNone(result)

    def test_bounds_filter_restricts_match(self) -> None:
        result = XMLLocator.from_string(SAMPLE_XML).find(
            {"text": "Start", "bounds": "[0,0][50,50]"}
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
