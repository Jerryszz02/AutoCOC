from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from autococ.config import OCRConfig
from autococ.errors import OCRError
from autococ.ocr import OCRText, RapidOCRProvider, create_ocr_provider, filter_ocr_results


class OCRTests(unittest.TestCase):
    def test_disabled_provider_returns_no_results(self) -> None:
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.png"
            image.write_bytes(b"placeholder")
            provider = create_ocr_provider(OCRConfig(provider="none"))

            self.assertEqual(provider.recognize(image), [])

    def test_auto_initializes_real_provider(self) -> None:
        with patch("autococ.ocr.RapidOCRProvider") as provider:
            self.assertIs(create_ocr_provider(OCRConfig()), provider.return_value)

    def test_missing_runtime_fails_explicitly(self) -> None:
        with patch("autococ.ocr.find_spec", return_value=None):
            with self.assertRaisesRegex(OCRError, "Local OCR is unavailable"):
                RapidOCRProvider()

    def test_full_ocr_restores_detection_after_line_read(self) -> None:
        import cv2
        import numpy as np

        class StatefulEngine:
            use_det = True

            def __call__(self, image, **switches):
                self.use_det = switches.get("use_det", self.use_det)
                if not self.use_det:
                    return SimpleNamespace(txts=("359565",), scores=(0.99,))
                return SimpleNamespace(txts=("商店",), scores=(0.99,),
                                       boxes=np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]]))

        provider = RapidOCRProvider.__new__(RapidOCRProvider)
        provider._engine = StatefulEngine()
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.png"
            cv2.imwrite(str(image), np.zeros((20, 20, 3), dtype=np.uint8))
            self.assertEqual(provider.recognize_line(image, (0, 0, 15, 15))[0].text, "359565")
            result = provider.recognize(image)
        self.assertEqual(result[0].text, "商店")
        self.assertEqual(result[0].bbox, (0, 0, 10, 10))

    def test_filters_by_confidence(self) -> None:
        results = [
            OCRText("low", 0.5),
            OCRText("high", 0.9),
        ]

        filtered = filter_ocr_results(results, 0.75)

        self.assertEqual(filtered, [OCRText("high", 0.9)])


if __name__ == "__main__":
    unittest.main()
