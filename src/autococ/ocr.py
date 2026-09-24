"""Local Chinese/English OCR with image-relative text boxes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

from .config import OCRConfig
from .errors import OCRError


@dataclass(frozen=True)
class OCRText:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


class OCRProvider(Protocol):
    def recognize(
        self,
        image_path: Path,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[OCRText]:
        """Return OCR results for the full image or a specific ROI."""

    def recognize_line(self, image_path: Path, roi: tuple[int, int, int, int]) -> list[OCRText]:
        """Read an already localized text line without running text detection."""


class NullOCRProvider:
    """Explicitly disabled OCR; never used as an automatic fallback."""

    def __init__(self, reason: str = "no OCR provider available") -> None:
        self.reason = reason

    def recognize(
        self,
        image_path: Path,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[OCRText]:
        if not Path(image_path).exists():
            raise OCRError(f"Image does not exist: {image_path}")
        return []

    def recognize_line(self, image_path: Path, roi: tuple[int, int, int, int]) -> list[OCRText]:
        return self.recognize(image_path, roi)


class RapidOCRProvider:
    """Use bundled ONNX models without downloading anything during inference."""

    def __init__(self) -> None:
        spec = find_spec("rapidocr")
        if spec is None or spec.origin is None:
            raise OCRError("Local OCR is unavailable. Install with: python -m pip install -e .")
        model_dir = Path(spec.origin).parent / "models"
        models = {
            "Det.model_path": model_dir / "PP-OCRv6_det_small.onnx",
            "Cls.model_path": model_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "Rec.model_path": model_dir / "PP-OCRv6_rec_small.onnx",
        }
        missing = [str(path) for path in models.values() if not path.is_file()]
        if missing:
            raise OCRError(f"Bundled OCR models missing; reinstall rapidocr>=3.9,<4: {missing}")
        try:
            from rapidocr import RapidOCR

            self._engine = RapidOCR(params={
                **{key: str(path) for key, path in models.items()},
                "Global.log_level": "warning",
                "EngineConfig.onnxruntime.intra_op_num_threads": 4,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
        except Exception as exc:
            raise OCRError(f"Unable to initialize local RapidOCR: {exc}") from exc

    def recognize(
        self,
        image_path: Path,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[OCRText]:
        try:
            import cv2
            import numpy as np

            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise OCRError(f"Unable to decode image: {image_path}")
            offset_x = offset_y = 0
            if roi is not None:
                left, top, right, bottom = roi
                height, width = image.shape[:2]
                left, top = max(0, left), max(0, top)
                right, bottom = min(width, right), min(height, bottom)
                if right <= left or bottom <= top:
                    raise OCRError(f"OCR ROI is outside image: {roi}")
                offset_x, offset_y = left, top
                image = image[top:bottom, left:right]
            # RapidOCR remembers these switches between calls; line OCR disables detection.
            output = self._engine(image, use_det=True, use_cls=True, use_rec=True)
            if output.txts is None or output.boxes is None or output.scores is None:
                return []
            results = []
            for text, score, box in zip(output.txts, output.scores, output.boxes, strict=True):
                points = np.asarray(box)
                left, top = np.floor(points.min(axis=0)).astype(int)
                right, bottom = np.ceil(points.max(axis=0)).astype(int)
                results.append(OCRText(str(text), float(score), (
                    int(left) + offset_x, int(top) + offset_y,
                    int(right) + offset_x, int(bottom) + offset_y,
                )))
            return results
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"Local OCR failed for {image_path}: {exc}") from exc

    def recognize_line(self, image_path: Path, roi: tuple[int, int, int, int]) -> list[OCRText]:
        try:
            import cv2
            import numpy as np

            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise OCRError(f"Unable to decode image: {image_path}")
            height, width = image.shape[:2]
            left, top, right, bottom = roi
            left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
            if right <= left or bottom <= top:
                raise OCRError(f"OCR ROI is outside image: {roi}")
            output = self._engine(image[top:bottom, left:right], use_det=False, use_cls=False, use_rec=True)
            if output.txts is None or output.scores is None:
                return []
            return [OCRText(str(text), float(score), (left, top, right, bottom))
                    for text, score in zip(output.txts, output.scores, strict=True)]
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"Local line OCR failed for {image_path}: {exc}") from exc


def create_ocr_provider(config: OCRConfig) -> OCRProvider:
    provider = config.provider.strip().lower()
    if provider in {"", "none", "disabled"}:
        return NullOCRProvider("OCR disabled by configuration")
    if provider in {"auto", "rapidocr"}:
        return RapidOCRProvider()
    raise OCRError(
        f"Unsupported OCR provider {config.provider!r}. "
        "Set ocr.provider to 'auto', 'rapidocr', or 'none'."
    )


def filter_ocr_results(results: list[OCRText], min_confidence: float) -> list[OCRText]:
    if min_confidence < 0 or min_confidence > 1:
        raise OCRError("min_confidence must be between 0 and 1")
    return [result for result in results if result.confidence >= min_confidence]
