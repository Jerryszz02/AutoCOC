import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.scene import SceneSnapshot
from scripts.evaluate_vision import EvaluationRow, evaluate_manifest, summarize_predictions


class VisionEvaluationTests(unittest.TestCase):
    def test_confusion_macro_f1_acceptance_and_unknown_false_positive(self) -> None:
        rows = [
            EvaluationRow("a.png", "village", "village", 0.9),
            EvaluationRow("b.png", "village", "battle", 0.9),
            EvaluationRow("c.png", "battle", "battle", 0.6),
            EvaluationRow("d.png", "unknown", "village", 0.95),
        ]
        result = summarize_predictions(rows)
        self.assertEqual(result["confusion_matrix"]["village"]["battle"], 1)
        self.assertAlmostEqual(result["macro_f1"], 7 / 18)
        self.assertAlmostEqual(result["accepted_accuracy"], 1 / 3)
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(result["unknown_actionable_false_positives"], 1)

    def test_numeric_denominator_uses_only_labels_and_checks_resource_source(self) -> None:
        rows = [
            EvaluationRow("a", "village", "village", 0.9, {"gold": 10}, {"gold": 10}),
            EvaluationRow("b", "village", "village", 0.9, {"gold": 20}, {"gold": 20}, "village_inventory", "battle_loot"),
            EvaluationRow("c", "village", "village", 0.9, {"elixir": 0}, {}),
            EvaluationRow("d", "village", "village", 0.9, {}, {"gold": 999}),
        ]
        result = summarize_predictions(rows)
        self.assertEqual(result["numeric_total"], 3)
        self.assertEqual(result["numeric_correct"], 1)
        self.assertEqual(result["numeric_exact_ratio"], 1 / 3)
        self.assertIsNone(result["numeric_by_resource"]["dark_elixir"]["exact_ratio"])

    def test_no_samples_or_acceptances_do_not_look_like_perfect_results(self) -> None:
        result = summarize_predictions([])
        for name in ("macro_f1", "accepted_accuracy", "coverage", "numeric_exact_ratio"):
            self.assertIsNone(result[name])
        rejected = summarize_predictions([EvaluationRow("a", "village", "village", float("nan"))])
        self.assertEqual(rejected["coverage"], 0)
        self.assertIsNone(rejected["accepted_accuracy"])

    def write_manifest(self, directory: Path, samples: list[dict]) -> Path:
        path = directory / "labels.json"
        path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
        return path

    def test_same_pixels_different_png_encoding_count_once_and_merge_labels(self) -> None:
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            for name, compression in (("a.png", 0), ("b.png", 9)):
                success, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, compression])
                self.assertTrue(success)
                (directory / name).write_bytes(encoded.tobytes())
            manifest = self.write_manifest(directory, [
                {"path": "a.png", "scene": "village", "resources": {"gold": 10}},
                {"path": "b.png", "scene": "village", "resources": {"elixir": 20, "gems": None}},
                {"path": "not-labeled.png"},
            ])
            recognizer = Mock()
            recognizer.recognize.return_value = SceneSnapshot("village", 0.95, directory / "a.png", {"resources": {"gold": 10, "elixir": 20}})
            result = evaluate_manifest(manifest, recognizer)
            recognizer.recognize.assert_called_once()
            self.assertEqual(result["metrics"]["sample_count"], 1)
            self.assertEqual(result["metrics"]["numeric_total"], 2)
            self.assertEqual(result["metrics"]["numeric_exact_ratio"], 1)
            self.assertEqual(len(result["duplicate_entries"]), 1)
            self.assertEqual(len(result["excluded_unlabeled"]), 1)

    def test_conflicting_duplicate_human_labels_are_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            success, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), dtype=np.uint8))
            self.assertTrue(success)
            (directory / "a.png").write_bytes(encoded.tobytes())
            manifest = self.write_manifest(directory, [{"path": "a.png", "scene": "village"}, {"path": "a.png", "scene": "battle"}])
            recognizer = Mock()
            with self.assertRaisesRegex(ValueError, "Conflicting human labels"):
                evaluate_manifest(manifest, recognizer)
            recognizer.recognize.assert_not_called()

    def test_missing_labeled_image_stays_in_error_denominator(self) -> None:
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = self.write_manifest(directory, [{"path": "missing.png", "scene": "village", "resources": {"gold": 10}}])
            recognizer = Mock()
            result = evaluate_manifest(manifest, recognizer)
            self.assertEqual(result["metrics"]["sample_count"], 1)
            self.assertEqual(result["metrics"]["recognition_errors"], 1)
            self.assertEqual(result["metrics"]["confusion_matrix"]["village"]["__error__"], 1)
            self.assertEqual(result["metrics"]["numeric_exact_ratio"], 0)
            recognizer.recognize.assert_not_called()

    def test_empty_image_and_recognizer_failure_are_not_silently_excluded(self) -> None:
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "empty.png").write_bytes(b"")
            success, encoded = cv2.imencode(".png", np.ones((2, 2, 3), dtype=np.uint8))
            self.assertTrue(success)
            (directory / "valid.png").write_bytes(encoded.tobytes())
            manifest = self.write_manifest(directory, [{"path": "empty.png", "scene": "village"}, {"path": "valid.png", "scene": "battle"}])
            recognizer = Mock()
            recognizer.recognize.side_effect = RuntimeError("OCR failed")
            result = evaluate_manifest(manifest, recognizer)
            self.assertEqual(result["metrics"]["sample_count"], 2)
            self.assertEqual(result["metrics"]["recognition_errors"], 2)
            self.assertEqual(result["metrics"]["coverage"], 0)
            self.assertIsNone(result["metrics"]["accepted_accuracy"])
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
