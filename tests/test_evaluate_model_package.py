"""Frozen split report field compatibility; model inference is not needed here."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_model_package


class EvaluateModelPackageScopeTests(unittest.TestCase):
    def test_frozen_val_and_test_report_split_specific_counts(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reviewed = root / "reviewed-scenes.jsonl"
            reviewed.write_text("", encoding="utf-8")
            model = Mock(model_sha256="a" * 64,
                         metadata={"model_contract_sha256": "b" * 64},
                         manifest={"classes": ["battle"]})
            verified = {"metadata": {"data_version": "v1"},
                        "snapshot_sha256": "c" * 64, "split_sha256": "d" * 64}
            with patch.object(evaluate_model_package, "SceneModel", return_value=model), \
                 patch.object(evaluate_model_package, "select_frozen_rows",
                              return_value=([{"image": "one.png"}], verified)), \
                 patch.object(evaluate_model_package, "evaluate",
                              return_value={"failed": 0, "samples": 1}):
                for split in ("val", "test"):
                    with self.subTest(split=split):
                        output = StringIO()
                        with redirect_stdout(output):
                            code = evaluate_model_package.main([
                                str(root / "model"), "--kind", "scene", "--client-version", "v1",
                                "--reviewed-jsonl", str(reviewed), "--dataset-dir", str(root),
                                "--split", split])
                        self.assertEqual(code, 0)
                        scope = json.loads(output.getvalue())["evaluation_scope"]
                        self.assertEqual(scope["split"], split)
                        self.assertEqual(scope["image_count"], 1)
                        if split == "test":
                            self.assertEqual(scope["test_image_count"], 1)
                        else:
                            self.assertNotIn("test_image_count", scope)


if __name__ == "__main__":
    unittest.main()
