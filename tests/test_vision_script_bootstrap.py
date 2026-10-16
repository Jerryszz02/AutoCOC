"""Direct training-tool launches must find this checkout's product loaders."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "export_vision_model.py", "evaluate_model_package.py", "evaluate_vision.py",
    "train_dynamic_state.py", "export_dynamic_state.py",
    "train_u_position.py", "export_u_position.py",
)


class VisionScriptBootstrapTests(unittest.TestCase):
    def test_direct_entrypoints_load_this_checkouts_product_package(self):
        source = str(ROOT / "src")
        code = (
            "import pathlib, runpy, sys\n"
            "script, source = sys.argv[1:]\n"
            "sys.argv = [script, '--help']\n"
            "try:\n"
            "    runpy.run_path(script, run_name='__main__')\n"
            "except SystemExit as exc:\n"
            "    if exc.code not in (None, 0): raise\n"
            "if source not in sys.path: raise RuntimeError('source path not bootstrapped')\n"
            "import autococ\n"
            "if not pathlib.Path(autococ.__file__).resolve().is_relative_to(pathlib.Path(source)):\n"
            "    raise RuntimeError('product package came from another checkout')\n"
            "print('PRODUCT_IMPORT_OK')\n"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with TemporaryDirectory() as outside:
            for name in SCRIPTS:
                with self.subTest(script=name):
                    script = ROOT / "scripts" / name
                    completed = subprocess.run(
                        [sys.executable, "-c", code, str(script), source],
                        cwd=outside, env=environment, capture_output=True,
                        text=True, timeout=30, check=False)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn("PRODUCT_IMPORT_OK", completed.stdout)

    def test_export_entrypoints_reach_controlled_missing_input_error_after_late_import(self):
        # Export imports its product loader only after argparse has run. A help
        # check alone missed the original S/B failure, so execute that branch
        # without requiring a training checkpoint or touching a package path.
        code = (
            "import runpy, sys, types\n"
            "sys.modules['torch'] = types.ModuleType('torch')\n"
            "script, missing, package = sys.argv[1:]\n"
            "sys.argv = [script, '--training-dir', missing, '--snapshot', missing, "
            "'--package', package]\n"
            "runpy.run_path(script, run_name='__main__')\n"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with TemporaryDirectory() as outside:
            root = Path(outside)
            for name, prefix in (("export_vision_model.py", "export failed:"),
                                 ("export_dynamic_state.py", "R export failed:"),
                                 ("export_u_position.py", "U export failed:")):
                with self.subTest(script=name):
                    completed = subprocess.run(
                        [sys.executable, "-c", code, str(ROOT / "scripts" / name),
                         str(root / "absent-training"), str(root / f"{name}-package")],
                        cwd=outside, env=environment, capture_output=True,
                        text=True, timeout=30, check=False)
                    self.assertEqual(completed.returncode, 2, completed.stderr)
                    self.assertIn(prefix, completed.stderr)
                    self.assertNotIn("ModuleNotFoundError", completed.stderr)
                    self.assertFalse((root / f"{name}-package").exists())


if __name__ == "__main__":
    unittest.main()
