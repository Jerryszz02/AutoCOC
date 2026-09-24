from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from unittest.mock import patch

from autococ.adb import ADBClient
from autococ.errors import AdbError


class ADBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.executable = Path(self.temp.name) / "adb.exe"
        self.executable.touch()
        self.client = ADBClient(self.executable)

    def test_text_and_binary_calls_have_default_timeout(self) -> None:
        for method, output in ((self.client.run, "ok"), (self.client.run_bytes, b"ok")):
            with self.subTest(method=method.__name__), patch("autococ.adb.subprocess.run") as process:
                process.return_value = subprocess.CompletedProcess([], 0, output, output[:0])
                method(["devices"])
                self.assertEqual(process.call_args.kwargs["timeout"], 10)

    def test_explicit_timeout_is_preserved(self) -> None:
        with patch("autococ.adb.subprocess.run") as process:
            process.return_value = subprocess.CompletedProcess([], 0, "", "")
            self.client.run(["devices"], timeout_sec=3)
            self.assertEqual(process.call_args.kwargs["timeout"], 3)

    def test_process_failures_are_domain_errors_even_when_check_false(self) -> None:
        for method in (self.client.run, self.client.run_bytes):
            for failure in (subprocess.TimeoutExpired("adb", 10), OSError("cannot start")):
                with self.subTest(method=method.__name__, failure=type(failure).__name__):
                    with patch("autococ.adb.subprocess.run", side_effect=failure):
                        with self.assertRaises(AdbError) as raised:
                            method(["devices"], check=False)
                        self.assertIs(raised.exception.__cause__, failure)

    def test_unbounded_or_nonpositive_timeouts_are_rejected(self) -> None:
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(AdbError):
                ADBClient(self.executable, default_timeout_sec=value)


if __name__ == "__main__":
    unittest.main()
