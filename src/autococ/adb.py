"""ADB command wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import subprocess
from typing import Any

from .errors import AdbError


@dataclass(frozen=True)
class ADBResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class DeviceInfo:
    serial: str
    state: str


class ADBClient:
    def __init__(self, adb_path: str | Path, *, default_timeout_sec: float = 10) -> None:
        self.adb_path = Path(adb_path)
        if not self.adb_path.exists():
            raise AdbError(f"ADB executable does not exist: {self.adb_path}")
        self.default_timeout_sec = self._validate_timeout(default_timeout_sec)

    def run(
        self,
        args: list[str],
        *,
        serial: str | None = None,
        timeout_sec: int | None = None,
        check: bool = True,
    ) -> ADBResult:
        command = self._command(args, serial=serial)
        completed = self._run_process(
            command,
            capture_output=True,
            text=True,
            timeout_sec=timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
        result = ADBResult(tuple(command), completed.returncode, completed.stdout, completed.stderr)
        if check and completed.returncode != 0:
            raise AdbError(
                "ADB command failed: "
                f"{' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return result

    def run_bytes(
        self,
        args: list[str],
        *,
        serial: str | None = None,
        timeout_sec: int | None = None,
        check: bool = True,
    ) -> bytes:
        command = self._command(args, serial=serial)
        completed = self._run_process(
            command,
            capture_output=True,
            text=False,
            timeout_sec=timeout_sec,
        )
        if check and completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")
            stdout = completed.stdout.decode("utf-8", errors="replace")
            raise AdbError(
                "ADB command failed: "
                f"{' '.join(command)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        return completed.stdout

    def _run_process(
        self, command: list[str], *, timeout_sec: int | None, **kwargs: Any
    ) -> subprocess.CompletedProcess[Any]:
        timeout = self.default_timeout_sec if timeout_sec is None else self._validate_timeout(timeout_sec)
        try:
            return subprocess.run(command, timeout=timeout, **kwargs)
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"ADB command timed out after {timeout:g}s: {' '.join(command)}") from exc
        except OSError as exc:
            raise AdbError(f"Unable to execute ADB: {self.adb_path}: {exc}") from exc

    @staticmethod
    def _validate_timeout(timeout_sec: float) -> float:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise AdbError("ADB timeout must be finite and greater than 0")
        return timeout_sec

    def devices(self) -> list[DeviceInfo]:
        result = self.run(["devices"], check=True)
        return parse_devices(result.stdout)

    def raw_devices_output(self) -> str:
        return self.run(["devices"], check=False).stdout

    def connect(self, target: str, timeout_sec: int | None = None) -> ADBResult:
        return self.run(["connect", target], timeout_sec=timeout_sec, check=False)

    def _command(self, args: list[str], *, serial: str | None) -> list[str]:
        command = [str(self.adb_path)]
        if serial:
            command.extend(["-s", serial])
        command.extend(args)
        return command


def parse_devices(output: str) -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    for line in output.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        devices.append(DeviceInfo(serial=parts[0], state=parts[1]))
    return devices
