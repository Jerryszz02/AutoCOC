"""Device discovery and connection strategy."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re

from .adb import ADBClient, DeviceInfo
from .config import AppConfig
from .errors import AdbError, DeviceConnectionError


@dataclass(frozen=True)
class ConnectedDevice:
    serial: str


@dataclass(frozen=True)
class DisplayTarget:
    logical_id: int
    physical_id: str


class DeviceManager:
    def __init__(self, config: AppConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.adb = ADBClient(config.adb.adb_path)

    def connect(self) -> ConnectedDevice:
        scanned_targets: list[str] = []
        initial_devices = self.adb.devices()
        manual_serial = self.config.adb.manual_serial.strip()
        if manual_serial:
            connected = find_ready_device(initial_devices, manual_serial)
            if connected:
                self.logger.info("Using manual device serial: %s", connected.serial)
                return ConnectedDevice(connected.serial)
            if ":" in manual_serial:
                scanned_targets.append(manual_serial)
                self.logger.info("Trying manual adb connect %s", manual_serial)
                try:
                    self.adb.connect(manual_serial, timeout_sec=self.config.adb.connect_timeout_sec)
                except AdbError as exc:
                    raise DeviceConnectionError(f"Unable to connect to configured device {manual_serial}: {exc}") from exc
                connected = find_ready_device(self.adb.devices(), manual_serial)
                if connected:
                    return ConnectedDevice(connected.serial)
            raise DeviceConnectionError(
                f"Configured device {manual_serial!r} is not ready. "
                f"Available devices: {initial_devices!r}. No other device was selected."
            )

        connected = unique_ready_device(initial_devices)
        if connected:
            self.logger.info("Using connected device: %s", connected.serial)
            return ConnectedDevice(connected.serial)

        if self.config.adb.auto_scan_ports:
            for host in self.config.adb.scan_hosts:
                for port in self.config.adb.scan_ports:
                    target = f"{host}:{port}"
                    scanned_targets.append(target)
                    self.logger.info("Trying adb connect %s", target)
                    try:
                        self.adb.connect(target, timeout_sec=self.config.adb.connect_timeout_sec)
                    except AdbError as exc:
                        self.logger.warning("Unable to connect to %s: %s", target, exc)

            connected = unique_ready_device(self.adb.devices())
            if connected:
                self.logger.info("Connected device after port scan: %s", connected.serial)
                return ConnectedDevice(connected.serial)

        raw_devices = self.adb.raw_devices_output()
        raise DeviceConnectionError(
            "Unable to find an Android device in state 'device'.\n"
            f"ADB path: {self.config.adb.adb_path}\n"
            f"Scanned targets: {scanned_targets or 'none'}\n"
            f"adb devices output:\n{raw_devices}\n"
            "Check that MuMu is running, ADB debugging is available, and the emulator port is not blocked."
        )


def unique_ready_device(devices: list[DeviceInfo]) -> DeviceInfo | None:
    ready = {device.serial: device for device in devices if device.state == "device"}
    if len(ready) > 1:
        raise DeviceConnectionError(
            f"Multiple ADB targets are ready: {sorted(ready)}. Set adb.manual_serial to the intended emulator."
        )
    return next(iter(ready.values()), None)


def find_ready_device(devices: list[DeviceInfo], serial: str) -> DeviceInfo | None:
    for device in devices:
        if device.serial == serial and device.state == "device":
            return device
    return None


def parse_display_targets(text: str) -> list[DisplayTarget]:
    targets: dict[int, DisplayTarget] = {}
    for viewport in re.findall(r"DisplayViewport\{[^}]+\}", text):
        logical = re.search(r"\bdisplayId=(\d+)", viewport)
        physical = re.search(r"\buniqueId=['\"]local:(\d+)['\"]", viewport)
        if logical and physical:
            logical_id = int(logical.group(1))
            targets[logical_id] = DisplayTarget(logical_id, physical.group(1))
    for match in re.finditer(
        r"mDisplayId=(\d+)\s*\n\s*mPrimaryDisplayDevice=[^\n]*\(local:(\d+)\)", text
    ):
        logical_id = int(match.group(1))
        targets[logical_id] = DisplayTarget(logical_id, match.group(2))
    return [targets[key] for key in sorted(targets)]


def discover_displays(adb: ADBClient, serial: str, *, timeout_sec: int = 10) -> list[DisplayTarget]:
    result = adb.run(["shell", "dumpsys", "display"], serial=serial, timeout_sec=timeout_sec)
    targets = parse_display_targets(result.stdout)
    if not targets:
        raise DeviceConnectionError("Unable to map logical Android displays to physical screencap displays")
    return targets


def resolve_game_display(
    adb: ADBClient,
    serial: str,
    package_name: str,
    *,
    display_id: int | None = None,
    timeout_sec: int = 10,
) -> DisplayTarget:
    targets = discover_displays(adb, serial, timeout_sec=timeout_sec)
    if display_id is None:
        for command in (["shell", "dumpsys", "window"], ["shell", "dumpsys", "activity", "activities"]):
            result = adb.run(command, serial=serial, timeout_sec=timeout_sec)
            display_id = _package_display(result.stdout, package_name)
            if display_id is not None:
                break
    for target in targets:
        if target.logical_id == display_id:
            return target
    raise DeviceConnectionError(
        f"Unable to select display {display_id!r} for {package_name}. "
        f"Available displays: {targets!r}. Start the game on a known display or specify game.display_id."
    )


def _package_display(text: str, package_name: str) -> int | None:
    current_display: int | None = None
    focused_displays: set[int] = set()
    package_pattern = re.compile(r"(?<![\w.])" + re.escape(package_name) + r"/")
    for line in text.splitlines():
        header = re.search(r"^\s*Display(?:: mDisplayId=| #)(\d+)", line)
        if header:
            current_display = int(header.group(1))
        if current_display is not None and re.search(r"mCurrentFocus=|mFocusedApp=|topResumedActivity=", line):
            if package_pattern.search(line):
                focused_displays.add(current_display)
    if len(focused_displays) > 1:
        raise DeviceConnectionError(f"{package_name} is active on multiple displays; specify game.display_id")
    return next(iter(focused_displays), None)


def foreground_package(adb: ADBClient, serial: str, *, timeout_sec: int = 10) -> str:
    result = adb.run(["shell", "dumpsys", "window"], serial=serial, timeout_sec=timeout_sec, check=False)
    text = result.stdout + "\n" + result.stderr
    for pattern in (
        r"mCurrentFocus=.*?\s(?P<package>[A-Za-z0-9_.]+)/",
        r"mFocusedApp=.*?\s(?P<package>[A-Za-z0-9_.]+)/",
        r"topResumedActivity=.*?\s(?P<package>[A-Za-z0-9_.]+)/",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group("package")
    return "unknown"
