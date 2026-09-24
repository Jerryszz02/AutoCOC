"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib
import warnings

from .errors import ConfigError


VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
VALID_COLOR_SPACES = {"HSV", "RGB", "BGR"}
KNOWN_TASKS = {"launch", "collect", "train", "battle", "settle", "recover", "donate", "request"}


@dataclass(frozen=True)
class AdbConfig:
    adb_path: Path
    auto_scan_ports: bool = True
    manual_serial: str = ""
    scan_hosts: tuple[str, ...] = ("127.0.0.1",)
    scan_ports: tuple[int, ...] = (7555, 16384, 16416, 16448, 16512)
    connect_timeout_sec: int = 5


@dataclass(frozen=True)
class MuMuConfig:
    install_dir: Path
    instance_index: int


@dataclass(frozen=True)
class RuntimeConfig:
    log_level: str = "INFO"
    screenshot_dir: Path = Path("screenshots")
    report_dir: Path = Path("reports")
    step_timeout_sec: int = 10
    dry_run: bool = False
    task_timeout_sec: int = 300
    poll_interval_sec: float = 1.0


@dataclass(frozen=True)
class GameConfig:
    package_name: str = "com.supercell.clashofclans"
    launch_activity: str = ""
    home_scene: str = "village"
    baseline_resolution: tuple[int, int] = (1280, 720)
    startup_timeout_sec: int = 90
    display_id: int | None = None


@dataclass(frozen=True)
class VisionConfig:
    template_dir: Path = Path("assets/templates")
    roi_base_resolution: tuple[int, int] = (1280, 720)
    template_threshold: float = 0.8
    feature_threshold: float = 0.7
    color_space: str = "HSV"
    save_debug_frames: bool = True


@dataclass(frozen=True)
class OCRConfig:
    provider: str = "auto"
    language: str = "en"
    confidence_threshold: float = 0.75
    roi_padding: int = 4


@dataclass(frozen=True)
class ProfileConfig:
    enabled_tasks: tuple[str, ...]


@dataclass(frozen=True)
class BattleConfig:
    objective: str = "resources"
    min_expected_resources: int = 300000
    max_searches: int = 30
    deploy_timeout_sec: int = 180

    def __post_init__(self) -> None:
        if self.objective != "resources":
            raise ConfigError("battle.objective supports only 'resources' (gold + elixir threshold)")
        for name, minimum in (("min_expected_resources", 0), ("max_searches", 1), ("deploy_timeout_sec", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ConfigError(f"battle.{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ReportingConfig:
    write_markdown: bool = True
    save_success_screenshots: bool = False
    save_failure_screenshots: bool = True
    save_decision_trace: bool = True


@dataclass(frozen=True)
class StopConfig:
    max_runs: int = 10
    max_duration_sec: int = 1800
    max_failures: int = 3


@dataclass(frozen=True)
class AppConfig:
    adb: AdbConfig
    runtime: RuntimeConfig
    game: GameConfig
    vision: VisionConfig
    ocr: OCRConfig
    profiles: dict[str, ProfileConfig]
    battle: BattleConfig
    reporting: ReportingConfig
    stop: StopConfig
    source_path: Path
    mumu: MuMuConfig | None = None


def load_config(path: str | Path = "config.toml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(
            f"Configuration file not found: {config_path}. "
            "Copy config.example.toml to config.toml and edit it."
        )

    try:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    adb_config = _load_adb(_section(raw, "adb"))
    runtime_config = _load_runtime(_section(raw, "runtime"))
    game_config = _load_game(_section(raw, "game"))
    vision_config = _load_vision(_section(raw, "vision"))
    ocr_config = _load_ocr(_section(raw, "ocr"))
    profiles = _load_profiles(_section(raw, "profiles"))
    battle_config = _load_battle(_section(raw, "battle"))
    reporting_config = _load_reporting(_section(raw, "reporting"))
    stop_config = _load_stop(_section(raw, "stop"))

    return AppConfig(
        adb=adb_config,
        runtime=runtime_config,
        game=game_config,
        vision=vision_config,
        ocr=ocr_config,
        profiles=profiles,
        battle=battle_config,
        reporting=reporting_config,
        stop=stop_config,
        source_path=config_path,
        mumu=_load_mumu(_section(raw, "mumu")) if "mumu" in raw else None,
    )


def _load_mumu(section: dict[str, Any]) -> MuMuConfig:
    root = _string(section, "install_dir", "").strip()
    if not root or not Path(root).is_absolute() or "instance_index" not in section:
        raise ConfigError("mumu requires an absolute install_dir and an explicit instance_index")
    return MuMuConfig(Path(root), _non_negative_int(section, "instance_index", 0))


def _load_adb(section: dict[str, Any]) -> AdbConfig:
    scan_hosts = _string_list(section, "scan_hosts", ["127.0.0.1"])
    if not scan_hosts:
        raise ConfigError("scan_hosts must not be empty")
    return AdbConfig(
        adb_path=Path(_string(section, "adb_path", "adb.exe")),
        auto_scan_ports=_bool(section, "auto_scan_ports", True),
        manual_serial=_string(section, "manual_serial", ""),
        scan_hosts=tuple(scan_hosts),
        scan_ports=tuple(_int_list(section, "scan_ports", [7555, 16384, 16416, 16448, 16512])),
        connect_timeout_sec=_positive_int(section, "connect_timeout_sec", 5),
    )


def _load_runtime(section: dict[str, Any]) -> RuntimeConfig:
    log_level = _string(section, "log_level", "INFO").upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigError(
            f"runtime.log_level must be one of {sorted(VALID_LOG_LEVELS)}, got {log_level!r}"
        )
    return RuntimeConfig(
        log_level=log_level,
        screenshot_dir=Path(_string(section, "screenshot_dir", "screenshots")),
        report_dir=Path(_string(section, "report_dir", "reports")),
        step_timeout_sec=_positive_int(section, "step_timeout_sec", 10),
        dry_run=_bool(section, "dry_run", False),
        task_timeout_sec=_positive_int(section, "task_timeout_sec", 300),
        poll_interval_sec=_non_negative_float(section, "poll_interval_sec", 1.0),
    )


def _load_game(section: dict[str, Any]) -> GameConfig:
    display_id = None
    if "display_id" in section:
        display_id = _non_negative_int(section, "display_id", 0)
    return GameConfig(
        package_name=_string(section, "package_name", "com.supercell.clashofclans"),
        launch_activity=_string(section, "launch_activity", ""),
        home_scene=_string(section, "home_scene", "village"),
        baseline_resolution=_resolution(section, "baseline_resolution", (1280, 720)),
        startup_timeout_sec=_positive_int(section, "startup_timeout_sec", 90),
        display_id=display_id,
    )


def _load_vision(section: dict[str, Any]) -> VisionConfig:
    color_space = _string(section, "color_space", "HSV").upper()
    if color_space not in VALID_COLOR_SPACES:
        raise ConfigError(
            f"vision.color_space must be one of {sorted(VALID_COLOR_SPACES)}, got {color_space!r}"
        )
    return VisionConfig(
        template_dir=Path(_string(section, "template_dir", "assets/templates")),
        roi_base_resolution=_resolution(section, "roi_base_resolution", (1280, 720)),
        template_threshold=_ratio(section, "template_threshold", 0.8),
        feature_threshold=_ratio(section, "feature_threshold", 0.7),
        color_space=color_space,
        save_debug_frames=_bool(section, "save_debug_frames", True),
    )


def _load_ocr(section: dict[str, Any]) -> OCRConfig:
    return OCRConfig(
        provider=_string(section, "provider", "auto"),
        language=_string(section, "language", "en"),
        confidence_threshold=_ratio(section, "confidence_threshold", 0.75),
        roi_padding=_non_negative_int(section, "roi_padding", 4),
    )


def _load_profiles(section: dict[str, Any]) -> dict[str, ProfileConfig]:
    if not section:
        return {
            "core-loop": ProfileConfig(("launch", "collect", "request", "donate", "battle")),
            "village-only": ProfileConfig(("launch", "collect", "request", "donate")),
            "battle-only": ProfileConfig(("launch", "battle")),
            "social-only": ProfileConfig(("launch", "request", "donate")),
        }

    profiles: dict[str, ProfileConfig] = {}
    for name, value in section.items():
        if not isinstance(value, dict):
            raise ConfigError(f"profiles.{name} must be a TOML table")
        tasks = tuple(_string_list(value, "enabled_tasks", []))
        if not tasks:
            raise ConfigError(f"profiles.{name}.enabled_tasks must not be empty")
        unknown_tasks = sorted(set(tasks) - KNOWN_TASKS)
        if unknown_tasks:
            raise ConfigError(f"profiles.{name}.enabled_tasks contains unknown tasks: {unknown_tasks}")
        profiles[name] = ProfileConfig(tasks)
    return profiles


def _load_battle(section: dict[str, Any]) -> BattleConfig:
    legacy = sorted(set(section) & {"resource_weight", "trophy_weight", "win_rate_weight", "training_cost_weight"})
    if legacy:
        warnings.warn(
            "Ignored obsolete battle weights: " + ", ".join(legacy)
            + ". Remove these keys; resources selection uses only the gold + elixir threshold.",
            UserWarning, stacklevel=2,
        )
    return BattleConfig(
        objective=_string(section, "objective", "resources"),
        min_expected_resources=_non_negative_int(section, "min_expected_resources", 300000),
        max_searches=_positive_int(section, "max_searches", 30),
        deploy_timeout_sec=_positive_int(section, "deploy_timeout_sec", 180),
    )


def _load_reporting(section: dict[str, Any]) -> ReportingConfig:
    return ReportingConfig(
        write_markdown=_bool(section, "write_markdown", True),
        save_success_screenshots=_bool(section, "save_success_screenshots", False),
        save_failure_screenshots=_bool(section, "save_failure_screenshots", True),
        save_decision_trace=_bool(section, "save_decision_trace", True),
    )


def _load_stop(section: dict[str, Any]) -> StopConfig:
    return StopConfig(
        max_runs=_positive_int(section, "max_runs", 10),
        max_duration_sec=_positive_int(section, "max_duration_sec", 1800),
        max_failures=_non_negative_int(section, "max_failures", 3),
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a TOML table")
    return value


def _string(section: dict[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value


def _bool(section: dict[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _positive_int(section: dict[str, Any], key: str, default: int) -> int:
    value = _int(section, key, default)
    if value <= 0:
        raise ConfigError(f"{key} must be greater than 0")
    return value


def _non_negative_int(section: dict[str, Any], key: str, default: int) -> int:
    value = _int(section, key, default)
    if value < 0:
        raise ConfigError(f"{key} must be greater than or equal to 0")
    return value


def _int(section: dict[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer")
    return value


def _ratio(section: dict[str, Any], key: str, default: float) -> float:
    value = _float(section, key, default)
    if value < 0 or value > 1:
        raise ConfigError(f"{key} must be between 0 and 1")
    return value


def _non_negative_float(section: dict[str, Any], key: str, default: float) -> float:
    value = _float(section, key, default)
    if value < 0:
        raise ConfigError(f"{key} must be greater than or equal to 0")
    return value


def _float(section: dict[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    return float(value)


def _string_list(section: dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = section.get(key, default)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{key} must be a list of strings")
    return value


def _int_list(section: dict[str, Any], key: str, default: list[int]) -> list[int]:
    value = section.get(key, default)
    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ConfigError(f"{key} must be a list of integers")
    if not value:
        raise ConfigError(f"{key} must not be empty")
    invalid_ports = [port for port in value if port <= 0 or port > 65535]
    if invalid_ports:
        raise ConfigError(f"{key} contains invalid port values: {invalid_ports}")
    return value


def _resolution(section: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
    value = section.get(key, list(default))
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ConfigError(f"{key} must be a two-item integer list")
    width, height = value
    if width <= 0 or height <= 0:
        raise ConfigError(f"{key} dimensions must be greater than 0")
    return (width, height)
