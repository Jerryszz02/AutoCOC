"""Project-specific exceptions."""


class AutoCOCError(Exception):
    """Base exception for framework errors."""


class ConfigError(AutoCOCError):
    """Raised when configuration is missing or invalid."""


class AdbError(AutoCOCError):
    """Raised when an ADB command fails."""


class DeviceConnectionError(AdbError):
    """Raised when no usable Android device can be connected."""


class CaptureError(AutoCOCError):
    """Raised when screenshot or UI XML capture fails."""


class LocatorError(AutoCOCError):
    """Raised when a locator cannot find a safe target."""


class OCRError(AutoCOCError):
    """Raised when an OCR provider cannot be initialized or used."""


class SceneError(AutoCOCError):
    """Raised when scene analysis cannot be completed."""


class RuleEngineError(AutoCOCError):
    """Raised when the rule engine cannot build a valid plan."""


class ActionError(AutoCOCError):
    """Raised when an automation action is invalid or fails."""


class FlowError(AutoCOCError):
    """Raised when a flow cannot be executed."""


class StopRequested(KeyboardInterrupt):
    """Cooperative user stop; task failure handlers must not swallow it."""


class MuMuDisplayUnavailable(FlowError):
    """The native renderer cannot yet capture the selected game display."""


class DeploymentError(FlowError):
    """Deployment stopped; already verified consumption remains inspectable."""

    def __init__(self, message: str, *, partial_receipt: dict[str, object]) -> None:
        super().__init__(message)
        self.partial_receipt = {**partial_receipt, "completed": False}
