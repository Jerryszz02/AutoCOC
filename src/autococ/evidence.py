"""Bounded admission of new run frames; existing evidence is never evicted."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from .errors import FlowError


class EvidenceBudget:
    def __init__(self, directory: Path, limit_bytes: int) -> None:
        if type(limit_bytes) is not int or limit_bytes <= 0:
            raise ValueError("Evidence capacity must be a positive byte count")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit_bytes = limit_bytes
        self._transient: set[Path] = set()

    @property
    def used_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.iterdir()
                   if path.is_file() and path not in self._transient)

    def retain(self, path: Path) -> None:
        path = Path(path).resolve()
        if path not in self._transient:
            return
        if self.used_bytes + path.stat().st_size > self.limit_bytes:
            raise FlowError("Evidence capacity insufficient to retain the final polling frame")
        self._transient.remove(path)

    def require_frames(self, resolution: tuple[int, int], count: int = 6) -> None:
        width, height = resolution
        if (type(count) is not int or count < 1 or any(type(v) is not int or v <= 0 for v in resolution)):
            raise FlowError("Invalid evidence reservation")
        # RGBA raw payload plus row/filter/PNG overhead, conservative for RGB PNG0.
        required = count * (width * height * 4 + height * 16 + 65536)
        if self.used_bytes + required > self.limit_bytes:
            raise FlowError("Evidence capacity insufficient for post-deployment and result frames")

    def capture(self, client, path: Path, *, transient: bool = False):
        path = Path(path).resolve()
        if path.parent != self.directory or path.exists():
            raise FlowError("Evidence destination is outside this run or already exists")
        if not transient and self.used_bytes >= self.limit_bytes:
            raise FlowError("Evidence capacity exhausted; existing frames preserved")
        # Only this newly captured temporary frame is discarded on rejection.
        # It never overwrites or removes any published/user evidence.
        with TemporaryDirectory(prefix=".pending-", dir=self.directory) as temporary:
            staged = Path(temporary) / path.name
            artifact = client.capture_screenshot_artifact(staged)
            if staged.stat().st_size + self.used_bytes > self.limit_bytes:
                raise FlowError("New frame exceeds evidence capacity; existing frames preserved")
            staged.replace(path)
            # Only unsuccessful wait-loop frames explicitly registered by this
            # instance may be recycled. All published evidence remains intact.
            for previous in tuple(self._transient):
                if previous.parent != self.directory:
                    raise FlowError("Transient frame ownership mismatch")
                previous.unlink(missing_ok=True)
                self._transient.remove(previous)
            if transient:
                self._transient.add(path)
            return replace(artifact, path=path)
