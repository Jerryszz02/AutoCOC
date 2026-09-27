from pathlib import Path

import pytest

from autococ.capture import ScreenshotCapture
from autococ.errors import FlowError
from autococ.evidence import EvidenceBudget


class Capture:
    def capture_screenshot_artifact(self, path):
        path.write_bytes(b"frame" * 10)
        return ScreenshotCapture(path, 10, 10, .1)


def test_capacity_never_deletes_or_overwrites_existing_frames(tmp_path):
    previous = tmp_path / "previous.png"
    previous.write_bytes(b"existing evidence")
    budget = EvidenceBudget(tmp_path, 80)
    published = tmp_path / "new.png"
    artifact = budget.capture(Capture(), published)
    assert artifact.path == published and published.read_bytes() == b"frame" * 10
    with pytest.raises(FlowError, match="capacity"):
        budget.capture(Capture(), tmp_path / "too-big.png")
    assert previous.read_bytes() == b"existing evidence"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["new.png", "previous.png"]
    with pytest.raises(FlowError, match="already exists"):
        budget.capture(Capture(), published)


def test_preflight_reserves_complete_result_frames(tmp_path):
    budget = EvidenceBudget(tmp_path, 30_000_000)
    budget.require_frames((1280, 720), 6)
    with pytest.raises(FlowError):
        budget.require_frames((2560, 1440), 6)


def test_partial_capture_is_cleaned_up_without_touching_published_evidence(tmp_path):
    class BrokenCapture:
        def capture_screenshot_artifact(self, path):
            path.write_bytes(b"partial")
            raise RuntimeError("transport failure")
    budget = EvidenceBudget(tmp_path, 80)
    with pytest.raises(RuntimeError):
        budget.capture(BrokenCapture(), tmp_path / "failed.png")
    assert list(tmp_path.iterdir()) == []


def test_no_write_outside_run(tmp_path):
    budget = EvidenceBudget(tmp_path / "frames", 80)
    with pytest.raises(FlowError):
        budget.capture(Capture(), tmp_path / "outside.png")


def test_wait_polling_is_bounded_but_selected_frames_are_never_recycled(tmp_path):
    budget = EvidenceBudget(tmp_path, 120)
    selected = tmp_path / "selected.png"
    budget.capture(Capture(), selected)
    for index in range(30):
        latest = tmp_path / f"poll-{index}.png"
        budget.capture(Capture(), latest, transient=True)
        assert selected.exists()
        assert len(list(tmp_path.glob("*.png"))) == 2
    assert budget.used_bytes == 50
    budget.retain(latest)
    assert budget.used_bytes == 100
    with pytest.raises(FlowError):
        budget.capture(Capture(), tmp_path / "over-capacity.png")
    assert selected.exists() and latest.exists()
