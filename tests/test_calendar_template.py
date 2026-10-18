from pathlib import Path

import pytest

from autococ.errors import LocatorError
from autococ.locator import find_template


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("variant", ["plain", "badge"])
def test_current_calendar_entry_ignores_notification_badge(variant):
    frame = ROOT / "tests/fixtures" / f"event_calendar_{variant}_18_647_2.png"
    template = ROOT / "assets/templates/daily_event_calendar_18_647_2.png"
    result = find_template(frame, template, threshold=.88, roi=(270, 610, 365, 710),
                           roi_base_resolution=(1280, 720))
    assert result.confidence >= .88
    assert 570 <= result.bounds.center[0] <= 660
    assert 1290 <= result.bounds.center[1] <= 1400
    with pytest.raises(LocatorError):
        find_template(frame, template, threshold=.88, roi=(365, 610, 550, 710),
                      roi_base_resolution=(1280, 720))
