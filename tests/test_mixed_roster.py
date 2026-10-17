"""Pure mixed-roster gates; no game or OCR is involved."""

import pytest

from autococ.errors import FlowError
from autococ.mixed_roster import mixed_archer_count


CORE = ({"unit_id": "electro_dragon", "source": "army", "count": 10},
        {"unit_id": "dragon", "source": "army", "count": 1},
        {"unit_id": "furnace", "source": "army", "count": 1},
        {"unit_id": "balloon", "source": "army", "count": 1})


@pytest.mark.parametrize("archers", [0, 1, 3, 9])
def test_only_exact_core_and_bounded_archers_are_accepted(archers):
    rows = [dict(row) for row in CORE]
    if archers:
        rows.insert(0, {"unit_id": "archer", "source": "army", "count": archers})
    assert mixed_archer_count(rows) == archers


@pytest.mark.parametrize("rows", [
    None, {}, [*CORE, CORE[0]], CORE[:-1],
    [*CORE, {"unit_id": "archer", "source": "army", "count": 0}],
    [*CORE, {"unit_id": "archer", "source": "army", "count": 10}],
    [*CORE, {"unit_id": "archer", "source": "army", "count": True}],
    [*CORE, {"unit_id": "barbarian", "source": "army", "count": 3}],
    [*CORE, {"unit_id": "archer", "source": "event", "count": 3}],
    [{**CORE[0], "count": 9}, *CORE[1:]],
    [{**CORE[0], "source": "super_troop"}, *CORE[1:]],
])
def test_unknown_duplicate_wrong_core_and_invalid_filler_are_rejected(rows):
    with pytest.raises(FlowError):
        mixed_archer_count(rows)
