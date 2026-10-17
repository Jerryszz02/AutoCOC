"""The only troop rosters authorized for the opt-in mixed furnace line."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .errors import FlowError


MIXED_CORE_COUNTS = {"electro_dragon": 10, "dragon": 1,
                     "furnace": 1, "balloon": 1}


def mixed_archer_count(rows: Sequence[Mapping[str, object]]) -> int:
    """Return the bounded filler count after checking every original troop row.

    This does not establish full housing capacity or card identity. Those are
    independently proved by the caller from current game evidence.
    """
    if not isinstance(rows, (list, tuple)):
        raise FlowError("Mixed troop roster rows are unavailable")
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise FlowError("Mixed troop roster contains an invalid row")
        unit_id, source, count = row.get("unit_id"), row.get("source"), row.get("count")
        if (not isinstance(unit_id, str) or unit_id not in set(MIXED_CORE_COUNTS) | {"archer"}
                or unit_id in counts or source != "army" or type(count) is not int or count <= 0):
            raise FlowError("Mixed troop roster contains an unknown, repeated or invalid card")
        counts[unit_id] = count
    archers = counts.pop("archer", 0)
    if counts != MIXED_CORE_COUNTS or archers > 9:
        raise FlowError("Mixed troop roster differs from the exact core or bounded archers")
    return archers
