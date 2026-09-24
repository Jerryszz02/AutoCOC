"""Donation execution over calibrated observations; the live reader is pending.

The screenshot recognizer does not yet emit ``donation_view``. Until a real
dialog is sampled and its reader implemented, live execution stops before any
unit click. See docs/DESKTOP.md for the observation contract and remaining work.
"""

from __future__ import annotations

from pathlib import Path
import math
import time

from .errors import FlowError
from .scene import SceneSnapshot
from .session import GameSession


def read_donation_view(snapshot: SceneSnapshot) -> dict:
    view = snapshot.observations.get("donation_view")
    if (snapshot.scene != "donation" or not math.isfinite(snapshot.confidence) or snapshot.confidence < 0.9
            or not isinstance(view, dict) or view.get("frame") != str(snapshot.screenshot_path)):
        raise FlowError("donation_layout_unverified: a calibrated current-frame donation view is required")
    if not isinstance(view.get("request_id"), str) or not view["request_id"].strip():
        raise FlowError("Donation request identity is unreadable")
    for key in ("filled", "capacity", "own_units", "gems", "elixir", "dark_elixir"):
        if type(view.get(key)) is not int or view[key] < 0:
            raise FlowError(f"Donation field is unreadable: {key}")
    if not 0 <= view["filled"] <= view["capacity"] or view["capacity"] == 0:
        raise FlowError("Donation capacity is invalid")
    if not isinstance(view.get("units"), list):
        raise FlowError("Donation unit choices are unreadable")
    seen = set()
    for unit in view["units"]:
        if not isinstance(unit, dict) or not isinstance(unit.get("id"), str) or not unit["id"]:
            raise FlowError("Donation unit identity is unreadable")
        if unit["id"] in seen:
            raise FlowError("Donation unit identity is ambiguous")
        seen.add(unit["id"])
        if type(unit.get("allowed")) is not bool or type(unit.get("enabled")) is not bool:
            raise FlowError("Donation unit restrictions are unreadable")
        for key, minimum in (("space", 1), ("cost", 0)):
            if type(unit.get(key)) is not int or unit[key] < minimum:
                raise FlowError(f"Donation unit {key} is unreadable")
        if unit.get("currency") not in {"elixir", "dark_elixir", "gems"}:
            raise FlowError("Donation currency is unreadable")
        box = unit.get("bbox")
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= 1280 or not 0 <= box[1] < box[3] <= 720):
            raise FlowError("Donation unit bounds are invalid")
    return view


def execute_donation(session: GameSession, current: SceneSnapshot,
                     evidence: list[Path], metrics: dict) -> tuple[str, str]:
    """One unit at a time; no repeat click while a result remains uncertain."""
    metrics.setdefault("donation_actions", 0)
    metrics.setdefault("verified_donations", [])
    for _ in range(50):
        session.check_deadline()
        before = read_donation_view(current)
        units = [unit for unit in before["units"] if unit["allowed"] and unit["enabled"]
                 and unit["currency"] in {"elixir", "dark_elixir"}
                 and unit["space"] <= before["capacity"] - before["filled"]
                 and unit["cost"] <= before[unit["currency"]]]
        if not units:
            return ("succeeded", "donation_units_and_cost_verified") if metrics["donated_units"] else (
                "skipped", "no_eligible_resource_donation_in_verified_dialog")
        unit = units[0]
        currency = unit["currency"]
        cost_key = "donation_cost_" + currency
        previous_cost = metrics[cost_key]
        left, top, right, bottom = unit["bbox"]
        metrics["donation_actions"] += 1
        metrics[cost_key] = None
        metrics["donation_cost_gems"] = None
        session.tap(current, ((left + right) // 2, (top + bottom) // 2),
                    reason=f"Donate verified eligible unit {unit['id']} using {currency}")
        verified = False
        for attempt in range(3):
            session.check_deadline()
            after_frame = session.observe("donation-verification")
            evidence.append(after_frame.screenshot_path)
            if after_frame.screenshot_path == current.screenshot_path:
                raise FlowError("Donation verification reused its before screenshot")
            after = read_donation_view(after_frame)
            if after["request_id"] != before["request_id"] or after["capacity"] != before["capacity"]:
                raise FlowError("Donation request changed before verification")
            if after["gems"] != before["gems"]:
                metrics["donation_cost_gems"] = before["gems"] - after["gems"]
                raise FlowError("Gem balance changed during donation")
            metrics["donation_cost_gems"] = 0
            other_currency = "elixir" if currency == "dark_elixir" else "dark_elixir"
            if (after["own_units"] == before["own_units"] + 1
                    and after["filled"] >= before["filled"] + unit["space"]
                    and before[currency] - after[currency] == unit["cost"]
                    and after[other_currency] == before[other_currency]):
                receipt = {"unit": unit["id"], "currency": currency, "cost": unit["cost"],
                           "before_frame": str(current.screenshot_path), "after_frame": str(after_frame.screenshot_path)}
                metrics["donated_units"] += 1
                metrics[cost_key] = previous_cost + unit["cost"]
                metrics["verified_donations"].append(receipt)
                session.event("donation_verified", **receipt)
                current = after_frame
                verified = True
                break
            if attempt < 2:
                time.sleep(min(session.config.runtime.poll_interval_sec, 1.0))
        if not verified:
            raise FlowError("Donation result unverified; no repeat click was sent")
    raise FlowError("Donation action limit reached; verified partial receipts retained")
