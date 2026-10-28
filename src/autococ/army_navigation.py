"""Read-only navigation within positively recognized army surfaces."""

from .errors import FlowError


def open_current_army(session, home, evidence: list):
    """Use the current client's observed army entrance without starting a match."""
    if home.scene != "village":
        raise FlowError("Army navigation requires a recognized village")
    if getattr(session, "client_version", None) in {"18.647.2", "18.647.5"}:
        session.click(home, "attack")
        menu = session.wait_for({"search"}, timeout_sec=30, label="army-menu")
        evidence.append(menu.screenshot_path)
        session.click(menu, "edit_army")
    else:
        session.click_template(home, "hud_army", roi=(0, 450, 140, 600))
    return session.wait_for({"training"}, timeout_sec=30, label="inspect-army")


def swipe_current_siege_picker(session, snapshot, *, direction: str) -> None:
    """Page the observed siege catalog without clicking a unit or remove control."""
    session._validate_snapshot(snapshot)
    editor = snapshot.observations.get("army_editor")
    if (session.client_version != "18.647.2" or snapshot.scene != "training"
            or tuple(session.config.game.baseline_resolution) != (1280, 720)
            or not isinstance(editor, dict) or editor.get("frame") != str(snapshot.screenshot_path)
            or editor.get("surface") != "current_picker" or editor.get("ready") is not True
            or editor.get("editing_kind") != "siege" or direction not in {"next", "previous"}):
        raise FlowError("Current siege picker navigation is unverified")
    start, end = ((980, 560), (330, 560)) if direction == "next" else ((330, 560), (980, 560))
    session.event("swipe_current_siege_picker", frame=str(snapshot.screenshot_path),
                  start=list(start), end=list(end), direction=direction)
    session.context.swipe(*start, *end, 550)
    session.action_count += 1
    session.last_snapshot = None
