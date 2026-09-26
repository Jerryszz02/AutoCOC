"""Read actual goal progress; battle count is never a proxy for game progress."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import re
import time
import tomllib

from .errors import CapabilityUnavailable, ConfigError, FlowError
from .routine_config import TaskSpec


@dataclass(frozen=True)
class GoalProgress:
    completed: bool
    current: object = None
    target: object = None
    reason: str = "goal_not_configured"
    evidence: tuple[Path, ...] = ()
    values: dict = field(default_factory=dict)
    available: bool = True


@dataclass(frozen=True)
class NavigationStep:
    """Open one read-only control on an independently identified page."""
    page_title: str
    open_text: str
    open_roi: tuple[int, int, int, int]
    title_roi: tuple[int, int, int, int] = (0, 0, 1280, 240)
    section_label: str = ""
    section_roi: tuple[int, int, int, int] = (0, 0, 1280, 720)

    def validate(self):
        if not isinstance(self.page_title, str) or not self.page_title.strip():
            raise ConfigError("Navigation step needs an exact page_title")
        if not isinstance(self.open_text, str) or _normalized(self.open_text) not in {
                "打开", "查看", "详情", "open", "view", "details"}:
            raise ConfigError("Navigation may only use a read-only Open/View/Details control")
        if not isinstance(self.section_label, str):
            raise ConfigError("navigation.section_label must be text")
        for key in ("open_roi", "title_roi", "section_roi"):
            _validate_roi(getattr(self, key), f"navigation.{key}")
        if self.section_label and not _roi_contains(self.section_roi, self.open_roi):
            raise ConfigError("navigation.open_roi must stay inside the named section_roi")


@dataclass(frozen=True)
class ProgressAdapter:
    id: str
    kind: str
    title: str
    progress_roi: tuple[int, int, int, int]
    entry_text: str = ""
    entry_template: str = ""
    entry_roi: tuple[int, int, int, int] = (0, 0, 1280, 720)
    title_roi: tuple[int, int, int, int] = (0, 0, 1280, 240)
    progress_pattern: str = r"(?P<current>\d[\d, ]*)\s*/\s*(?P<target>\d[\d, ]*)"
    active_label: str = ""
    scoring_condition: str = ""
    scoring_units: tuple[str, ...] = ()
    accepted_label: str = ""
    building_type: str = ""
    completed_label: str = ""
    unavailable_label: str = ""
    close_text: str = ""
    version: int = 1
    navigation: tuple[NavigationStep, ...] = ()

    def validate(self):
        if type(self.version) is not int or self.version != 1 or self.kind not in {"event", "clan_games"}:
            raise ConfigError("Unsupported progress adapter version/kind")
        for key in ("id", "title", "entry_text", "entry_template", "active_label", "accepted_label",
                    "building_type", "completed_label", "unavailable_label", "close_text", "progress_pattern", "scoring_condition"):
            if not isinstance(getattr(self, key), str):
                raise ConfigError(f"adapter.{key} must be text")
        if not self.id or not self.title or bool(self.entry_text) == bool(self.entry_template):
            raise ConfigError("Adapter requires id, exact page title and one entry text/template")
        if any(_unsafe_navigation_label(value) for value in (self.entry_text, self.entry_template, self.close_text)):
            raise ConfigError("Progress navigation cannot use task, reward or purchase controls")
        for key in ("progress_roi", "entry_roi", "title_roi"):
            _validate_roi(getattr(self, key), f"adapter.{key}")
        if not isinstance(self.navigation, (tuple, list)) or len(self.navigation) > 3:
            raise ConfigError("Adapter navigation supports at most three read-only steps")
        titles = set()
        for step in self.navigation:
            if not isinstance(step, NavigationStep):
                raise ConfigError("Invalid adapter navigation step")
            step.validate()
            title = _normalized(step.page_title)
            if title in titles or title == _normalized(self.title):
                raise ConfigError("Navigation page titles must be distinct from each other and the progress page")
            titles.add(title)
        if self.kind == "clan_games" and (not self.accepted_label or self.building_type not in {"air_defense", "spell_factory"}):
            raise ConfigError("Clan adapter requires an accepted-task label and supported building type")
        if self.kind == "event" and not self.active_label:
            raise ConfigError("Event adapter requires an active-event label")
        if self.kind == "event" and self.scoring_condition not in {"battle", "destroy_building", "deploy_units"}:
            raise ConfigError("Event adapter must declare a supported scoring_condition")
        if self.scoring_condition == "destroy_building" and self.building_type not in {"air_defense", "spell_factory"}:
            raise ConfigError("Building-scoring events require a supported building_type")
        from .unit_catalog import get_unit
        if (not isinstance(self.scoring_units, (list, tuple)) or
                any(not isinstance(name, str) or get_unit(name) is None for name in self.scoring_units)):
            raise ConfigError("Event scoring_units must contain catalogued unit IDs")
        if self.scoring_condition == "deploy_units" and not self.scoring_units:
            raise ConfigError("Unit-scoring events require scoring_units")
        try:
            pattern = re.compile(self.progress_pattern)
        except re.error as exc:
            raise ConfigError(f"Invalid adapter progress regex: {exc}") from exc
        if not {"current", "target"} <= pattern.groupindex.keys():
            raise ConfigError("Progress regex needs named current and target groups")


def load_adapter(path: str | Path) -> ProgressAdapter:
    try:
        with Path(path).open("rb") as stream:
            values = tomllib.load(stream)
        for key in ("progress_roi", "entry_roi", "title_roi", "scoring_units"):
            if key in values:
                values[key] = tuple(values[key])
        if "navigation" in values:
            if not isinstance(values["navigation"], list):
                raise ConfigError("adapter.navigation must be an array of read-only steps")
            steps = []
            for item in values["navigation"]:
                if not isinstance(item, dict):
                    raise ConfigError("Each navigation step must be a table")
                for key in ("open_roi", "title_roi", "section_roi"):
                    if key in item:
                        item[key] = tuple(item[key])
                steps.append(NavigationStep(**item))
            values["navigation"] = tuple(steps)
        adapter = ProgressAdapter(**values)
        adapter.validate()
        return adapter
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigError(f"Cannot load progress adapter {path}: {exc}") from exc


def _texts(snapshot, *, roi=None):
    width, height = snapshot.observations.get("baseline_resolution", (1280, 720))
    if (type(width) is not int or type(height) is not int or min(width, height) <= 0):
        return
    for item in snapshot.observations.get("ocr", []):
        if not isinstance(item, dict):
            continue
        confidence = item.get("confidence", 0)
        box = item.get("bbox")
        if (type(confidence) not in {int, float} or not math.isfinite(confidence)
                or not .9 <= confidence <= 1 or not isinstance(item.get("text"), str)
                or not isinstance(box, (tuple, list)) or len(box) != 4
                or any(type(v) not in {int, float} or not math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            continue
        x = (box[0] + box[2]) * 640 / width
        y = (box[1] + box[3]) * 360 / height
        if roi is None or roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]:
            yield item


def _normalized(text):
    return re.sub(r"\s+", "", text).casefold()


def _unsafe_navigation_label(text):
    name = _normalized(text)
    return any(word in name for word in ("开始", "接取", "接受", "放弃", "领取", "购买", "支付", "升级", "报名",
                                         "start", "accept", "claim", "buy", "purchase", "abandon", "upgrade"))


def _validate_roi(roi, name):
    if (not isinstance(roi, (tuple, list)) or len(roi) != 4 or any(type(x) is not int for x in roi)
            or not 0 <= roi[0] < roi[2] <= 1280 or not 0 <= roi[1] < roi[3] <= 720):
        raise ConfigError(f"{name} must be a valid 1280x720 ROI")


def _roi_contains(outer, inner):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def _exact(snapshot, label, roi=None):
    return [item for item in _texts(snapshot, roi=roi) if _normalized(item["text"]) == _normalized(label)]


def parse_progress(snapshot, adapter: ProgressAdapter, task: TaskSpec) -> GoalProgress:
    """A page title plus active/accepted marker guards the exact progress ROI."""
    if len(_exact(snapshot, adapter.title, adapter.title_roi)) != 1:
        raise FlowError(f"Cannot confirm progress page: {adapter.title}")
    if adapter.kind != task.kind or task.goal.building_type and adapter.building_type != task.goal.building_type:
        raise FlowError("Progress adapter does not match the selected task")
    if adapter.unavailable_label and len(_exact(snapshot, adapter.unavailable_label)) == 1:
        return GoalProgress(False, reason="event_inactive" if task.kind == "event" else "no_accepted_clan_challenge",
                            evidence=(snapshot.screenshot_path,), available=False)
    complete = bool(adapter.completed_label and len(_exact(snapshot, adapter.completed_label)) == 1)
    marker = adapter.accepted_label if adapter.kind == "clan_games" else adapter.active_label
    if len(_exact(snapshot, marker)) != 1 and not complete:
        raise FlowError("Cannot verify active-event or accepted-challenge status")
    readings = []
    for item in _texts(snapshot, roi=adapter.progress_roi):
        match = re.fullmatch(adapter.progress_pattern, item["text"].strip())
        if match:
            try:
                readings.append(tuple(int(re.sub(r"[, ]", "", match.group(key))) for key in ("current", "target")))
            except ValueError:
                continue
    if len(readings) != 1 or readings[0][1] <= 0 or not 0 <= readings[0][0] <= readings[0][1]:
        raise FlowError("Goal progress unreadable or ambiguous")
    current, game_target = readings[0]
    target = game_target if task.kind == "clan_games" else task.goal.target or game_target
    if target > game_target:
        raise ConfigError("Configured event target exceeds the displayed progress track")
    if task.kind == "clan_games" and task.goal.target is not None and task.goal.target != game_target:
        raise ConfigError("Clan target must match the accepted challenge, not a partial custom count")
    if complete and current < game_target:
        raise FlowError("Completion label and numeric progress disagree")
    return GoalProgress(current >= target, current, target, "goal_reached" if current >= target else "in_progress",
                        (snapshot.screenshot_path,), {"adapter_id": adapter.id, "game_target": game_target,
                            "building_type": adapter.building_type, "scoring_condition": adapter.scoring_condition,
                            "scoring_units": adapter.scoring_units})


def resource_progress(snapshot, task: TaskSpec) -> GoalProgress:
    if (snapshot.scene != "village" or not math.isfinite(snapshot.confidence) or snapshot.confidence < .8
            or snapshot.observations.get("resource_source") != "village_inventory"):
        raise FlowError("Goal needs current village inventory, not opponent or settlement resources")
    targets = dict(task.goal.resource_targets)
    resources = snapshot.observations.get("resources") or {}
    capacities = snapshot.observations.get("resource_capacities") or {}
    for name in task.goal.full_resources:
        capacity = capacities.get(name)
        if type(capacity) is not int or capacity <= 0:
            raise CapabilityUnavailable(f"Cannot read {name} storage capacity; use a numeric stock target")
        targets[name] = max(targets.get(name, 0), capacity)
    if not targets:
        return GoalProgress(False, reason="bounded_battles_only", evidence=(snapshot.screenshot_path,), values=resources)
    observed = {name: resources.get(name) for name in targets}
    if any(type(value) is not int or value < 0 for value in observed.values()):
        raise FlowError("Required village resource amount unreadable")
    completed = all(observed[name] >= target for name, target in targets.items())
    return GoalProgress(completed, observed, targets, "goal_reached" if completed else "in_progress",
                        (snapshot.screenshot_path,), resources)


def _resource_signature(snapshot, task: TaskSpec):
    """Relevant inventory and capacity values, or None while their counters animate."""
    if (snapshot.scene != "village" or not math.isfinite(snapshot.confidence) or snapshot.confidence < .8
            or snapshot.observations.get("resource_source") != "village_inventory"):
        raise FlowError("Goal needs current village inventory, not opponent or settlement resources")
    resources = snapshot.observations.get("resources") or {}
    capacities = snapshot.observations.get("resource_capacities") or {}
    names = set(task.goal.resource_targets) | set(task.goal.full_resources)
    amounts = tuple(resources.get(name) for name in sorted(names))
    full = tuple(capacities.get(name) for name in sorted(task.goal.full_resources))
    if any(type(value) is not int or value < 0 for value in amounts):
        return None
    if any(type(value) is not int or value <= 0 for value in full):
        return None
    return amounts, full


def _stable_resource_progress(session, task: TaskSpec, first):
    """Wait through the counter animation, then require three matching fresh readings."""
    from .flow import return_to_village
    signatures = [_resource_signature(first, task)]
    latest = first
    seen_frames = {first.screenshot_path}
    evidence = [first.screenshot_path]
    for _ in range(4):
        # A capture loop alone can sample the same early zero repeatedly.
        for _ in range(5):
            session.check_deadline()
            time.sleep(.1)
        latest = return_to_village(session)
        if latest.screenshot_path in seen_frames:
            raise FlowError("Inventory verification requires a fresh screenshot")
        seen_frames.add(latest.screenshot_path)
        evidence.append(latest.screenshot_path)
        signatures.append(_resource_signature(latest, task))
    if signatures[-1] is not None and signatures[-1] == signatures[-2] == signatures[-3]:
        return replace(resource_progress(latest, task), evidence=tuple(evidence))
    if signatures[-1] is None:
        # Preserve the established distinction between unreadable stock and unknown capacity.
        resource_progress(latest, task)
    raise FlowError("Village inventory did not stabilize across fresh observations")


def _observe_page(session, title: str, title_roi, label: str):
    """Re-observe transitions without reissuing the click or Back action."""
    for attempt in range(3):
        session.check_deadline()
        current = session.observe(label if attempt == 0 else label + "-reread")
        if current.scene in {"maintenance", "disconnected"}:
            raise FlowError(f"Game interruption while reading progress: {current.scene}")
        if len(_exact(current, title, title_roi)) == 1:
            anchored = replace(current, scene="goal_panel", confidence=.95)
            session.last_snapshot = anchored
            return anchored
    raise FlowError(f"Cannot confirm page {title!r}; no further input issued")


def _verified_village(snapshot):
    controls = {button.get("name") for button in snapshot.observations.get("buttons", [])
                if isinstance(button, dict)}
    return (snapshot.scene == "village" and math.isfinite(snapshot.confidence)
            and snapshot.confidence >= .8 and {"attack", "shop"} <= controls)


def _observe_previous_or_home(session, step: NavigationStep):
    for attempt in range(3):
        session.check_deadline()
        current = session.observe("goal-return-page" if attempt == 0 else "goal-return-page-reread")
        if current.scene in {"maintenance", "disconnected"}:
            raise FlowError(f"Game interruption while returning from progress: {current.scene}")
        if _verified_village(current):
            return None, current
        if len(_exact(current, step.page_title, step.title_roi)) == 1:
            anchored = replace(current, scene="goal_panel", confidence=.95)
            session.last_snapshot = anchored
            return anchored, None
    raise FlowError(f"Cannot confirm previous page {step.page_title!r} or village; no further input issued")


def read_progress(session, task: TaskSpec) -> GoalProgress:
    from .flow import return_to_village
    home = return_to_village(session)
    if task.kind == "resources":
        if not task.goal.resource_targets and not task.goal.full_resources:
            return resource_progress(home, task)
        return _stable_resource_progress(session, task, home)
    if not task.goal.adapter_path:
        raise CapabilityUnavailable("No calibrated progress adapter selected for this task")
    path = Path(task.goal.adapter_path)
    if not path.is_absolute():
        path = session.config.source_path.resolve().parent / path
    adapter = load_adapter(path)
    if adapter.kind != task.kind:
        raise ConfigError("Adapter kind does not match task kind")
    if adapter.entry_text:
        entries = _exact(home, adapter.entry_text, adapter.entry_roi)
        if len(entries) != 1:
            raise CapabilityUnavailable(f"Task entry unavailable: {adapter.entry_text}")
        left, top, right, bottom = entries[0]["bbox"]
        session.tap(home, ((left + right) // 2, (top + bottom) // 2), reason=f"Open progress: {adapter.id}")
    else:
        template = session.config.vision.template_dir / f"{adapter.entry_template}.png"
        if not template.is_file():
            raise CapabilityUnavailable(f"Progress entry sample unavailable: {adapter.entry_template}")
        session.click_template(home, adapter.entry_template, roi=adapter.entry_roi)
    first_title = adapter.navigation[0].page_title if adapter.navigation else adapter.title
    first_roi = adapter.navigation[0].title_roi if adapter.navigation else adapter.title_roi
    current = _observe_page(session, first_title, first_roi, "goal-page")
    for index, step in enumerate(adapter.navigation):
        if step.section_label and len(_exact(current, step.section_label, step.section_roi)) != 1:
            raise FlowError(f"Cannot confirm unique section {step.section_label!r}; no further input issued")
        controls = _exact(current, step.open_text, step.open_roi)
        if len(controls) != 1:
            raise FlowError(f"Expected one read-only {step.open_text!r} control in its section, found {len(controls)}")
        left, top, right, bottom = controls[0]["bbox"]
        session.tap(current, ((left + right) // 2, (top + bottom) // 2),
                    reason=f"Open verified progress page: {adapter.id}, step {index + 1}")
        next_step = adapter.navigation[index + 1] if index + 1 < len(adapter.navigation) else None
        next_title = next_step.page_title if next_step else adapter.title
        next_roi = next_step.title_roi if next_step else adapter.title_roi
        current = _observe_page(session, next_title, next_roi, f"goal-page-{index + 2}")
    pending = None
    try:
        progress = parse_progress(current, adapter, task)
    except (CapabilityUnavailable, ConfigError, FlowError) as exc:
        pending = exc
    if adapter.close_text:
        closes = _exact(current, adapter.close_text)
        if len(closes) != 1:
            raise FlowError("Cannot identify the progress panel close control")
        l, t, r, b = closes[0]["bbox"]
        session.tap(current, ((l + r) // 2, (t + b) // 2), reason="Close verified progress panel")
    else:
        session.back(current, reason="Close verified progress panel")
    home_after = None
    for step in reversed(adapter.navigation):
        previous, home_after = _observe_previous_or_home(session, step)
        if home_after is not None:
            break
        session.back(previous, reason=f"Return from verified progress page: {step.page_title}")
    if home_after is None:
        home_after = session.wait_for({"village"}, timeout_sec=20, label="goal-return-home")
    return_to_village(session, initial_snapshot=home_after)
    if pending:
        raise pending
    return progress
