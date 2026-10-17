"""Picker collector checks use fake sessions and never connect to a device."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from autococ import picker_collection as picker
from scripts.vision_data import parser as vision_parser
from training.vision.data import DatasetError
from training.vision.picker_candidates import import_picker_candidates


def _shot(name: str, surface: str, kind: str | None = None):
    frame = Path(name)
    counts = {"heroes": (1, 1), "troops": (2, 10), "spells": (1, 2), "siege": (1, 1)}
    manifest = {"frame": str(frame), "layout": "my_army_rows_v1", "supported_layout": True,
                "complete": True, "identity_complete": True, "unknowns": [],
                "complete_kinds": {"troop": True, "spell": True},
                "troops": [{"kind": "troop", "source": "army", "unit_id": "dragon",
                            "count": 2, "bbox": [1, 2, 3, 4]}],
                "spells": [{"kind": "spell", "source": "army", "unit_id": "lightning",
                            "count": 1, "bbox": [2, 3, 4, 5]}]}
    army = {name: {"used": used, "capacity": total} for name, (used, total) in counts.items()}
    army.update({"manifest": manifest, "identity_coverage": {"hero": True, "siege": True},
                 "hero_evidence": {"frame": str(frame), "complete": True, "unknowns": []},
                 "siege_evidence": {"frame": str(frame), "complete": True, "unknowns": []},
                 "identity_cards": [{"kind": "hero", "unit_id": "queen", "count": 1,
                                     "bbox": [1, 1, 4, 4]},
                                    {"kind": "siege", "unit_id": "wagon", "count": 1,
                                     "bbox": [4, 4, 8, 8]}]})
    controls = ([{"action": "open_picker", "kind": k, "point": [606, 240],
                  "enabled": True, "cost_free": True} for k in ("troop", "spell")]
                if surface == "current" else
                [{"action": "close_picker", "point": [800, 320], "enabled": True, "cost_free": True}])
    editor = {"frame": str(frame), "surface": surface, "ready": True,
              "editing_kind": kind, "controls": controls,
              "capacities": {"troop": {"used": 2, "total": 10}, "spell": {"used": 1, "total": 2}}}
    if surface == "current_picker":
        editor.update({"cards": copy.deepcopy(manifest[kind + "s"]),
                       "complete_kinds": {kind: True}, "unknowns": []})
    return SimpleNamespace(scene="training", screenshot_path=frame,
                           observations={"army": None if kind == "spell" else army,
                                         "army_editor": editor,
                                         **{name: None for name in (
                                             "account_in_use", "cloud_cover", "request_dialog",
                                             "surrender_dialog", "unit_detail", "spell_detail",
                                             "hero_detail", "startup_logo")}})


class FakeSession:
    def __init__(self, shots):
        self.sequence = list(shots)
        self.config = SimpleNamespace(game=SimpleNamespace(baseline_resolution=picker.BASE))
        self.client_version = picker.VERSION
        self.client_version_code = picker.VERSION_CODE
        self.action_count = 0
        self._input_attempt_generation = 0
        self.last_snapshot = None
        self.actions = []
        self.stop_after_page = False
        self.stop_flag = False
        self.context = SimpleNamespace(swipe=self.swipe)

    def observe(self, _label):
        self.last_snapshot = self.sequence.pop(0)
        return self.last_snapshot

    def _validate_snapshot(self, shot):
        assert shot is self.last_snapshot

    def check_deadline(self):
        pass

    def tap(self, shot, point, *, reason):
        self._validate_snapshot(shot)
        self.actions.append(("tap", reason))
        self.action_count += 1

    def event(self, kind, **_data):
        self.actions.append(("event", kind))
        if self.stop_after_page and kind == "development_readonly_picker_page":
            self.stop_flag = True

    def swipe(self, *_args):
        self.actions.append(("swipe",))


def _open_army(session, home, _evidence):
    session._validate_snapshot(home)
    session.action_count += 2
    current = session.sequence.pop(0)
    session.last_snapshot = current
    return current


def _run(session, *, stop=lambda: False, foreground=lambda: True):
    images = {name: np.zeros((720, 1280, 3), dtype=np.uint8)
              for name in ("before.png", "troop1.png", "troop2.png", "closed.png",
                           "spell.png", "after.png")}
    images["troop2.png"][430:690, 30:500] = 170
    with patch.object(picker, "_calibration", return_value={"next_arrow_template": {"path": "next-arrow.png"}}), \
         patch.object(picker, "_image", side_effect=lambda path: images[str(path)]), \
         patch.object(picker, "_arrow_score", return_value=.99), \
         patch.object(picker, "sha", side_effect=lambda path: "sha:" + str(path)):
        return picker.run_flow(session, foreground=foreground, open_army=_open_army,
                               max_troop_pages=2, pause=lambda _: None, stop=stop)


def _session():
    home = SimpleNamespace(scene="village", screenshot_path=Path("home.png"), observations={})
    return FakeSession([home, _shot("before.png", "current"),
                        _shot("troop1.png", "current_picker", "troop"),
                        _shot("troop2.png", "current_picker", "troop"),
                        _shot("closed.png", "current"),
                        _shot("spell.png", "current_picker", "spell"),
                        _shot("after.png", "current")])


def test_calibration_asset_is_local_and_hash_bound():
    calibration = picker._calibration()
    assert calibration["schema"] == "readonly_picker_calibration_v1"
    assert calibration["swipe_start"] == [980, 560]
    assert calibration["source_troop_swipe_events"][0]["sha256"].startswith("c18dcabd")
    assert not Path(calibration["next_arrow_template"]["path"]).is_absolute()


def test_collect_picker_cli_arguments_are_explicit():
    args = vision_parser().parse_args(["collect-picker", "--config", "config.toml",
                                        "--output", "new-run"])
    assert args.command == "collect-picker"
    assert args.max_troop_pages == 3 and args.limit_sec == 150


def test_two_troop_pages_spell_first_and_unchanged_army():
    session = _session()
    result = _run(session)
    assert result["zero_roster_change_verified"] is True
    assert len(result["troop_pages"]) == 2
    assert result["spell_first_page"]["swipe_attempted"] is False
    assert [a[0] for a in session.actions].count("swipe") == 1
    assert session.action_count == 7


def test_stop_before_swipe_sends_no_more_input():
    session = _session()
    session.stop_after_page = True
    with pytest.raises(picker.SamplingStopped, match="stopped_by_user"):
        _run(session, stop=lambda: session.stop_flag)
    assert [a[0] for a in session.actions].count("swipe") == 0
    assert [a[0] for a in session.actions].count("tap") == 1


@pytest.mark.parametrize("completed_inputs", [2, 3, 4, 5, 6])
def test_stop_before_each_remaining_picker_input(completed_inputs: int):
    session = _session()
    with pytest.raises(picker.SamplingStopped, match="stopped_by_user"):
        _run(session, stop=lambda: session.action_count >= completed_inputs)
    assert session.action_count == completed_inputs


def test_stop_arriving_during_foreground_probe_blocks_tap():
    session = _session()
    current_probes = 0
    def foreground():
        nonlocal current_probes
        if session.action_count == 2:
            current_probes += 1
            if current_probes == 2:
                session.stop_flag = True
        return True
    with pytest.raises(picker.SamplingStopped, match="stopped_by_user"):
        _run(session, stop=lambda: session.stop_flag, foreground=foreground)
    assert session.action_count == 2


def test_existing_output_is_rejected_without_touching_it(tmp_path: Path):
    output = tmp_path / "already-exists"
    output.mkdir()
    marker = output / "user-file.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="existing"):
        picker.collect_picker(tmp_path / "config.toml", output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_dangling_symlink_output_is_rejected(tmp_path: Path):
    link = tmp_path / "output-link"
    try:
        os.symlink(tmp_path / "not-created", link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    with pytest.raises(ValueError, match="existing"):
        picker.collect_picker(tmp_path / "config.toml", link)
    assert link.is_symlink()
    assert not (tmp_path / "not-created").exists()


def test_calibration_directory_symlink_cannot_escape_project(tmp_path: Path):
    fake_repo = tmp_path / "checkout"
    fake_dir = fake_repo / "assets" / "picker" / picker.VERSION
    fake_dir.parent.mkdir(parents=True)
    try:
        os.symlink(picker.CALIBRATION_DIR, fake_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    with patch.object(picker, "REPO_ROOT", fake_repo), patch.object(picker, "CALIBRATION_DIR", fake_dir):
        with pytest.raises(picker.SamplingStopped, match="escapes_project"):
            picker._calibration()


def test_foreign_clean_repo_override_is_rejected_before_output_or_device(tmp_path: Path):
    foreign_repo = tmp_path / "clean-checkout"
    foreign_repo.mkdir()
    output = tmp_path / "new-picker-run"
    with patch.object(picker, "_git") as git_probe:
        with pytest.raises(ValueError, match="loaded picker collector checkout"):
            picker.collect_picker(tmp_path / "config.toml", output, repo_root=foreign_repo)
    git_probe.assert_not_called()
    assert not output.exists()


def test_foreign_override_cannot_allow_output_inside_loaded_checkout(tmp_path: Path):
    foreign_repo = tmp_path / "other-checkout"
    foreign_repo.mkdir()
    output = picker.REPO_ROOT / f"picker-should-not-create-{tmp_path.name}"
    assert not output.exists()
    with patch.object(picker, "_git") as git_probe:
        with pytest.raises(ValueError, match="loaded picker collector checkout"):
            picker.collect_picker(tmp_path / "config.toml", output, repo_root=foreign_repo)
    git_probe.assert_not_called()
    assert not output.exists()


def test_failed_collect_preserves_raw_and_cannot_import(tmp_path: Path):
    from autococ import config as config_module, device as device_module, session as session_module

    repo = tmp_path / "checkout"
    for name in picker.SOURCE_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source")
    collector_copy = repo / "src" / "autococ" / "picker_collection.py"
    collector_copy.write_bytes(Path(picker.__file__).read_bytes())
    calibration_dir = repo / "assets" / "picker" / picker.VERSION
    calibration_dir.mkdir(parents=True)
    for name in ("calibration.json", "next-arrow.png"):
        (calibration_dir / name).write_bytes((picker.CALIBRATION_DIR / name).read_bytes())
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(b"config")
    output = tmp_path / "raw-picker-run"
    config = SimpleNamespace(game=SimpleNamespace(baseline_resolution=picker.BASE,
                                                  package_name=picker.PACKAGE),
                             runtime=SimpleNamespace(step_timeout_sec=5))
    manager = SimpleNamespace(adb=object(), connect=lambda: SimpleNamespace(serial="fake"))

    def interrupted_connect(*_args, **_kwargs):
        frame_dir = output / "frames"
        frame_dir.mkdir()
        (frame_dir / "00001-home.png").write_bytes(b"raw bytes already captured")
        raise picker.SamplingStopped("stopped_by_user")

    with patch.object(picker, "_git", side_effect=lambda _repo, *args: "head" if args[0] == "rev-parse" else ""), \
         patch.object(picker, "REPO_ROOT", repo), \
         patch.object(picker, "CALIBRATION_DIR", calibration_dir), \
         patch.object(picker, "__file__", str(collector_copy)), \
         patch.object(config_module, "load_config", return_value=config), \
         patch.object(device_module, "DeviceManager", return_value=manager), \
         patch.object(device_module, "foreground_package", return_value=picker.PACKAGE), \
         patch.object(session_module.GameSession, "connect", side_effect=interrupted_connect):
        result = picker.collect_picker(config_path, output)
    assert result["status"] == "failed"
    assert (output / "frames" / "00001-home.png").read_bytes() == b"raw bytes already captured"
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"] == "failed"
    with pytest.raises(DatasetError):
        import_picker_candidates(output, tmp_path / "queue")
    assert not (tmp_path / "queue").exists()


def test_navigation_stop_between_inputs():
    calls = []
    session = SimpleNamespace(scene="village", client_version=picker.VERSION,
                              check_deadline=lambda: None,
                              click=lambda _shot, action: calls.append(action),
                              wait_for=lambda *_args, **_kwargs: SimpleNamespace(screenshot_path=Path("menu.png")))
    home = SimpleNamespace(scene="village")
    with pytest.raises(picker.SamplingStopped):
        picker._open_army_checked(session, home, [], foreground=lambda: True,
                                  stop=lambda: bool(calls))
    assert calls == ["attack"]
