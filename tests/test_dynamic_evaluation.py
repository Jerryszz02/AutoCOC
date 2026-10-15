import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from autococ.dynamic_vision import DynamicObservation, DynamicUnavailable
from training.vision.data import DatasetError, sha256
from training.vision.dynamic_data import verify_dynamic_windows
from training.vision.dynamic_evaluation import evaluate_dynamic
from training.vision.dynamic_snapshot import freeze_dynamic_snapshot, verify_dynamic_snapshot
from scripts.evaluate_dynamic_vision import main as evaluate_dynamic_cli


class FakeModel:
    kind = "r_building"
    model_sha256 = "a" * 64
    contract_sha256 = "b" * 64
    event_rules = {"consecutive_frames": 2, "min_iou": .4, "max_gap_seconds": 2.0,
                   "max_missed_frames": 1, "event_min_confidence": .95,
                   "max_frame_age_seconds": .5}

    def __init__(self, labels):
        self.labels = labels

    def infer(self, image, roi_bbox, frame, *, target_id, offline_replay):
        assert offline_replay
        label = self.labels[(frame.session_id, frame.frame_id)]
        if label == "fail":
            raise DynamicUnavailable("synthetic inference failure")
        if label == "missing":
            return []
        return [DynamicObservation(self.kind, label, tuple(roi_bbox), .99,
                frame.session_id, frame.camera_epoch, frame.frame_id,
                frame.captured_at_monotonic, tuple(roi_bbox), target_id,
                self.model_sha256, self.contract_sha256)]


def window(root, name, truth, *, group=None, split="test"):
    frames = []
    for index in (1, 2, 3):
        path = root / f"{name}-{index}.png"
        path.write_bytes(cv2.imencode(".png", np.full((20, 20, 3), index, np.uint8))[1].tobytes())
        frames.append({"image": path.name, "sha256": sha256(path),
                       "parent_image_sha256": sha256(path), "session_id": name,
                       "battle_id": f"battle-{name}", "layout_id": f"layout-{name}",
                       "camera_epoch": "camera-1",
                       "frame_id": f"f{index}", "captured_at_monotonic": float(index),
                       "captured_at_utc": "2020-01-01T00:00:00+00:00",
                       "client_version": "18.647.2", "roi_bbox": [2, 2, 18, 18],
                       "target_id": "hall-1", "camera_stable": True,
                       "zoom_verified": True,
                       "state_label": "intact" if index == 1 or not truth else "rubble",
                       "state_review_status": "reviewed", "state_review_method": "agent_visual",
                       "state_reviewer": "fixture", "state_review_evidence": "independent visible state",
                       "state_target_id": "hall-1", "state_source_frame_sha256": sha256(path),
                       "state_roi_bbox": [2, 2, 18, 18]})
    return {"schema_version": 1, "kind": "r_building", "window_id": name,
            "event_group_id": group or name, "split": split, "review_status": "reviewed",
            "reviewer": "fixture", "review_method": "agent_visual",
            "review_evidence": "fixture's independent truth", "prototype_only": True,
            "action_kind": "building_destroyed", "truth_event": truth,
            "event_at_monotonic": 2.0 if truth else None, "frames": frames}


def u_window(root, name="position", *, boxes=None):
    row = window(root, name, False)
    row["kind"] = "u_position"
    row["truth_boxes"] = [] if boxes is None else boxes
    for frame in row["frames"]:
        frame.update({"boxes_complete": True, "boxes_review_status": "reviewed",
                      "boxes_review_method": "agent_visual", "boxes_reviewer": "fixture",
                      "boxes_review_evidence": "reviewed every target in source ROI",
                      "boxes_source_frame_sha256": frame["parent_image_sha256"],
                      "boxes_roi_bbox": frame["roi_bbox"][:]})
    return row


def test_u_reviewed_empty_boxes_and_source_bound_positive_freeze():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "source"
        source.mkdir()
        reviewed = u_window(source, boxes=[{"frame_id": "f2", "label": "hero:king",
                                            "bbox": [3, 3, 12, 12]}])
        frozen = root / "frozen"
        result = freeze_dynamic_snapshot(source, [reviewed], frozen,
                                         kind="u_position", prototype_only=True)
        assert result["metadata"]["classes"] == ["hero:king"]
        assert verify_dynamic_snapshot(frozen, kind="u_position")["snapshot_sha256"] == result["snapshot_sha256"]
        negative = u_window(source, "negative")
        assert verify_dynamic_windows(source, [negative], kind="u_position",
                                      prototype_only=True)["frames"] == 3


def test_u_rejects_incomplete_unbound_or_outside_roi_boxes():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        good = u_window(root, boxes=[{"frame_id": "f2", "label": "troop_group",
                                      "bbox": [3, 3, 12, 12]}])
        for field, value in (("boxes_complete", False), ("boxes_review_status", None),
                             ("boxes_review_method", None), ("boxes_reviewer", ""),
                             ("boxes_review_evidence", ""),
                             ("boxes_source_frame_sha256", "0" * 64),
                             ("boxes_roi_bbox", [0, 0, 20, 20])):
            bad = copy.deepcopy(good)
            bad["frames"][1][field] = value
            with pytest.raises(DatasetError, match="complete source-bound reviewed boxes"):
                verify_dynamic_windows(root, [bad], kind="u_position", prototype_only=True)
        bad = copy.deepcopy(good)
        bad["truth_boxes"][0]["bbox"] = [0, 0, 12, 12]  # inside pixels, outside reviewed ROI
        with pytest.raises(DatasetError, match="outside reviewed ROI"):
            verify_dynamic_windows(root, [bad], kind="u_position", prototype_only=True)
        bad = copy.deepcopy(good)
        bad["frames"][1]["frame_id"] = "f1"
        with pytest.raises(DatasetError, match="repeated source frame_id"):
            verify_dynamic_windows(root, [bad], kind="u_position", prototype_only=True)
        formal = copy.deepcopy(good)
        formal["prototype_only"] = False
        formal["review_method"] = "human"
        with pytest.raises(DatasetError, match="complete source-bound reviewed boxes"):
            verify_dynamic_windows(root, [formal], kind="u_position", prototype_only=False)


def test_event_recall_false_trigger_and_failure_remain_in_denominator():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        windows = [window(root, "positive", True), window(root, "negative", False),
                   window(root, "failure", True)]
        # Distinct battle fixtures must not reuse byte-identical source frames
        # while claiming different reviewed states for the same ROI and target.
        for offset, reviewed in enumerate(windows):
            for index, frame in enumerate(reviewed["frames"], 1):
                source = root / frame["image"]
                source.write_bytes(cv2.imencode(".png", np.full((20, 20, 3),
                                   index + offset * 20, np.uint8))[1].tobytes())
                frame["sha256"] = frame["parent_image_sha256"] = sha256(source)
                frame["state_source_frame_sha256"] = frame["sha256"]
        labels = {("positive", "f1"): "intact", ("positive", "f2"): "rubble",
                  ("positive", "f3"): "rubble", ("negative", "f1"): "intact",
                  ("negative", "f2"): "rubble", ("negative", "f3"): "rubble",
                  ("failure", "f1"): "intact", ("failure", "f2"): "fail",
                  ("failure", "f3"): "rubble"}
        report = evaluate_dynamic(FakeModel(labels), root, windows, prototype_only=True)
        assert report["status"] == "offline_only"
        assert report["metrics"]["tp"] == 1
        assert report["metrics"]["fp"] == 1
        assert report["metrics"]["fn"] == 1
        assert report["metrics"]["event_recall"] == .5
        assert report["metrics"]["inference_failures"] == 1
        assert "No capture-to-input" in report["limits"][1]


def test_empty_selected_split_fails_api_and_cli_without_writing_report(monkeypatch, capsys):
    with TemporaryDirectory() as temp:
        root = Path(temp)
        train = window(root, "train", False, split="train")
        validation = window(root, "validation", False, split="val")
        for index, frame in enumerate(validation["frames"], 1):
            source = root / frame["image"]
            source.write_bytes(cv2.imencode(".png", np.full((20, 20, 3), index + 20,
                               np.uint8))[1].tobytes())
            frame["sha256"] = frame["parent_image_sha256"] = sha256(source)
            frame["state_source_frame_sha256"] = frame["sha256"]
        windows = [train, validation]
        with pytest.raises(DatasetError, match="split 'test' has no reviewed windows"):
            evaluate_dynamic(FakeModel({}), root, windows, prototype_only=True)
        windows_path = root / "windows.jsonl"
        windows_path.write_text("\n".join(json.dumps(row) for row in windows) + "\n", encoding="utf-8")
        monkeypatch.setattr("scripts.evaluate_dynamic_vision.DynamicModel", lambda *_args, **_kwargs: FakeModel({}))
        output = root / "report.json"
        assert evaluate_dynamic_cli([str(root / "package"), "--client-version", "18.647.2",
                                     "--source-root", str(root), "--windows", str(windows_path),
                                     "--prototype-only", "--output", str(output)]) == 2
        assert "no reviewed windows" in capsys.readouterr().err
        assert not output.exists()


def test_group_leakage_and_agent_review_cannot_be_formal():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        first = window(root, "first", True, group="same", split="train")
        second = window(root, "second", False, group="same", split="test")
        with pytest.raises(DatasetError, match="crosses"):
            verify_dynamic_windows(root, [first, second], kind="r_building", prototype_only=True)
        formal = copy.deepcopy(first)
        formal["prototype_only"] = False
        with pytest.raises(DatasetError, match="review method"):
            verify_dynamic_windows(root, [formal], kind="r_building", prototype_only=False)


def test_source_graph_blocks_new_group_id_and_window_id_leakage():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        first = window(root, "first", True, split="train")
        second = window(root, "second", False, split="test")
        # Different window/group/session IDs are still the same battle/layout.
        for frame in second["frames"]:
            frame["battle_id"] = first["frames"][0]["battle_id"]
            frame["layout_id"] = first["frames"][0]["layout_id"]
        with pytest.raises(DatasetError, match="source-linked"):
            verify_dynamic_windows(root, [first, second], kind="r_building", prototype_only=True)
        for frame in second["frames"]:
            frame["battle_id"] = "different-battle"
            frame["layout_id"] = "different-layout"
        # The fixture's equal pixels have the same source SHA, also linking splits.
        with pytest.raises(DatasetError, match="source-linked"):
            verify_dynamic_windows(root, [first, second], kind="r_building", prototype_only=True)


def test_r_conflicting_state_on_same_source_roi_target_rejected_across_windows():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        first = window(root, "first", False, split="train")
        second = window(root, "second", True, split="train")
        # IDs and capture times may differ, while identical source pixels/ROI/target
        # are still one supervised state and cannot have contradictory labels.
        with pytest.raises(DatasetError, match="Conflicting R state labels"):
            verify_dynamic_windows(root, [first, second], kind="r_building", prototype_only=True)
        second = window(root, "second", False, split="train")
        assert verify_dynamic_windows(root, [first, second], kind="r_building",
                                      prototype_only=True)["frames"] == 6
        for frame in second["frames"]:
            frame["roi_bbox"] = [3, 3, 18, 18]
            frame["state_roi_bbox"] = [3, 3, 18, 18]
            frame["state_label"] = "damaged"
        assert verify_dynamic_windows(root, [first, second], kind="r_building",
                                      prototype_only=True)["frames"] == 6


def test_evaluated_groups_count_source_components_not_event_group_labels():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        first = window(root, "first", False, group="label-one")
        second = window(root, "second", False, group="label-two")
        for frame in second["frames"]:
            frame["battle_id"] = first["frames"][0]["battle_id"]
            frame["layout_id"] = first["frames"][0]["layout_id"]
        labels = {(name, f"f{index}"): "intact"
                  for name in ("first", "second") for index in (1, 2, 3)}
        report = evaluate_dynamic(FakeModel(labels), root, [first, second], prototype_only=True)
        assert report["evaluated_windows"] == 2
        assert report["evaluated_groups"] == 1
        assert report["inventory"]["groups_by_split"]["test"] == 1


def test_dynamic_windows_reject_nan_future_time_and_roi_outside_source():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        good = window(root, "one", True)
        for field, value, message in (("captured_at_monotonic", float("nan"), "finite"),
                                      ("captured_at_monotonic", math.inf, "finite"),
                                      ("roi_bbox", [0, 0, 100, 100], "ROI outside")):
            bad = copy.deepcopy(good)
            bad["frames"][0][field] = value
            with pytest.raises(DatasetError, match=message):
                verify_dynamic_windows(root, [bad], kind="r_building", prototype_only=True)
        bad = copy.deepcopy(good)
        bad["event_at_monotonic"] = float("nan")
        with pytest.raises(DatasetError, match="finite"):
            verify_dynamic_windows(root, [bad], kind="r_building", prototype_only=True)
        bad = copy.deepcopy(good)
        bad["frames"][0]["captured_at_utc"] = "2999-01-01T00:00:00+00:00"
        with pytest.raises(DatasetError, match="future"):
            verify_dynamic_windows(root, [bad], kind="r_building", prototype_only=True)
        bad = copy.deepcopy(good)
        bad["frames"][0]["parent_image_sha256"] = "0" * 64
        with pytest.raises(DatasetError, match="parent source"):
            verify_dynamic_windows(root, [bad], kind="r_building", prototype_only=True)


def test_parent_roi_uses_verified_original_frame_coordinates():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        reviewed = window(root, "roi", True)
        labels = {}
        for index, frame in enumerate(reviewed["frames"], 1):
            original = root / f"parent-{index}.png"
            original.write_bytes(cv2.imencode(".png", np.full((30, 30, 3), index, np.uint8))[1].tobytes())
            frame["parent_image"] = original.name
            frame["parent_image_sha256"] = sha256(original)
            frame["roi_bbox"] = [2, 2, 25, 25]  # outside the 20px derivative, inside original
            frame["state_source_frame_sha256"] = sha256(original)
            frame["state_roi_bbox"] = [2, 2, 25, 25]
            labels[("roi", f"f{index}")] = "intact" if index == 1 else "rubble"
        report = evaluate_dynamic(FakeModel(labels), root, [reviewed], prototype_only=True)
        assert report["metrics"]["tp"] == 1
        assert report["metrics"]["inference_failures"] == 0


def test_event_truth_cannot_substitute_for_reviewed_per_frame_r_state():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        reviewed = window(root, "state", True)
        for field, value in (("state_label", None), ("state_label", "available"),
                             ("state_source_frame_sha256", "0" * 64),
                             ("state_roi_bbox", [0, 0, 10, 10]),
                             ("state_review_method", None)):
            bad = copy.deepcopy(reviewed)
            bad["frames"][1][field] = value
            with pytest.raises(DatasetError, match="independently reviewed"):
                verify_dynamic_windows(root, [bad], kind="r_building", prototype_only=True)


def test_frozen_dynamic_snapshot_copies_pixels_and_rechecks_semantic_split():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "source"
        source.mkdir()
        reviewed = window(source, "state", True)
        frozen = root / "frozen"
        result = freeze_dynamic_snapshot(source, [reviewed], frozen,
                                         kind="r_building", prototype_only=True)
        assert result["metadata"]["classes"] == ["intact", "damaged", "rubble", "occlusion_unknown"]
        assert result["metadata"]["groups_by_split"] == {"train": 0, "val": 0, "test": 1}
        assert result["windows"][0]["frames"][0]["image"].startswith("images/")
        assert verify_dynamic_snapshot(frozen, kind="r_building")["snapshot_sha256"] == result["snapshot_sha256"]
        with pytest.raises(DatasetError, match="already exists"):
            freeze_dynamic_snapshot(source, [reviewed], frozen, kind="r_building", prototype_only=True)
        # A signature over deliberately forged split bytes cannot make the labels agree.
        split_path = frozen / "split.json"
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        payload["window_assignments"]["state"] = "train"
        split_path.write_text(json.dumps(payload), encoding="utf-8")
        sums = frozen / "SHA256SUMS"
        sums.write_text("".join(f"{sha256(path)}  {path.relative_to(frozen).as_posix()}\n"
                                for path in sorted(frozen.rglob("*")) if path.is_file() and path != sums),
                        encoding="utf-8")
        with pytest.raises(DatasetError, match="frozen split"):
            verify_dynamic_snapshot(frozen, kind="r_building")
