"""Offline YOLO26 U-position training from fully reviewed frozen ROI windows."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from .data import sha256
from .dynamic_snapshot import verify_dynamic_snapshot
from .snapshot import TrainingError
from .train import _environment, _offline_ultralytics, _signature


U_PREPROCESS = {"width": 640, "height": 640, "color": "RGB", "resize": "stretch",
                "scale": 1 / 255, "mean": [0, 0, 0], "std": [1, 1, 1]}


def u_rows(snapshot: dict) -> dict[str, list[dict]]:
    """Require a complete, source-bound review for every training ROI, including empty ones."""
    rows = {name: [] for name in ("train", "val", "test")}
    seen: dict[tuple[str, tuple[int, ...]], tuple] = {}
    prototype = snapshot["metadata"]["prototype_only"]
    methods = {"human", "agent_visual"} if prototype else {"human"}
    for window in snapshot["windows"]:
        boxes_by_frame: dict[str, list[dict]] = {}
        for box in window["truth_boxes"]:
            boxes_by_frame.setdefault(box["frame_id"], []).append(box)
        frame_ids = [frame["frame_id"] for frame in window["frames"]]
        if len(set(frame_ids)) != len(frame_ids):
            raise TrainingError(f"U window {window['window_id']} reuses a frame_id")
        for frame in window["frames"]:
            roi = frame["roi_bbox"]
            if (frame.get("boxes_complete") is not True or
                    frame.get("boxes_review_status") != "reviewed" or
                    frame.get("boxes_review_method") not in methods or
                    not frame.get("boxes_reviewer") or not frame.get("boxes_review_evidence") or
                    frame.get("boxes_source_frame_sha256") != frame["parent_image_sha256"] or
                    frame.get("boxes_roi_bbox") != roi):
                raise TrainingError(f"U frame {frame['frame_id']} lacks complete source-bound box review")
            boxes = boxes_by_frame.get(frame["frame_id"], [])
            for box in boxes:
                left, top, right, bottom = box["bbox"]
                if not (roi[0] <= left < right <= roi[2] and roi[1] <= top < bottom <= roi[3]):
                    raise TrainingError(f"U box outside fully reviewed ROI: {frame['frame_id']}")
            source = frame["image"] if frame["sha256"] == frame["parent_image_sha256"] else frame["parent_image"]
            key = (frame["parent_image_sha256"], tuple(roi))
            truth = tuple(sorted((box["label"], tuple(box["bbox"])) for box in boxes))
            if key in seen and seen[key] != (window["split"], truth):
                raise TrainingError("U repeated source ROI has conflicting split or truth boxes")
            if key in seen:
                continue
            seen[key] = (window["split"], truth)
            rows[window["split"]].append({"frame": frame, "boxes": boxes, "source": source,
                                           "component": snapshot["inventory"]["window_components"][window["window_id"]]})
    return rows


def u_config(snapshot: dict, path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = snapshot["metadata"]
    if (not isinstance(config, dict) or config.get("kind") != "u_position" or
            config.get("classes") != metadata["classes"] or not metadata["classes"] or
            config.get("client_versions") != metadata["client_versions"] or
            config.get("prototype_only") is not metadata["prototype_only"] or
            config.get("architecture") != "yolo26n" or
            config.get("preprocess") != U_PREPROCESS or
            not isinstance(config.get("code_version"), str) or not config["code_version"].strip()):
        raise TrainingError("U config differs from frozen classes/version/review or fixed YOLO26 ROI contract")
    app, output, rules = (config.get(key) for key in ("applicability", "output", "event_rules"))
    if (not isinstance(app, dict) or app.get("roi_kind") != "u_position" or
            app.get("requires_stable_camera") is not True or
            type(app.get("requires_verified_zoom")) is not bool or
            not isinstance(output, dict) or output.get("format") != "xyxy_score_class" or
            output.get("max_detections") != 300 or not isinstance(rules, dict) or
            rules.get("consecutive_frames") != 2):
        raise TrainingError("U runtime applicability/output/event contract is incomplete")
    for key in ("source_width_range", "source_height_range"):
        bounds = app.get(key)
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                any(type(value) is not int or value < 1 for value in bounds) or bounds[0] > bounds[1]):
            raise TrainingError(f"U {key} invalid")
    age, confidence = config.get("max_frame_age_seconds"), output.get("min_confidence")
    if (type(age) not in (int, float) or not math.isfinite(age) or not 0 < age <= 10 or
            type(confidence) not in (int, float) or not math.isfinite(confidence) or
            not 0 <= confidence <= 1):
        raise TrainingError("U frame age or confidence is invalid")
    try:
        from autococ.dynamic_tracking import DynamicTracker
        tracker = DynamicTracker(**{key: rules[key] for key in
            ("min_iou", "max_gap_seconds", "max_missed_frames", "event_min_confidence", "max_frame_age_seconds")})
        if tracker.max_frame_age_seconds > age:
            raise ValueError("tracker age exceeds model age")
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingError("U event rules are invalid") from exc
    root = Path(snapshot["root"])
    for window in snapshot["windows"]:
        for frame in window["frames"]:
            source = frame["image"] if frame["sha256"] == frame["parent_image_sha256"] else frame["parent_image"]
            image = cv2.imdecode(np.fromfile(root / source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if (image is None or not app["source_width_range"][0] <= image.shape[1] <= app["source_width_range"][1] or
                    not app["source_height_range"][0] <= image.shape[0] <= app["source_height_range"][1]):
                raise TrainingError("U source image outside declared applicability")
    return config


def stage_u_rois(root: Path, output: Path, rows: dict[str, list[dict]], classes: list[str]) -> dict[str, int]:
    staged = output / "staged"
    counts = {}
    for split in ("train", "val"):
        names = []
        for index, row in enumerate(rows[split]):
            frame, roi = row["frame"], row["frame"]["roi_bbox"]
            image = cv2.imdecode(np.fromfile(root / row["source"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise TrainingError("U frozen source cannot be decoded")
            crop = cv2.resize(image[roi[1]:roi[3], roi[0]:roi[2]], (640, 640), interpolation=cv2.INTER_LINEAR)
            name = f"{split}-{index:06d}"
            image_path, label_path = staged / "images" / f"{name}.png", staged / "labels" / f"{name}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), crop):
                raise TrainingError("Could not stage U ROI")
            labels = []
            width, height = roi[2] - roi[0], roi[3] - roi[1]
            for box in row["boxes"]:
                left, top, right, bottom = box["bbox"]
                labels.append(f"{classes.index(box['label'])} {(left+right-2*roi[0])/(2*width):.10f} "
                              f"{(top+bottom-2*roi[1])/(2*height):.10f} {(right-left)/width:.10f} {(bottom-top)/height:.10f}")
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            names.append(str(image_path))
        (output / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        counts[split] = len(names)
    (output / "data.yaml").write_text("path: " + json.dumps(str(staged)) + "\ntrain: " +
        json.dumps(str(output / "train.txt")) + "\nval: " + json.dumps(str(output / "val.txt")) +
        "\nnames:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(classes)), encoding="utf-8")
    return counts


def train_u_position(snapshot_root: Path, config_path: Path, output: Path, *, device: str,
                     pretrained: Path, seed: int = 42, resume: Path | None = None,
                     max_epochs: int = 35, deadline_seconds: float = 4 * 3600) -> dict:
    if not 1 <= max_epochs <= 100 or deadline_seconds <= 0:
        raise TrainingError("U training requires 1..100 epochs and a positive deadline")
    snapshot = verify_dynamic_snapshot(snapshot_root, kind="u_position")
    config = u_config(snapshot, config_path)
    rows = u_rows(snapshot)
    if not rows["train"] or not rows["val"]:
        raise TrainingError("U training needs reviewed train and validation ROI frames")
    if (not config["prototype_only"] and any(set(config["classes"]) -
            {box["label"] for row in rows[split] for box in row["boxes"]} for split in ("train", "val"))):
        raise TrainingError("Formal U training requires every class in train and validation")
    pretrained = Path(pretrained).resolve()
    if not pretrained.is_file() or pretrained.name != "yolo26n.pt":
        raise TrainingError("U training requires a local yolo26n.pt checkpoint")
    batch = int(config.get("batch", 2))
    if not 1 <= batch <= 8:
        raise TrainingError("U batch must be 1..8")
    output = Path(output).resolve()
    if output.exists() and resume is None:
        raise TrainingError(f"U output exists: {output}")
    environment = _environment()
    recipe = {"imgsz": 640, "batch": batch, "optimizer": "AdamW", "lr0": .001,
              "weight_decay": .0005, "patience": 20, "amp": False, "workers": 0,
              "cache": False, "rect": False, "hsv_h": 0., "hsv_s": 0., "hsv_v": 0.,
              "degrees": 0., "translate": 0., "scale": 0., "shear": 0.,
              "perspective": 0., "flipud": 0., "fliplr": 0., "mosaic": 0.,
              "mixup": 0., "copy_paste": 0., "save": True, "val": True,
              "deterministic": True}
    config.update({"snapshot_sha256": snapshot["snapshot_sha256"], "split_sha256": snapshot["split_sha256"],
                   "pretrained_sha256": sha256(pretrained), "seed": seed, "device": device,
                   "batch": batch, "max_epochs": max_epochs, "deadline_seconds": deadline_seconds})
    signature = _signature({"config": {k: v for k, v in config.items() if k != "deadline_seconds"},
                            "recipe": recipe, "ultralytics_version": environment["packages"]["ultralytics"],
                            "training_recipe_version": 1})
    resumable, metadata_path = output / "resume.pt", output / "resume.metadata.json"
    if resume is not None:
        resume = Path(resume).resolve()
        if resume != resumable or not resume.is_file() or not metadata_path.is_file():
            raise TrainingError("U resume requires this run's resume.pt and metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature or metadata.get("checkpoint_sha256") != sha256(resume):
            raise TrainingError("U resume config/snapshot/version/checkpoint differs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    counts = stage_u_rois(Path(snapshot["root"]), output, rows, config["classes"])
    YOLO = _offline_ultralytics(output / "ultralytics_config")
    import torch
    torch.set_num_threads(2)
    model = YOLO(str(resume or pretrained))
    args = {"data": str(output / "data.yaml"), "epochs": max_epochs, "device": device,
            "project": str(output), "name": "run", "exist_ok": bool(resume),
            "resume": bool(resume), "seed": seed, **recipe}
    (output / "effective_train_args.json").write_text(json.dumps(args, indent=2), encoding="utf-8")
    began = time.monotonic()
    def preserve(trainer):
        shutil.copyfile(trainer.last, resumable)
        metadata_path.write_text(json.dumps({"signature": signature, "checkpoint_sha256": sha256(resumable),
            "snapshot_sha256": snapshot["snapshot_sha256"], "split_sha256": snapshot["split_sha256"],
            "trainer_version": environment["packages"]["ultralytics"]}, indent=2), encoding="utf-8")
    def stop(trainer):
        if time.monotonic() - began >= deadline_seconds:
            trainer.stop = True
    model.add_callback("on_model_save", preserve)
    model.add_callback("on_fit_epoch_end", stop)
    model.train(**args)
    csv = output / "run" / "results.csv"
    epochs = max(0, len(csv.read_text(encoding="utf-8").splitlines()) - 1) if csv.is_file() else 0
    best_checkpoint = output / "run" / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise TrainingError("U training ended without a best checkpoint")
    report = {"kind": "u_position", "prototype_only": config["prototype_only"],
              "train_frames": counts["train"], "validation_frames": counts["val"],
              "train_groups": len({row["component"] for row in rows["train"]}),
              "validation_groups": len({row["component"] for row in rows["val"]}),
              "epochs_completed": epochs, "snapshot_sha256": snapshot["snapshot_sha256"],
              "best_checkpoint": str(best_checkpoint),
              "best_checkpoint_sha256": sha256(best_checkpoint),
              "resume_checkpoint": str(resumable),
              "status": "prototype_only" if config["prototype_only"] else "trained_unevaluated"}
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
