"""Rebuild anonymous card-only pixels from the saved 18.647.2 battle frames."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


REPORTS = Path(r"E:\projects\AutoCOC\reports\completion-20261015")
SOURCE_DIR = REPORTS / "live/20261016-101005-741427-f4b46a81/frames"
OUT = Path(__file__).resolve().parent
BOX = (638, 515, 742, 720)
NAMES = {
    "green_24": "00024-frozen-line-post.png",
    "yellow_25": "00025-frozen-line-post-recheck.png",
    "yellow_26": "00026-frozen-line-post-recheck.png",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    records = {}
    x1, y1, x2, y2 = BOX
    for name, filename in NAMES.items():
        source = SOURCE_DIR / filename
        raw = source.read_bytes()
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (1440, 2560):
            raise RuntimeError(f"Expected an observed 2560x1440 frame: {source}")
        baseline = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        crop = baseline[y1:y2, x1:x2]
        anonymous = np.zeros_like(baseline)
        anonymous[y1:y2, x1:x2] = crop
        output = OUT / f"{name}.png"
        encoded = cv2.imencode(".png", anonymous)[1].tobytes()
        output.write_bytes(encoded)
        records[name] = {
            "source_relative_to_reports": str(source.relative_to(REPORTS)).replace("\\", "/"),
            "source_sha256": sha(raw),
            "card_bbox_1280": list(BOX),
            "card_pixels_sha256": sha(crop.tobytes()),
            "fixture_sha256": sha(encoded),
            "transformation": "resize_INTER_AREA_1280x720_then_zero_outside_card_box",
        }
    (OUT / "provenance.json").write_text(json.dumps({
        "client_version": "18.647.2",
        "human_ground_truth": False,
        "dynamic_event_validation": False,
        "fixtures": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
