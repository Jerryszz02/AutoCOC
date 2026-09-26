"""Offline digit-template experiment; never used to authorize game input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np


def glyphs(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 115]), np.array([180, 65, 255]))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = sorted((x, y, w, h) for x, y, w, h, area in stats[1:] if area >= 8 and h >= 8)
    result = []
    for x, y, w, h in boxes:
        if x == 0 or y == 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]:
            return []
        result.append(cv2.resize(mask[y:y+h, x:x+w], (20, 28), interpolation=cv2.INTER_AREA))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/digit-template-experiment.json"))
    args = parser.parse_args()
    events = [json.loads(line) for line in (args.run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    snapshots = {Path(row["frame"]).name: row for row in events if row.get("kind") == "observation"}
    deployments = [row for row in events if row.get("kind") == "line_deployment_consumed"]
    first = Path(deployments[0]["before_frame"]).name
    reference = max(name for name in snapshots if name < first and
                    (snapshots[name]["observations"].get("battle") or {}).get("slots"))
    slots = snapshots[reference]["observations"]["battle"]["slots"]
    bank = {}
    training = []

    def crop(name, slot):
        image = cv2.imdecode(np.fromfile(args.run / "frames" / name, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
        left, top, right, bottom = slot["bbox"]
        return image[top-5:top+35, left+5:right-2]

    def learn(image, label, source):
        pieces = glyphs(image)
        training.append({"source": source, "label": label, "segments": len(pieces), "accepted": len(pieces) == len(label)})
        if len(pieces) == len(label):
            for char, piece in zip(label, pieces, strict=True):
                bank.setdefault(char.lower(), []).append(piece)

    for slot in slots:
        if slot["kind"] == "troop" and slot.get("count"):
            learn(crop(reference, slot), f"x{slot['count']}", reference)
    first_zero = deployments[0]
    zero_name = Path(first_zero["after_frame"]).name
    zero_slot = next(slot for slot in slots if slot["point"] == first_zero["slot"])
    learn(crop(zero_name, zero_slot), "x0", zero_name)
    rows = []
    for event in deployments:
        slot = next(slot for slot in slots if slot["point"] == event["slot"])
        for field, count in (("before_frame", "before_count"), ("after_frame", "after_count")):
            name = Path(event[field]).name
            if name in {reference, zero_name}:
                continue
            image = crop(name, slot)
            samples = []
            for _ in range(100):
                started = perf_counter()
                text, scores, accepted = "", [], True
                for piece in glyphs(image):
                    ranked = sorted(((max(float(cv2.matchTemplate(piece, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                                          for template in templates), char) for char, templates in bank.items()), reverse=True)
                    if len(ranked) < 2 or ranked[0][0] < .9 or ranked[0][0] - ranked[1][0] < .05:
                        accepted = False
                        break
                    text += ranked[0][1]
                    scores.append(ranked[0][0])
                accepted = accepted and text.startswith("x") and text[1:].isdigit()
                prediction = int(text[1:]) if accepted else None
                samples.append((perf_counter() - started) * 1000)
            rows.append({"frame": name, "expected": event[count], "prediction": prediction,
                         "accepted": accepted, "median_ms": median(samples), "scores": scores})
    missing = sorted(set("0123456789") - bank.keys())
    result = {"production_enabled": False, "training": training, "trained_characters": sorted(bank),
              "missing_digits": missing, "held_out_frames": rows,
              "accepted_correct": sum(row["accepted"] and row["prediction"] == row["expected"] for row in rows),
              "accepted_wrong": sum(row["accepted"] and row["prediction"] != row["expected"] for row in rows),
              "rejected": sum(not row["accepted"] for row in rows),
              "decision": "Keep local OCR: digit/state coverage is insufficient for a production reader."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
