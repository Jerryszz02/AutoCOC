"""Rebuild anonymized 18.647.2 battle chest fixtures from immutable live captures.

This is an audit helper, not a runtime data source.  Source screenshots stay in
the canonical report directory; tests use only the produced fixtures/provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2


REPORTS = Path(r"E:\projects\AutoCOC\reports\completion-20261015")
OUT = Path(__file__).resolve().parent
TEMPLATES = OUT.parents[2] / "assets" / "templates"
SOURCES = {
    "settlement_claim": "live/20261016-092326-164096-a62d874d/frames/00070-battle-settlement.png",
    "chest_ordinary_4": "probes/20261016-093242-575639-claim-battle-reward-v1/frames/00003-after-claim-reread.png",
    "chest_rare_3": "probes/20261016-093517-654374-chest-step/frames/00003-after-inspected-step-reread.png",
    "chest_rare_2": "probes/20261016-093617-121390-chest-step/frames/00003-after-inspected-step-reread.png",
    "chest_rare_1": "probes/20261016-093740-121552-chest-step/frames/00003-after-inspected-step-reread.png",
    "hero_potion_temporary_storage": "probes/20261016-093903-836291-chest-step/frames/00003-after-inspected-step-reread.png",
    "home_after": "probes/20261016-094007-973593-chest-step/frames/00003-after-inspected-step-reread.png",
}
SUMMARY_DIR = {
    "chest_ordinary_4": "20261016-093242-575639-claim-battle-reward-v1",
    "chest_rare_3": "20261016-093517-654374-chest-step",
    "chest_rare_2": "20261016-093617-121390-chest-step",
    "chest_rare_1": "20261016-093740-121552-chest-step",
    "hero_potion_temporary_storage": "20261016-093903-836291-chest-step",
    "home_after": "20261016-094007-973593-chest-step",
}
TEMPLATE_CROPS = {
    "battle_reward_claim_button_18_647_2": ("settlement_claim", (530, 585, 755, 655)),
    "battle_chest_background_18_647_2": ("chest_ordinary_4", (30, 220, 180, 360)),
    "battle_chest_ordinary_18_647_2": ("chest_ordinary_4", (520, 260, 760, 490)),
    "battle_chest_rare_18_647_2": ("chest_rare_3", (520, 260, 760, 490)),
    "battle_chest_hero_potion_18_647_2": ("hero_potion_temporary_storage", (515, 260, 745, 485)),
    "battle_chest_continue_18_647_2": ("hero_potion_temporary_storage", (555, 620, 745, 685)),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    run = json.loads((REPORTS / "live/run-20261016-092326-164096-a62d874d.json").read_text(encoding="utf-8"))
    settlement_event = next(json.loads(line) for line in Path(run["events_path"]).read_text(encoding="utf-8").splitlines()
                            if '"kind": "observation"' in line and "00070-battle-settlement.png" in line)
    allowed_settlement = {"47%", "胜利", "您得到了", "77944", "1000109", "1770", "75%奖励：",
                          "+281250", "+2081", "损耗的部队", "领取奖励"}
    entries = {}
    frames = {}
    for name, source_relative in SOURCES.items():
        source = REPORTS / source_relative
        image = cv2.imread(str(source))
        if image is None or image.shape[:2] != (1440, 2560):
            raise RuntimeError(f"Expected the observed 2560x1440 source: {source}")
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
        redactions = []
        if name == "settlement_claim":
            # Opponent name/clan and exposed opponent village portrait.
            image[0:75, 0:290] = (0, 0, 0)
            redactions.append([0, 0, 290, 75])
            # A second clan name remains in the battlefield just above the
            # result's troop-loss heading; keep that heading's glyphs intact.
            image[450:468, 600:680] = (0, 0, 0)
            redactions.append([600, 450, 680, 468])
            ocr = [item for item in settlement_event["observations"]["ocr"]
                   if item["text"] in allowed_settlement]
        else:
            summary = json.loads((REPORTS / "probes" / SUMMARY_DIR[name] / "summary.json").read_text(encoding="utf-8"))
            ocr = summary["final_ocr"]
            if name == "home_after":
                image[0:120, 95:270] = (0, 0, 0)
                redactions.append([95, 0, 270, 120])
                image[258:284, 535:600] = (0, 0, 0)
                redactions.append([535, 258, 600, 284])
                ocr = [item for item in ocr if item["text"] not in {"JeRRYszz", "荣耀"}]
        output = OUT / f"{name}.png"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"Could not save fixture {output}")
        frames[name] = image
        entries[name] = {
            "source_relative_to_reports": source_relative,
            "source_sha256": sha(source),
            "fixture_sha256": sha(output),
            "transformation": "resize_INTER_AREA_1280x720_and_blackout_boxes",
            "blackout_boxes_1280x720": redactions,
            "ocr_from_saved_event_or_summary": ocr,
            "human_review": False,
            "formal_acceptance": False,
        }
    templates = {}
    for name, (frame, box) in TEMPLATE_CROPS.items():
        x1, y1, x2, y2 = box
        output = TEMPLATES / f"{name}.png"
        if not cv2.imwrite(str(output), frames[frame][y1:y2, x1:x2]):
            raise RuntimeError(f"Could not save template {output}")
        templates[name] = {"fixture": frame, "box_1280x720": box, "sha256": sha(output)}
    (OUT / "provenance.json").write_text(json.dumps({"client_version": "18.647.2", "fixtures": entries,
                                                      "templates": templates},
                                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
