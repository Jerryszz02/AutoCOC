"""Serve a local human scene review UI or export explicitly confirmed scenes.

Examples:
  python scripts/review_vision_scenes.py serve --root DATASET --journal REVIEWS.jsonl
  python scripts/review_vision_scenes.py export --root DATASET --journal REVIEWS.jsonl --output NEW_DATASET
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.vision.data import DatasetError
from training.vision.human_review import SceneReview, serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "export"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True, help="Existing source dataset directory")
        command.add_argument("--journal", type=Path, required=True, help="Separate append-only human verdict log")
        command.add_argument("--manifest", default="manifest.jsonl")
        command.add_argument("--annotations", default="annotations.jsonl")
        if name == "serve":
            command.add_argument("--port", type=int, default=8765)
        else:
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        review = SceneReview(args.root, args.journal, manifest=args.manifest, annotations=args.annotations)
        if args.command == "serve":
            serve(review, args.port)
        else:
            print(json.dumps(review.export(args.output), ensure_ascii=False, indent=2))
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        print(f"scene-human-review: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
