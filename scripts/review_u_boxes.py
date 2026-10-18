"""Serve a local U-box review UI or export its explicit review journal."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.vision.data import DatasetError
from training.vision.u_review import UReviewJournal, serve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "export"):
        command = sub.add_parser(name)
        command.add_argument("--queue", type=Path, required=True)
        command.add_argument("--journal", type=Path, required=True)
        command.add_argument("--policy", choices=("prototype", "formal"), required=True)
        if name == "serve":
            command.add_argument("--port", type=int, default=8765)
        else:
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        review = UReviewJournal(args.queue, args.journal,
                                prototype_only=args.policy == "prototype")
        if args.command == "serve":
            serve(review, args.port)
            return 0
        print(json.dumps(review.export_reviews(args.output), ensure_ascii=False, indent=2))
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        print(f"u-box-review: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
