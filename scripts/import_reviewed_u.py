"""Import explicit U box reviews through the existing dynamic snapshot freezer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.vision.data import DatasetError
from training.vision.u_import import import_reviewed_u


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="New output directory; no overwrite")
    parser.add_argument("--policy", choices=("prototype", "formal"), required=True)
    args = parser.parse_args(argv)
    try:
        result = import_reviewed_u(args.queue, args.journal, args.reviews,
                                   args.snapshot,
                                   prototype_only=args.policy == "prototype")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        print(f"u-import: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
