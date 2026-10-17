"""Freeze explicitly reviewed R windows against a hashed live candidate queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from training.vision.data import DatasetError
from training.vision.dynamic_import import import_reviewed_r_windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("r_building", "r_card"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-only", action="store_true",
                        help="Allow agent_visual reviewed prototype; default requires human review")
    args = parser.parse_args(argv)
    try:
        result = import_reviewed_r_windows(args.queue, args.reviews, args.source_root,
                                           args.output, kind=args.kind,
                                           prototype_only=args.prototype_only)
    except (DatasetError, OSError, ValueError) as exc:
        print(f"import-reviewed-r: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()),
                      "kind": args.kind, "prototype_only": args.prototype_only,
                      "windows": result["inventory"]["windows"],
                      "groups": result["inventory"]["groups"],
                      "snapshot_sha256": result["snapshot_sha256"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
