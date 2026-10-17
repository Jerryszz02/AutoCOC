"""Collect unlabelled scene candidates on one bounded, no-cost navigation route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh directory outside the repository; never reused")
    parser.add_argument("--split", choices=("TRAIN", "VAL"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--group-id", required=True,
                        help="Predeclared candidate leakage group; independence remains unverified")
    parser.add_argument("--limit-sec", type=int, default=180)
    args = parser.parse_args(argv)

    from autococ.scene_collection import collect_scene_samples

    result = collect_scene_samples(args.config, args.output, split=args.split,
                                   session_id=args.session_id, group_id=args.group_id,
                                   limit_sec=args.limit_sec, repo_root=REPO_ROOT)
    print(json.dumps({"status": result["status"], "reason": result["reason"],
                      "summary": str(args.output.resolve() / "summary.json")},
                     ensure_ascii=False))
    return 0 if result["status"] == "candidate_captured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
