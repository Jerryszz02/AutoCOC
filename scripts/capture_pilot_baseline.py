"""Capture a bounded current HOME and army snapshot for tactical pilot batch 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autococ.pilot_preflight import capture_pilot_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory outside this checkout")
    parser.add_argument("--limit-sec", type=int, default=150)
    args = parser.parse_args(argv)
    try:
        result = capture_pilot_baseline(args.config, args.output, limit_sec=args.limit_sec)
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "summary": str(args.output.resolve() / "summary.json")},
                         ensure_ascii=False))
        return 130
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": result["status"], "reason": result["reason"],
                      "summary": str(args.output.resolve() / "summary.json")}, ensure_ascii=False))
    return 0 if result["status"] == "candidate_captured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
