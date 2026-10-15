"""Report frozen S/B class, source-session, version and split coverage without mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.vision.coverage import coverage_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-snapshot", type=Path, required=True)
    parser.add_argument("--building-snapshot", type=Path, required=True)
    parser.add_argument("--client-version", required=True)
    parser.add_argument("--acceptance", type=Path, default=Path(__file__).resolve().parents[1]
                        / "training/vision/acceptance-v1.json")
    parser.add_argument("--output", type=Path, help="New JSON report; existing files are never replaced")
    args = parser.parse_args(argv)
    try:
        report = coverage_report(args.scene_snapshot, args.building_snapshot,
                                 acceptance_path=args.acceptance,
                                 current_client_version=args.client_version)
        result = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(result)
        print(result, end="")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"vision-coverage: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
