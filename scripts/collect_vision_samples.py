"""Collect a bounded set of unreviewed enemy-preview screenshots from one live session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autococ.config import load_config
from autococ.device import DeviceManager
from autococ.errors import AutoCOCError
from autococ.session import GameSession
from autococ.vision_collection import collect_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for images, unreviewed manifest and fee ledger")
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--max-search-cost-gold", type=int, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_samples <= 10:
        parser.error("--max-samples must be between 1 and 10")
    if args.max_search_cost_gold < 0:
        parser.error("--max-search-cost-gold must be nonnegative")
    if args.output.exists():
        parser.error(f"--output must be a new directory: {args.output}")
    session = None
    try:
        config = load_config(args.config)
        manager = DeviceManager(config)
        device = manager.connect()
        run_directory = args.output.parent / f"{args.output.name}-session"
        if run_directory.exists():
            raise ValueError(f"Refusing to overwrite session evidence: {run_directory}")
        session = GameSession.connect(config, manager.adb, device.serial,
                                      run_directory, launch=True)
        result = collect_samples(session, args.output, max_samples=args.max_samples,
                                 max_search_cost_gold=args.max_search_cost_gold)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "completed" else 1
    except (AutoCOCError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False))
        return 1
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())
