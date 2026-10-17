"""Report class/group coverage for unassigned candidates without freezing or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.vision.candidate_coverage import candidate_coverage_report
from training.vision.data import DatasetError, load_taxonomy, read_jsonl, sha256


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True, help="Read-only root for candidate images")
    cli.add_argument("--manifest", type=Path, required=True, help="Manifest whose rows must be split=unassigned")
    cli.add_argument("--annotations", type=Path, required=True, help="Candidate annotations JSONL")
    cli.add_argument("--task", choices=("scene", "building"), required=True)
    cli.add_argument("--classes", nargs="+",
                     help="Building classes whose complete scope is being previewed")
    cli.add_argument("--taxonomy", type=Path,
                     default=Path(__file__).resolve().parents[1] / "training/vision/taxonomy.json")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--near-duplicate-review", type=Path,
                     help="Optional source-bound visual reviews for detected dHash pairs")
    cli.add_argument("--output", type=Path,
                     help="Create a new JSON report; existing files and candidate roots are never overwritten")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        root = args.root.resolve()
        manifest_path = args.manifest.resolve()
        annotations_path = args.annotations.resolve()
        taxonomy_path = args.taxonomy.resolve()
        review_path = args.near_duplicate_review.resolve() if args.near_duplicate_review else None
        output_path = args.output.resolve() if args.output else None
        input_paths = {manifest_path, annotations_path, taxonomy_path}
        if review_path is not None:
            input_paths.add(review_path)
        if output_path is not None:
            if output_path in input_paths:
                raise DatasetError("Output path must be separate from every input file")
            if output_path == root or output_path.is_relative_to(root):
                raise DatasetError("Output report must be outside the candidate image root")
            if output_path.exists():
                raise DatasetError(f"Refusing to overwrite report: {output_path}")

        input_hashes_before = {
            "manifest": sha256(manifest_path),
            "annotations": sha256(annotations_path),
            "taxonomy": sha256(taxonomy_path),
            "near_duplicate_review": sha256(review_path) if review_path is not None else None,
        }
        manifest = read_jsonl(manifest_path)
        annotations = read_jsonl(annotations_path)
        try:
            taxonomy = load_taxonomy(taxonomy_path)
        except (AttributeError, TypeError) as exc:
            raise DatasetError("Taxonomy must be an object with valid class entries") from exc
        reviews = None
        if review_path is not None:
            reviews = json.loads(review_path.read_text(encoding="utf-8"))
        report = candidate_coverage_report(
            root, manifest, annotations, task=args.task, classes=args.classes,
            taxonomy=taxonomy, seed=args.seed, near_duplicate_reviews=reviews,
        )
        input_hashes_after = {
            "manifest": sha256(manifest_path),
            "annotations": sha256(annotations_path),
            "taxonomy": sha256(taxonomy_path),
            "near_duplicate_review": sha256(review_path) if review_path is not None else None,
        }
        if input_hashes_after != input_hashes_before:
            raise DatasetError("Candidate input file changed while report was being generated")
        report["source_files"] = {
            "root": str(root),
            "manifest": {"path": str(manifest_path), "sha256": input_hashes_before["manifest"]},
            "annotations": {"path": str(annotations_path), "sha256": input_hashes_before["annotations"]},
            "taxonomy": {"path": str(taxonomy_path), "sha256": input_hashes_before["taxonomy"]},
            "near_duplicate_review": (
                {"path": str(review_path),
                 "sha256": input_hashes_before["near_duplicate_review"]}
                if review_path is not None else None
            ),
        }
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
        print(serialized, end="")
        return 0
    except (DatasetError, OSError, ValueError, KeyError) as exc:
        print(f"vision-candidate-coverage: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
