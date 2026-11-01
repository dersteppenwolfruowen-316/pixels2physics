"""Command-line entry point for the local data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_pipeline import build_dataset, build_sft_subset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build split MPMWorlds training manifests without executing scene programs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument("--subset-from", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/m2"))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.data_root is not None and not args.data_root.is_dir():
        parser.error(f"--data-root is not a directory: {args.data_root}")
    if args.subset_from is not None and not (args.subset_from / "canonical_records.jsonl").is_file():
        parser.error(f"--subset-from does not contain canonical_records.jsonl: {args.subset_from}")
    if args.subset_from is not None and args.sample_size is None:
        parser.error("--sample-size is required with --subset-from")
    if args.data_root is not None and args.sample_size is not None:
        parser.error("--sample-size can only be used with --subset-from")
    try:
        report = (
            build_sft_subset(args.subset_from, args.output_root, args.sample_size, args.seed, args.overwrite)
            if args.subset_from is not None
            else build_dataset(args.data_root, args.output_root, args.seed, args.workers, args.overwrite)
        )
    except (FileExistsError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
