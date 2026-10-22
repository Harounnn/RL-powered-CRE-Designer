"""Validate SHARPR records and create reproducible sequence-level partitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.diffusion import DEFAULT_DIFFUSION_STEPS, DEFAULT_SEQUENCE_LENGTH

DEFAULT_INPUT = Path("data/processed/sharpr/processed_sharpr_sequence_activity.csv")
DEFAULT_OUTPUT = Path("data/processed/sharpr/diffusion_splits")
MIN_COVERAGE = 10
SPLIT_SEED = 42


def prepare_splits(
    input_csv: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = SPLIT_SEED,
) -> dict[str, int]:
    """Filter, validate, and split rows by unique sequence (80/10/10)."""
    frame = pd.read_csv(input_csv, dtype={"sequence": str})
    required = {"sequence", "rna_total", "dna_total", "log2_ratio"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {', '.join(sorted(missing))}")

    frame["coverage"] = pd.to_numeric(frame["rna_total"], errors="coerce").fillna(0) + pd.to_numeric(
        frame["dna_total"], errors="coerce"
    ).fillna(0)
    frame = frame.loc[frame["coverage"] >= MIN_COVERAGE].copy()
    frame["sequence"] = frame["sequence"].str.upper()
    frame["log2_ratio"] = pd.to_numeric(frame["log2_ratio"], errors="coerce")

    invalid_length = frame["sequence"].str.len() != DEFAULT_SEQUENCE_LENGTH
    invalid_bases = ~frame["sequence"].str.fullmatch(r"[ACGT]+", na=False)
    invalid_target = ~np.isfinite(frame["log2_ratio"])
    if invalid_length.any() or invalid_bases.any() or invalid_target.any():
        raise ValueError(
            "filtered data failed validation: "
            f"wrong_length={int(invalid_length.sum())}, "
            f"invalid_bases={int(invalid_bases.sum())}, "
            f"nonfinite_targets={int(invalid_target.sum())}"
        )

    unique_sequences = frame["sequence"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_sequences)
    n_unique = len(shuffled)
    n_train = int(n_unique * 0.8)
    n_validation = int(n_unique * 0.1)
    split_sequences = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "test": shuffled[n_train + n_validation :],
    }
    sequence_to_split = {
        sequence: split for split, sequences in split_sequences.items() for sequence in sequences
    }
    frame["split"] = frame["sequence"].map(sequence_to_split)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        partition = frame.loc[frame["split"] == split].drop(columns=["split"])
        partition.to_csv(output_dir / f"{split}.csv", index=False)
        counts[split] = int(partition["sequence"].nunique())
    manifest = pd.DataFrame(
        [(sequence, split) for split, sequences in split_sequences.items() for sequence in sequences],
        columns=["sequence", "split"],
    )
    manifest.to_csv(output_dir / "sequence_split_manifest.csv", index=False)
    config = {
        "input_csv": str(input_csv),
        "coverage_filter": "rna_total + dna_total >= 10",
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "alphabet": "ACGT",
        "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "seed": seed,
        "diffusion_steps_pilot": DEFAULT_DIFFUSION_STEPS,
        "cumulative_retention_schedule": "linear: alpha_bar(t) = 1 - t/T",
        "unique_sequences": n_unique,
        "rows_after_coverage_filter": len(frame),
        "unique_sequences_per_split": counts,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    counts = prepare_splits(args.input, args.output_dir, args.seed)
    print(f"Saved sequence-level partitions to {args.output_dir}")
    print(f"Unique sequences per split: {counts}")


if __name__ == "__main__":
    main()
