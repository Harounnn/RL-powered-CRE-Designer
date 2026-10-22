"""Small, configurable nucleotide corruption utilities for discrete diffusion.

The pilot schedule linearly decreases cumulative retention from 1 at t=0 to
0 at t=T. Thus q(x_t | x_0) retains each original base with probability
1 - t/T and otherwise draws uniformly from A/C/G/T.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

BASES = "ACGT"
BASE_TO_TOKEN = {base: token for token, base in enumerate(BASES)}
DEFAULT_SEQUENCE_LENGTH = 145
DEFAULT_DIFFUSION_STEPS = 32


def cumulative_retention(t: int, steps: int = DEFAULT_DIFFUSION_STEPS) -> float:
    """Return the linear cumulative retention probability at timestep ``t``."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(t, int) or not 0 <= t <= steps:
        raise ValueError("t must be an integer in [0, steps]")
    return 1.0 - (t / steps)


def encode_sequences(sequences: Sequence[str], sequence_length: int = DEFAULT_SEQUENCE_LENGTH) -> np.ndarray:
    """Encode A/C/G/T sequences as an int64 array with shape (batch, length)."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    encoded = np.empty((len(sequences), sequence_length), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        sequence = sequence.upper()
        if len(sequence) != sequence_length:
            raise ValueError(f"sequence {row} has length {len(sequence)}; expected {sequence_length}")
        invalid = set(sequence) - set(BASES)
        if invalid:
            raise ValueError(f"sequence {row} contains invalid bases: {''.join(sorted(invalid))}")
        encoded[row] = [BASE_TO_TOKEN[base] for base in sequence]
    return encoded


def decode_sequences(tokens: np.ndarray) -> list[str]:
    """Decode a one- or two-dimensional array of nucleotide token IDs."""
    tokens = np.asarray(tokens)
    if tokens.ndim not in (1, 2):
        raise ValueError("tokens must have shape (length,) or (batch, length)")
    if not np.issubdtype(tokens.dtype, np.integer) or np.any((tokens < 0) | (tokens >= len(BASES))):
        raise ValueError("tokens must contain integer IDs in [0, 3]")
    rows = tokens[None, :] if tokens.ndim == 1 else tokens
    return ["".join(BASES[int(token)] for token in row) for row in rows]


def _validate_token_batch(tokens: np.ndarray) -> tuple[np.ndarray, bool]:
    tokens = np.asarray(tokens)
    was_vector = tokens.ndim == 1
    if was_vector:
        tokens = tokens[None, :]
    if tokens.ndim != 2 or tokens.shape[1] == 0:
        raise ValueError("tokens must have shape (length,) or (batch, length), with positive length")
    if not np.issubdtype(tokens.dtype, np.integer) or np.any((tokens < 0) | (tokens >= len(BASES))):
        raise ValueError("tokens must contain integer IDs in [0, 3]")
    return tokens.astype(np.int64, copy=False), was_vector


def corrupt_tokens(
    clean_tokens: np.ndarray,
    t: int,
    steps: int = DEFAULT_DIFFUSION_STEPS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample q(x_t | x_0) by retaining or uniformly resampling each base."""
    clean, was_vector = _validate_token_batch(clean_tokens)
    keep_probability = cumulative_retention(t, steps)
    rng = np.random.default_rng() if rng is None else rng
    keep = rng.random(clean.shape) < keep_probability
    uniform = rng.integers(0, len(BASES), size=clean.shape, dtype=np.int64)
    noisy = np.where(keep, clean, uniform)
    return noisy[0] if was_vector else noisy


def sample_terminal_noise(
    n_sequences: int,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample independent uniform nucleotide tokens with shape (n_sequences, length)."""
    if not isinstance(n_sequences, int) or n_sequences < 0:
        raise ValueError("n_sequences must be a nonnegative integer")
    if not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    return rng.integers(0, len(BASES), size=(n_sequences, sequence_length), dtype=np.int64)
