import re
import numpy as np
from src.predictor import CREPredictor


# -------------------------
# Hyperparameters (safe defaults)
# -------------------------
ALPHA = 1.0     # predictor weight
BETA = 0.15     # motif bonus weight
GAMMA = 0.20    # GC penalty weight
DELTA = 0.20    # homopolymer penalty weight
EPSILON = 0.50  # blacklist penalty weight

TARGET_GC = 0.45
GC_TOL = 0.15


predictor = CREPredictor()


# -------------------------
# Motif bonus (simple motif scoring)
# -------------------------
MOTIFS = [
    "TATAAA",   # TATA box
    "CCAAT",    # CCAAT box
    "GGGCGG",   # GC box (SP1-like)
]


def motif_bonus(seq):
    score = 0.0
    for motif in MOTIFS:
        if motif in seq:
            score += 1.0
    return score / len(MOTIFS)  # normalize to [0,1]


# -------------------------
# GC penalty
# -------------------------
def gc_penalty(seq):
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    deviation = abs(gc - TARGET_GC)
    return min(deviation / GC_TOL, 1.0)


# -------------------------
# Homopolymer penalty
# -------------------------
def homopolymer_penalty(seq, max_run=6):
    penalty = 0.0
    for base in "ACGT":
        runs = re.findall(f"{base}{{{max_run},}}", seq)
        penalty += len(runs)
    return min(penalty, 1.0)


# -------------------------
# Blacklist penalty
# -------------------------
BLACKLIST = [
    "AAAAAAA",
    "CCCCCCC",
    "GGGGGGG",
    "TTTTTTT",
]


def blacklist_penalty(seq):
    for bad in BLACKLIST:
        if bad in seq:
            return 1.0
    return 0.0


# -------------------------
# Final reward
# -------------------------
def reward(seq):
    seq = seq.upper().replace("N", "A")

    p = predictor.percentile_score(seq)
    m = motif_bonus(seq)
    g = gc_penalty(seq)
    h = homopolymer_penalty(seq)
    b = blacklist_penalty(seq)

    total = (
        ALPHA * p
        + BETA * m
        - GAMMA * g
        - DELTA * h
        - EPSILON * b
    )

    return float(total)
