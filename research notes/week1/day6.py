import numpy as np
from itertools import groupby

# PWM (same TATA-like)
PWM = np.array([
    [0.1, 0.1, 0.1, 0.7],
    [0.7, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.7],
    [0.7, 0.1, 0.1, 0.1]
])
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

def pwm_score(window, pwm):
    score = 1.0
    for i, b in enumerate(window):
        score *= pwm[i, BASE_INDEX[b]]
    return score

def normalized_pwm_score(window, pwm):
    max_score = np.prod(np.max(pwm, axis=1))
    return pwm_score(window, pwm) / max_score

def best_pwm_score(seq, pwm):
    k = pwm.shape[0]
    best = 0.0
    best_idx = None
    for i in range(len(seq)-k+1):
        w = seq[i:i+k]
        s = normalized_pwm_score(w, pwm)
        if s > best:
            best = s
            best_idx = i
    return best, best_idx

# Improved penalties
def gc_score(seq, target_gc=0.5, sigma=0.1):
    """Return a score in [0,1] that decays smoothly with distance from target_gc.
       sigma controls width of tolerance (e.g., 0.1).
    """
    gc = (seq.count("G") + seq.count("C")) / max(1, len(seq))
    diff = abs(gc - target_gc)
    # Gaussian-like decay -> score near 1 when diff ~0, decays to 0 when diff >> sigma
    score = np.exp(- (diff**2) / (2 * sigma**2))
    return float(score)

def homopolymer_score(seq, max_allowed_run=4):
    runs = [len(list(g)) for _, g in groupby(seq)]
    max_run = max(runs) if runs else 0
    if max_run <= max_allowed_run:
        return 1.0
    # decay to 0 as run becomes much larger
    excess = max_run - max_allowed_run
    return float(max(0.0, 1.0 - (excess / (len(seq)/2))))  # simple normalization

def blacklist_penalty(seq, blacklist=None):
    """Return 1.0 if clean, else 0.0 (you can treat it as multiplier or subtractive)."""
    if not blacklist:
        blacklist = ["GAATTC", "GGATCC"]  # EcoRI, BamHI examples
    for motif in blacklist:
        if motif in seq:
            return 0.0
    return 1.0

def positional_bonus(seq, pwm, target_range=(5,9)):
    """Bonus if best PWM occurs inside target_range (inclusive). Returns 0..1"""
    _, idx = best_pwm_score(seq, pwm)
    if idx is None:
        return 0.0
    return 1.0 if (target_range[0] <= idx <= target_range[1]) else 0.0

# Placeholder predictor (replace later with trained model)
def toy_predictor(seq):
    # weaker proxy: count of 'A' near motif region; just example
    return min(1.0, seq.count("A")/10.0)

# Inspector and reward
def reward_inspector(seq, pwm, weights=None):
    if weights is None:
        weights = {"alpha":1.0, "beta":0.6, "gamma":0.4, "delta":0.3, "epsilon":0.2}
    pred = toy_predictor(seq)
    pwm_best, pwm_idx = best_pwm_score(seq, pwm)
    gc = gc_score(seq)
    hom = homopolymer_score(seq)
    black = blacklist_penalty(seq)
    pos = positional_bonus(seq, pwm, target_range=(5,9))

    # Combine: note gamma/delta act as multipliers on penalty (we'll convert penalties to scores)
    total = (weights["alpha"]*pred +
             weights["beta"]*pwm_best +
             weights["gamma"]*gc +
             weights["delta"]*hom +
             weights["epsilon"]*pos) * black  # zeroed if blacklist hit

    return {
        "seq": seq,
        "predictor": pred,
        "pwm_best": pwm_best,
        "pwm_idx": pwm_idx,
        "gc_score": gc,
        "homopolymer_score": hom,
        "positional_bonus": pos,
        "blacklist_ok": bool(black),
        "total": total
    }

# Diagnostic sequences
seq_A = "ACGTCTATAAAGGCTGACCT"   # TATA present in ~index 5
seq_B = "ACGTCTACAAAGGCTGACCT"   # one mismatch
seq_C = "GCGCGCGCGCGCGCGCGCGC"   # GC-rich, no TATA
seq_D = "ACGTCTATCATATATGACCT"   # TATA present beyond index 5
seq_E = "GAATTCATCATATATGACCT"   # Blacklist present

for s in [seq_A, seq_B, seq_C, seq_D, seq_E]:
    print(reward_inspector(s, PWM))
