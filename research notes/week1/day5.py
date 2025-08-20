import numpy as np

# Define a PWM (TATA-like motif)
PWM = np.array([
    [0.1, 0.1, 0.1, 0.7],  # Pos 1
    [0.7, 0.1, 0.1, 0.1],  # Pos 2
    [0.1, 0.1, 0.1, 0.7],  # Pos 3
    [0.7, 0.1, 0.1, 0.1]   # Pos 4
])
BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

def pwm_score(seq, pwm):
    """Compute multiplicative PWM score for a 4-mer."""
    score = 1.0
    for i, base in enumerate(seq):
        score *= pwm[i, BASE_INDEX[base]]
    return score

def normalized_pwm_score(seq, pwm):
    """Normalize by the theoretical max score of the PWM."""
    max_score = np.prod(np.max(pwm, axis=1))
    return pwm_score(seq, pwm) / max_score

def best_pwm_score(sequence, pwm):
    """Slide PWM across a longer sequence and return the best normalized score."""
    k = pwm.shape[0]
    best = 0
    for i in range(len(sequence) - k + 1):
        window = sequence[i:i+k]
        score = normalized_pwm_score(window, pwm)
        if score > best:
            best = score
    return best

# Dummy penalties (simple examples)
def gc_penalty(seq):
    gc_content = (seq.count("G") + seq.count("C")) / len(seq)
    return abs(gc_content - 0.5)  # penalize deviation from 50%

def homopolymer_penalty(seq):
    return max(len(run) for run in split_runs(seq)) - 3  # penalize long runs

def split_runs(seq):
    run, runs = [seq[0]], []
    for c in seq[1:]:
        if c == run[-1]:
            run.append(c)
        else:
            runs.append("".join(run))
            run = [c]
    runs.append("".join(run))
    return runs

# Reward function
def reward(seq, alpha=1.0, beta=0.5, gamma=0.1, delta=0.1):
    # Placeholder predictor (pretend ML model gives strength 0.7 for any seq)
    predictor_score = 0.7  
    
    return (
        alpha * predictor_score
        + beta * best_pwm_score(seq, PWM)
        - gamma * gc_penalty(seq)
        - delta * homopolymer_penalty(seq)
    )

# ---- Test ----
sequence1 = "ACGTCTATAAAGGCTGACCT"
sequence2 = "ACGTCTACAAAGGCTGACCT"

print("Best PWM score seq1:", best_pwm_score(sequence1, PWM))
print("Reward seq1:", reward(sequence1))

print("Best PWM score seq2:", best_pwm_score(sequence2, PWM))
print("Reward seq2:", reward(sequence2))
