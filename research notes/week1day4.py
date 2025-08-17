from math import log
from itertools import groupby

# Example PWMs (probabilities). Real PWMs often come from JASPAR.
pwm_TATA = [
    {'A':0.1, 'C':0.1, 'G':0.1, 'T':0.7},
    {'A':0.7, 'C':0.1, 'G':0.1, 'T':0.1},
    {'A':0.1, 'C':0.1, 'G':0.1, 'T':0.7},
    {'A':0.7, 'C':0.1, 'G':0.1, 'T':0.1},
]
pwm_SPOT = [   
    {'A':0.25,'C':0.25,'G':0.25,'T':0.25},
    {'A':0.1, 'C':0.1, 'G':0.7, 'T':0.1},
    {'A':0.1, 'C':0.7, 'G':0.1, 'T':0.1},
]

PWMS = {'TATA': pwm_TATA, 'SPOT': pwm_SPOT}

def pwm_score_window(window, pwm):
    score = 1.0
    for i, base in enumerate(window):
        score *= pwm[i].get(base, 0.0)
    return score

def best_pwm_score(seq, pwms):
    best = (None, 0.0, None)  # (motif_name, score, index)
    for name, pwm in pwms.items():
        k = len(pwm)
        for i in range(len(seq)-k+1):
            w = seq[i:i+k]
            s = pwm_score_window(w, pwm)
            if s > best[1]:
                best = (name, s, i)
    return best 

def gc_penalty(seq, target_gc=0.5):
    gc = (seq.count('G') + seq.count('C')) / max(1, len(seq))
    # normalized absolute deviation
    return abs(gc - target_gc)  

def homopolymer_penalty(seq, max_allowed_run=4):
    runs = [len(list(g)) for _, g in groupby(seq)]
    max_run = max(runs) if runs else 0
    # penalty is proportional to how much max_run exceeds allowed
    return max(0, (max_run - max_allowed_run) / max_allowed_run)  

# Toy predictor: placeholder function mapping sequence to [0,1]
def toy_predictor(seq):
    # simple proxy: presence of 'TATA' increases score, else low
    return 0.9 if 'TATA' in seq else 0.2

def reward(seq, predictor_func, pwms, weights=(1.0, 0.5, 0.3, 0.2)):
    alpha, beta, gamma, delta = weights
    pred = predictor_func(seq)          
    motif_name, motif_score, idx = best_pwm_score(seq, pwms)
    # motif_score is multiplicative; optionally normalize by max theoretical score (skip for toy)
    r_motif = motif_score
    r_gc = 1 - gc_penalty(seq)        
    r_hom = 1 - homopolymer_penalty(seq)  
    total = alpha*pred + beta*r_motif + gamma*r_gc + delta*r_hom
    return {
        'total': total,
        'pred': pred,
        'motif': (motif_name, motif_score, idx),
        'gc': ( (seq.count('G')+seq.count('C'))/len(seq), r_gc),
        'homopolymer': (max([len(list(g)) for _,g in __import__('itertools').groupby(seq)]), r_hom)
    }

# Usage
seq = "ACGTCTAGAAAGGCTGACCTGCTTATACGTACGATGCTG"
print(best_pwm_score(seq, PWMS))
print(reward(seq, toy_predictor, PWMS))
