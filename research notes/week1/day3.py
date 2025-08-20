import math

pwm = [
    {'A':0.1, 'C':0.1, 'G':0.1, 'T':0.7},
    {'A':0.7, 'C':0.1, 'G':0.1, 'T':0.1},
    {'A':0.1, 'C':0.1, 'G':0.1, 'T':0.7},
    {'A':0.7, 'C':0.1, 'G':0.1, 'T':0.1},
]

seq = "ACGTCTATAAAGGCTGACCTGCTGATATATACGATGCTG"

def pwm_score(window):
    score = 1.0
    for i, base in enumerate(window):
        score *= pwm[i].get(base, 0.0)
    return score

best = (None, -1.0, -1)
for i in range(len(seq)-len(pwm)+1):
    w = seq[i:i+len(pwm)]
    s = pwm_score(w)
    if s > best[1]:
        best = (w, s, i)

gc = (seq.count('G') + seq.count('C')) / len(seq)

print("Best window:", best[0], "at index", best[2], "score:", best[1])
print(f"GC% (Useful for the Promoter design): {gc*100:.1f}%")
