"""Generate synthetic promoter/non-promoter dataset for prototyping."""
import random
from src.features import kmerize

BASES = ["A","C","G","T"]

def random_seq(length, gc_bias=None):
    if gc_bias is None:
        return ''.join(random.choice(BASES) for _ in range(length))
    seq = []
    for _ in range(length):
        if random.random() < gc_bias:
            seq.append(random.choice(['G','C']))
        else:
            seq.append(random.choice(['A','T']))
    return ''.join(seq)


def insert_motif(background, motif, pos):
    return background[:pos] + motif + background[pos+len(motif):]

if __name__ == '__main__':
    N = 500
    L = 30
    motif = 'TATA'
    pos = 5
    pos_seqs = []
    neg_seqs = []
    for _ in range(N//2):
        bg = random_seq(L)
        pos_seqs.append(insert_motif(bg, motif, pos))
        neg = random_seq(L)
        while motif in neg:
            neg = random_seq(L)
        neg_seqs.append(neg)
    # save to files or return
    import json
    with open('data/synthetic_pos.json','w') as f:
        json.dump(pos_seqs, f)
    with open('data/synthetic_neg.json','w') as f:
        json.dump(neg_seqs, f)
    print('Synthetic data saved to data/')