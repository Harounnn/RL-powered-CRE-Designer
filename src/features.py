"""Feature utilities: one-hot, k-mer extraction and feature matrix builder."""
import numpy as np
from collections import Counter

BASES = ["A","C","G","T"]
BASE_INDEX = {b:i for i,b in enumerate(BASES)}

def onehot(seq):
    arr = np.zeros((len(seq), 4), dtype=int)
    for i, b in enumerate(seq):
        if b in BASE_INDEX:
            arr[i, BASE_INDEX[b]] = 1
    return arr

def kmerize(seq, k=6, overlap=True):
    if overlap:
        return [seq[i:i+k] for i in range(len(seq)-k+1)]
    else:
        return [seq[i:i+k] for i in range(0, len(seq)-k+1, k)]

def kmer_counts(seq, k=6):
    kmers = kmerize(seq, k=k, overlap=True)
    return Counter([kmer for kmer in kmers if len(kmer)==k and all(c in BASE_INDEX for c in kmer)])

def build_feature_matrix(seqs, k=6, vocab=None):
    counts = [kmer_counts(s, k=k) for s in seqs]
    if vocab is None:
        vocab = sorted({km for c in counts for km in c.keys()})
    V = len(vocab)
    X = np.zeros((len(seqs), V), dtype=int)
    idx = {v:i for i,v in enumerate(vocab)}
    for i,c in enumerate(counts):
        for kmer, ct in c.items():
            X[i, idx[kmer]] = ct
    return X, vocab

if __name__ == "__main__":
    # simple demo
    seqs = ["ACGTCTATAAAGGCTGACCT", "GCGCGCGCGCGC"]
    X, vocab = build_feature_matrix(seqs, k=4)
    print("Vocab size:", len(vocab))
    print(X)