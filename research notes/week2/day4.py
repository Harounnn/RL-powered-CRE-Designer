import numpy as np
from collections import Counter
from itertools import product

# ---------- one-hot ----------
BASES = ["A","C","G","T"]
BASE_INDEX = {b:i for i,b in enumerate(BASES)}

def onehot(seq):
    """Return (L,4) numpy array of ints (one-hot)."""
    arr = np.zeros((len(seq), 4), dtype=int)
    for i, b in enumerate(seq):
        if b in BASE_INDEX:
            arr[i, BASE_INDEX[b]] = 1
    return arr

# ---------- k-mer utilities ----------
def kmerize(seq, k=3, overlap=True):
    """Return list of k-mers. If overlap True, sliding window; else non-overlapping."""
    if overlap:
        return [seq[i:i+k] for i in range(len(seq)-k+1)]
    else:
        return [seq[i:i+k] for i in range(0, len(seq)-k+1, k)]

def kmer_counts(seq, k=3, overlap=True):
    """Return Counter of k-mers (only canonical A/C/G/T kmers)."""
    kmers = kmerize(seq, k=k, overlap=overlap)
    # filter out kmers with ambiguous bases
    kmers = [kmer for kmer in kmers if len(kmer)==k and all(c in BASE_INDEX for c in kmer)]
    return Counter(kmers)

def build_feature_matrix(seqs, k=3, overlap=True, vocab=None):
    """
    Build (N, V) matrix where V is sorted k-mer vocabulary.
    If vocab is None, build from data.
    Returns: features (numpy), vocab_list
    """
    counts = [kmer_counts(s, k=k, overlap=overlap) for s in seqs]
    if vocab is None:
        # collect all k-mers in the data and sort for stable ordering
        vocab = sorted({km for c in counts for km in c.keys()})
    V = len(vocab)
    mat = np.zeros((len(seqs), V), dtype=int)
    idx = {v:i for i,v in enumerate(vocab)}
    for i,c in enumerate(counts):
        for kmer, ct in c.items():
            mat[i, idx[kmer]] = ct
    return mat, vocab

# ----------  experiment ----------
seq_A = "ACGTCTATAAAGGCTGACCT"   # has TATA at ~index 5
seq_B = "ACGTCTACAAAGGCTGACCT"   # one mismatch in motif
seq_C = "GCGCGCGCGCGCGCGCGCGC"   # GC-rich, no TATA

seqs = [seq_A, seq_B, seq_C]

print("=== One-hot shapes ===")
for s in seqs:
    print(s, "len:", len(s), "onehot shape:", onehot(s).shape)

print("\n=== Example k-mer lists (k=4, overlapping) ===")
for s in seqs:
    print(s, "->", kmerize(s, k=4, overlap=True)[:8], " ...")  # show first 8

print("\n=== k-mer counts (k=4) ===")
for s in seqs:
    print(s, kmer_counts(s, k=4, overlap=True).most_common(5))

print("\n=== Build feature matrix (k=4) ===")
mat, vocab = build_feature_matrix(seqs, k=4, overlap=True)
print("Vocab size:", len(vocab))
print("Vocab (first 20):", vocab[:20])
print("Feature matrix shape:", mat.shape)
print("Feature matrix (rows=sequences):\n", mat)

# show which k-mers capture 'TATA' 
print("\n=== TATA-containing k-mers in vocab ===")
tata_kmers = [v for v in vocab if "TATA" in v]
print("TATA-related k-mers:", tata_kmers)
