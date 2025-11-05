import random
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------- utilities (k-mer functions) ----------
BASES = ["A","C","G","T"]

def random_seq(length, gc_bias=None):
    if gc_bias is None:
        return ''.join(random.choice(BASES) for _ in range(length))
    # gc_bias in [0,1] target GC fraction
    seq = []
    for _ in range(length):
        if random.random() < gc_bias:
            seq.append(random.choice(['G','C']))
        else:
            seq.append(random.choice(['A','T']))
    return ''.join(seq)

def insert_motif(background, motif, pos):
    return background[:pos] + motif + background[pos+len(motif):]

def kmerize(seq, k=4, overlap=True):
    if overlap:
        return [seq[i:i+k] for i in range(len(seq)-k+1)]
    else:
        return [seq[i:i+k] for i in range(0, len(seq)-k+1, k)]

def kmer_counts(seq, k=4):
    kmers = kmerize(seq, k=k, overlap=True)
    return Counter([k for k in kmers if all(c in BASES for c in k)])

def build_feature_matrix(seqs, k=4, vocab=None):
    counts = [kmer_counts(s, k=k) for s in seqs]
    if vocab is None:
        vocab = sorted({km for c in counts for km in c.keys()})
    V = len(vocab)
    mat = np.zeros((len(seqs), V), dtype=int)
    idx = {v:i for i,v in enumerate(vocab)}
    for i,c in enumerate(counts):
        for kmer, ct in c.items():
            mat[i, idx[kmer]] = ct
    return mat, vocab

# ---------- generate synthetic dataset ----------
random.seed(0)
N = 500  # total size (adjustable)
L = 30   # sequence length
motif = "TATA"
pos = 5  # insertion position for positives
k = 4

pos_seqs = []
neg_seqs = []
for _ in range(N//2):
    bg = random_seq(L)
    pos_seqs.append(insert_motif(bg, motif, pos))          # positive with motif
    # negative: ensure motif absent
    neg = random_seq(L)
    while motif in neg:
        neg = random_seq(L)
    neg_seqs.append(neg)

seqs = pos_seqs + neg_seqs
labels = [1]*(len(pos_seqs)) + [0]*(len(neg_seqs))

# ---------- features ----------
X, vocab = build_feature_matrix(seqs, k=k)
y = np.array(labels)

# split
X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
    X, y, seqs, test_size=0.3, random_state=42, stratify=y)

# train logistic regression (with small regularization)
clf = LogisticRegression(max_iter=1000, solver='liblinear')
clf.fit(X_train, y_train)

# eval
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# inspect top k-mers by coefficient
coefs = clf.coef_[0]  # shape (V,)
top_pos_idx = np.argsort(coefs)[-10:][::-1]
top_neg_idx = np.argsort(coefs)[:10]

print("Accuracy:", acc)
print("Confusion matrix:\n", cm)
print("\nTop positive k-mers (indicative of motif):")
for i in top_pos_idx:
    print(vocab[i], coefs[i])
print("\nTop negative k-mers:")
for i in top_neg_idx:
    print(vocab[i], coefs[i])


# Show a few example predictions with probabilities
print("\n=== Example sequence predictions ===")
for s in seq_test[:5]:  # just show first 5 test sequences
    feats, _ = build_feature_matrix([s], k=k, vocab=vocab)
    prob = clf.predict_proba(feats)[0,1]
    pred = clf.predict(feats)[0]
    print(f"Seq: {s}\n  Prediction: {pred} (prob={prob:.3f})\n")

# Highlight how much the motif contributes
print("\n=== Motif feature contribution check ===")
if "TATA" in vocab:
    idx_tata = vocab.index("TATA")
    print("Coefficient for 'TATA':", clf.coef_[0][idx_tata])
else:
    print("'TATA' not in vocab — maybe try different k")

