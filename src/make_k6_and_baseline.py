"""
Build overlapping 6-mer features (sparse) from processed SHARPR table,
filter low coverage (rna+dna >= 10), save outputs, and run a quick baseline.
"""
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import scipy.sparse as sp
import os

# Parameters
K = 6
MIN_COVERAGE = 10   # rna_total + dna_total >= MIN_COVERAGE
INPUT_CSV = "data/processed/sharpr/processed_sharpr_sequence_activity.csv"
OUT_DIR = "data/processed/sharpr/"
os.makedirs(OUT_DIR, exist_ok=True)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = None   
MIN_DF = 2           

# --------------- helpers ---------------
def kmer_tokenizer_factory(k):
    def tokenizer(seq):
        seq = seq.upper()
        n = len(seq)
        if n < k:
            return []
        return [seq[i:i+k] for i in range(n - k + 1)]
    return tokenizer

# --------------- load & filter ---------------
print("Loading processed table:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV, dtype={'sequence': str})
# ensure these columns exist
if 'rna_total' not in df.columns or 'dna_total' not in df.columns:
    raise ValueError("Expected 'rna_total' and 'dna_total' columns in processed table. Run processing step first.")

df['coverage'] = df['rna_total'].fillna(0).astype(int) + df['dna_total'].fillna(0).astype(int)
print("Total sequences before filter:", len(df))
df = df[df['coverage'] >= MIN_COVERAGE].copy()
print("Sequences after coverage filter (>= {}): {}".format(MIN_COVERAGE, len(df)))

# target
if 'log2_ratio' not in df.columns:
    # fallback - compute if not present
    df['log2_ratio'] = np.log2((df['rna_total'].astype(float) + 1.0) / (df['dna_total'].astype(float) + 1.0))
y = df['log2_ratio'].values
seqs = df['sequence'].astype(str).tolist()

# --------------- build sparse k-mer features via CountVectorizer ---------------
print("Building k-mer features (k={}) using CountVectorizer...".format(K))
tokenizer = kmer_tokenizer_factory(K)
vect = CountVectorizer(analyzer=tokenizer, lowercase=False, token_pattern=None,
                       min_df=MIN_DF, max_features=MAX_FEATURES)

# Fit + transform (sparse CSR)
X_sparse = vect.fit_transform(seqs)  # shape: (n_samples, n_features)
vocab = vect.get_feature_names_out()
print("Feature matrix shape:", X_sparse.shape)
print("Vocab size:", len(vocab))
density = X_sparse.nnz / float(X_sparse.shape[0]*X_sparse.shape[1])
print("Sparsity (nnz):", X_sparse.nnz, "density:", density)

# --------------- save outputs ---------------
# Save sparse matrix (scipy .npz), vocab (json), and y (npy)
sp.save_npz(os.path.join(OUT_DIR, "X_k6.npz"), X_sparse)
np.save(os.path.join(OUT_DIR, "y.npy"), y)
with open(os.path.join(OUT_DIR, "vocab_k6.json"), "w") as f:
    json.dump(vocab.tolist(), f)
print("Saved X_k6.npz, y.npy, vocab_k6.json in", OUT_DIR)

# --------------- quick baseline (sparse-aware) ---------------
print("Training quick baseline (SGDRegressor) on sparse features...")
X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
# Use a simple linear regressor with L2 reg; SGDRegressor accepts sparse input
reg = SGDRegressor(loss='squared_error', penalty='l2', alpha=1e-3, max_iter=1000, tol=1e-4, random_state=RANDOM_STATE)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, pred)
spearman_r = spearmanr(y_test, pred).correlation
print("Baseline results -> MAE: {:.4f}, Spearman r: {:.4f}".format(mae, spearman_r))

# --------------- inspect top k-mer coefficients ---------------
print("\nTop positive k-mers (by coefficient):")
coef = reg.coef_
# some sklearn SGDRegressor implementations don't guarantee dense coef when sparse input -> it's dense
if hasattr(vocab, '__len__'):
    idx_desc = np.argsort(coef)[-20:][::-1]
    for i in idx_desc[:20]:
        print(vocab[i], coef[i])
else:
    print("No vocab available for coef inspection.")

print("\nTop negative k-mers (by coefficient):")
idx_asc = np.argsort(coef)[:20]
for i in idx_asc[:20]:
    print(vocab[i], coef[i])

print("\nDone.")
