import os
import json
import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import pandas as pd

try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("LightGBM is required to run this script: " + str(e))

DATA_DIR = "data/processed/sharpr"
OUT_REPORT = os.path.join(DATA_DIR, "reports")
MODEL_DIR = "models"
os.makedirs(OUT_REPORT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "processed_sharpr_sequence_activity.csv")
X_K6_PATH = os.path.join(DATA_DIR, "X_k6.npz")
VOCAB_K6_PATH = os.path.join(DATA_DIR, "vocab_k6.json")
X_K7_PATH = os.path.join(DATA_DIR, "X_k7.npz")
VOCAB_K7_PATH = os.path.join(DATA_DIR, "vocab_k7.json")
X_K6K7_PATH = os.path.join(DATA_DIR, "X_k6k7.npz")
VOCAB_K6K7_PATH = os.path.join(DATA_DIR, "vocab_k6k7.json")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
REPORT_TXT = os.path.join(OUT_REPORT, "c1_k7_report.txt")

LGB_PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "objective": "regression",
    "metric": "l1",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_DF = 2  
MIN_COVERAGE = 10  

# Tokenizer factory for k-mers
def kmer_tokenizer_factory(k):
    def tokenizer(seq):
        seq = seq.upper()
        n = len(seq)
        if n < k:
            return []
        return [seq[i:i+k] for i in range(n - k + 1)]
    return tokenizer

def build_kmer_matrix(seqs, k, min_df=2, max_features=None):
    tokenizer = kmer_tokenizer_factory(k)
    vect = CountVectorizer(analyzer=tokenizer, lowercase=False, token_pattern=None,
                           min_df=min_df, max_features=max_features)
    X = vect.fit_transform(seqs)
    vocab = vect.get_feature_names_out().tolist()
    return X, vocab

def save_sparse_csr(path, matrix):
    sparse.save_npz(path, matrix)

def load_sparse_csr(path):
    return sparse.load_npz(path)

def train_lgbm_and_eval(X, y, model_out_path):
    # ensure float32
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float32)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    if not np.issubdtype(y.dtype, np.floating) or y.dtype != np.float32:
        y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)

    dtrain = lgb.Dataset(X_tr_sub, label=y_tr_sub, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    start = time.time()
    booster = lgb.train(
        params=LGB_PARAMS,
        train_set=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=200),
        ],
    )
    elapsed = time.time() - start
    best_iter = int(booster.best_iteration) if booster.best_iteration is not None else None

    num_iter = best_iter if best_iter is not None else -1
    y_pred = booster.predict(X_test, num_iteration=num_iter)
    mae = mean_absolute_error(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred).correlation

    booster.save_model(model_out_path)
    return dict(model_path=model_out_path, best_iter=best_iter, mae=float(mae), spearman=float(spearman), elapsed_sec=float(elapsed))

def main():
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"Processed CSV not found at {CSV_PATH}. Cannot rebuild k7 aligned to y.npy.")
    df = pd.read_csv(CSV_PATH)
    for c in ('sequence','rna_total','dna_total'):
        if c not in df.columns:
            raise RuntimeError(f"Column '{c}' not found in processed CSV. Columns: {df.columns.tolist()}")

    df['coverage'] = df['rna_total'].fillna(0).astype(int) + df['dna_total'].fillna(0).astype(int)
    df_filtered = df[df['coverage'] >= MIN_COVERAGE].copy()
    seqs = df_filtered['sequence'].astype(str).tolist()

    if not os.path.exists(Y_PATH):
        raise RuntimeError("y.npy not found at " + Y_PATH)
    y = np.load(Y_PATH)
    if len(seqs) != len(y):
        raise RuntimeError(f"Number of filtered sequences ({len(seqs)}) != length of y.npy ({len(y)}). They must match.")

    print(f"Building k=7 features for {len(seqs)} sequences (coverage >= {MIN_COVERAGE}) ...")
    X_k7, vocab_k7 = build_kmer_matrix(seqs, k=7, min_df=MIN_DF)
    print("k7 matrix shape:", X_k7.shape, "vocab size:", len(vocab_k7))

    save_sparse_csr(X_K7_PATH, X_k7.tocsr())
    with open(VOCAB_K7_PATH, "w") as f:
        json.dump(vocab_k7, f)
    print("Saved X_k7 and vocab_k7")

    # Train LightGBM on k7
    print("Training LightGBM on k=7 features...")
    res_k7 = train_lgbm_and_eval(X_k7, y, os.path.join(MODEL_DIR, "predictor_sharpr_lgbm_k7.txt"))
    print("k7 training result:", res_k7)

    # Hybrid: load existing k6 matrix and horizontally stack
    if os.path.exists(X_K6_PATH):
        print("Loading existing k6 matrix and building hybrid k6|k7...")
        X_k6 = load_sparse_csr(X_K6_PATH).tocsr()
        if X_k6.shape[0] != X_k7.shape[0]:
            raise RuntimeError(f"Row mismatch between k6 ({X_k6.shape}) and k7 ({X_k7.shape}). They must have same ordering and count.")
        X_hybrid = sparse.hstack([X_k6, X_k7], format='csr')
        print("Hybrid shape:", X_hybrid.shape)
        save_sparse_csr(X_K6K7_PATH, X_hybrid)

        # combine vocab (attempt load k6 vocab)
        if os.path.exists(VOCAB_K6_PATH):
            with open(VOCAB_K6_PATH, "r") as f:
                v6 = json.load(f)
        else:
            v6 = [f"k6_{i}" for i in range(X_k6.shape[1])]
        v6k7 = v6 + vocab_k7
        with open(VOCAB_K6K7_PATH, "w") as f:
            json.dump(v6k7, f)
        print("Saved hybrid vocab.")

        print("Training LightGBM on hybrid k6+k7 features...")
        res_hybrid = train_lgbm_and_eval(X_hybrid, y, os.path.join(MODEL_DIR, "predictor_sharpr_lgbm_k6k7.txt"))
        print("Hybrid training result:", res_hybrid)
    else:
        print("k6 feature matrix not found at", X_K6_PATH)
        res_hybrid = None

    # write short report
    with open(REPORT_TXT, "w") as f:
        f.write("C1 k7 and hybrid report\n\n")
        f.write("k7 result:\n")
        for k, v in res_k7.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        if res_hybrid:
            f.write("k6+k7 hybrid result:\n")
            for k, v in res_hybrid.items():
                f.write(f"{k}: {v}\n")
        else:
            f.write("Hybrid not run (k6 matrix missing)\n")
    print("Wrote report to", REPORT_TXT)
    print("Done.")

if __name__ == "__main__":
    main()
