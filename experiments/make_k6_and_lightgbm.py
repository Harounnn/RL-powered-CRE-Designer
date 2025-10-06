#!/usr/bin/env python3
"""
Train a LightGBM regressor (compatible with LightGBM 4.6.0) on sparse k=6 features,
or fall back to a sparse linear model if LightGBM is not installed. Evaluates MAE &
Spearman, computes per-kmer Pearson correlations, writes top k-mers, and saves models/artifacts.

This version fixes dtype issues by converting sparse feature matrix to float32
and labels to float32 (LightGBM expects float32/float64).
"""

import os
import json
import joblib
import time
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from sklearn.linear_model import SGDRegressor

# Paths
DATA_DIR = "data/processed/sharpr"
MODEL_DIR = "models"
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

X_PATH = os.path.join(DATA_DIR, "X_k6.npz")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab_k6.json")

# Training hyperparams
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
LGB_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "verbosity": -1,
    "boosting_type": "gbdt",
    "seed": RANDOM_STATE,
    "n_jobs": -1,
}

FALLBACK_ALPHA = 1e-3

# Load data
print("Loading data...")
X = sp.load_npz(X_PATH)   # CSR sparse matrix
y = np.load(Y_PATH)
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
print("Loaded X", X.shape, "y", y.shape, "vocab", len(vocab))

# Ensure dtypes compatible with LightGBM: float32 for features and labels
if not np.issubdtype(X.dtype, np.floating):
    print("Casting sparse matrix data to float32 for LightGBM compatibility...")
    X = X.astype(np.float32)          # creates a new sparse matrix with float32 data
else:
    # still make sure it's float32 (not float64) to save memory and match LGB expectations
    if X.dtype != np.float32:
        X = X.astype(np.float32)

if not np.issubdtype(y.dtype, np.floating) or y.dtype != np.float32:
    y = y.astype(np.float32)

# Basic checks
n_samples, n_features = X.shape
assert len(vocab) == n_features, "vocab length != n_features"

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Train/test split:", X_train.shape, X_test.shape)

# Try LightGBM
use_lgb = False
try:
    import lightgbm as lgb
    use_lgb = True
    print("LightGBM imported:", lgb.__version__)
except Exception as e:
    print("LightGBM import failed; falling back to SGDRegressor. Exception:", e)
    use_lgb = False

model_info = {}
start_time = time.time()

if use_lgb:
    print("Preparing a small validation split from training set for early stopping...")
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)

    # LightGBM Dataset objects accept scipy CSR matrices with float32 data
    dtrain = lgb.Dataset(X_tr_sub, label=y_tr_sub, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    print("Training LightGBM with callbacks (compatible with 4.6.0)...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=50),
    ]

    booster = lgb.train(
        params=LGB_PARAMS,
        train_set=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["validation"],
        callbacks=callbacks,
    )

    elapsed = time.time() - start_time
    best_iter = int(booster.best_iteration) if booster.best_iteration is not None else None
    print(f"LightGBM training done in {elapsed:.1f}s; best_iteration = {best_iter}")

    # Save booster to a file (text format)
    model_file = os.path.join(MODEL_DIR, "predictor_sharpr_lightgbm.txt")
    booster.save_model(model_file)
    print("Saved LightGBM booster to", model_file)
    model_info["type"] = "lightgbm"
    model_info["model_file"] = model_file
    model_info["best_iteration"] = best_iter
    model = booster  # use booster for prediction
else:
    print("Training fallback sparse linear model (SGDRegressor)...")
    sgd = SGDRegressor(loss="squared_loss", penalty="l2", alpha=FALLBACK_ALPHA, max_iter=1000, tol=1e-4, random_state=RANDOM_STATE)
    sgd.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"SGD training done in {elapsed:.1f}s")
    model_file = os.path.join(MODEL_DIR, "predictor_sharpr_sgd.joblib")
    joblib.dump(sgd, model_file)
    print("Saved SGD model to", model_file)
    model_info["type"] = "sgd"
    model_info["model_file"] = model_file
    model = sgd

# Evaluation
print("Evaluating on test set...")
if use_lgb:
    num_iter = model_info.get("best_iteration", None)
    if num_iter is None or num_iter <= 0:
        y_pred = booster.predict(X_test)
    else:
        y_pred = booster.predict(X_test, num_iteration=num_iter)
else:
    y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
spearman_r = spearmanr(y_test, y_pred).correlation
print(f"Test MAE: {mae:.6f}")
print(f"Test Spearman r: {spearman_r:.6f}")

# Compute per-kmer Pearson correlation (efficient with sparse matrices)
print("Computing per-kmer Pearson correlations...")
n = X.shape[0]
y_vec = y.astype(float)
mean_y = y_vec.mean()
std_y = y_vec.std(ddof=0)
if std_y == 0:
    print("Warning: zero std in target")

# compute col sums and sums of squares
col_sum = np.asarray(X.sum(axis=0)).ravel()
mean_x = col_sum / n
col_sq_sum = np.asarray(X.multiply(X).sum(axis=0)).ravel()
E_x2 = col_sq_sum / n
var_x = E_x2 - mean_x * mean_x
var_x[var_x < 0] = 0.0
std_x = np.sqrt(var_x)

# compute E[x*y]
xy_sum = np.asarray(X.T.dot(y_vec)).ravel()
E_xy = xy_sum / n
cov_xy = E_xy - mean_x * mean_y

with np.errstate(divide='ignore', invalid='ignore'):
    corr = cov_xy / (std_x * std_y)
corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

# Save correlations
corr_path = os.path.join(DATA_DIR, "feature_corrs.npy")
np.save(corr_path, corr)
print("Saved per-kmer correlations to:", corr_path)

# Top k-mers
top_k = 50
order_pos = np.argsort(corr)[-top_k:][::-1]
order_neg = np.argsort(corr)[:top_k]
top_pos = [(vocab[i], float(corr[i])) for i in order_pos]
top_neg = [(vocab[i], float(corr[i])) for i in order_neg]

pos_path = os.path.join(REPORT_DIR, "top_kmers_pos.txt")
neg_path = os.path.join(REPORT_DIR, "top_kmers_neg.txt")
with open(pos_path, "w") as f:
    for kmer, v in top_pos:
        f.write(f"{kmer}\t{v:.6f}\n")
with open(neg_path, "w") as f:
    for kmer, v in top_neg:
        f.write(f"{kmer}\t{v:.6f}\n")

print("\nTop positive-correlated k-mers (sample):")
for kmer, v in top_pos[:30]:
    print(kmer, f"{v:.6f}")

print("\nTop negative-correlated k-mers (sample):")
for kmer, v in top_neg[:30]:
    print(kmer, f"{v:.6f}")

# If LightGBM, also print top importances if available
if use_lgb:
    try:
        importances = booster.feature_importance(importance_type="gain")
        idx_imp = np.argsort(importances)[-30:][::-1]
        print("\nTop LightGBM importances (sample):")
        for i in idx_imp[:30]:
            print(vocab[i], importances[i])
    except Exception:
        pass

# save summary report
report_path = os.path.join(REPORT_DIR, "day3_lightgbm_report.txt")
with open(report_path, "w") as f:
    f.write(f"Model type: {model_info.get('type','unknown')}\n")
    f.write(f"Model file: {model_info.get('model_file')}\n")
    f.write(f"Test MAE: {mae:.6f}\n")
    f.write(f"Test Spearman r: {spearman_r:.6f}\n\n")
    f.write("Top positive-correlated k-mers:\n")
    for kmer, v in top_pos:
        f.write(f"{kmer}\t{v:.6f}\n")
    f.write("\nTop negative-correlated k-mers:\n")
    for kmer, v in top_neg:
        f.write(f"{kmer}\t{v:.6f}\n")
    if use_lgb:
        f.write("\nNote: LightGBM booster saved as text file; load with lgb.Booster(model_file=path)\n")

print("\nSaved summary report to:", report_path)
print("\nDone.")