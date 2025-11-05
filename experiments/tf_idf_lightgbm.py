import os
import time
import json
import joblib
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfTransformer

# Paths
DATA_DIR = "data/processed/sharpr"
X_PATH = os.path.join(DATA_DIR, "X_k6.npz")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab_k6.json")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
MODEL_DIR = "models"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
print("Loading X_k6 and y...")
X = sp.load_npz(X_PATH)   
y = np.load(Y_PATH)
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

print("Raw dtypes:", X.dtype, y.dtype)
# Cast to float32 if needed (LightGBM requires float32/float64)
if not np.issubdtype(X.dtype, np.floating) or X.dtype != np.float32:
    X = X.astype(np.float32)
if not np.issubdtype(y.dtype, np.floating) or y.dtype != np.float32:
    y = y.astype(np.float32)

print("Shapes:", X.shape, y.shape)

# Build TF-IDF weighted matrix (sparse)
print("Applying TF-IDF weighting to k-mer counts (TfidfTransformer)...")
tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
# TfidfTransformer expects CSR/COO; X is CSR already
X_tfidf = tfidf.fit_transform(X)  # sparse float64 by default
# convert to float32 to save memory & for LightGBM
X_tfidf = X_tfidf.astype(np.float32)
print("TF-IDF matrix dtype:", X_tfidf.dtype, "shape:", X_tfidf.shape)

# Train/validation/test split
print("Creating train/val/test split (80/10/10)...")
X_temp, X_test, y_temp, y_test = train_test_split(X_tfidf, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111111, random_state=42)  # 0.1111*0.9 = 0.10 => 0.8/0.1/0.1
print("Splits:", X_train.shape, X_val.shape, X_test.shape)

# LightGBM params (best from tuning)
lgb_params = {
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

# Train LightGBM (callback-based early stopping for v4.x)
use_lgb = True
try:
    import lightgbm as lgb
except Exception as e:
    print("LightGBM import failed:", e)
    use_lgb = False

report = {}
start = time.time()

if use_lgb:
    print("Preparing LightGBM Dataset objects...")
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    print("Training LightGBM on TF-IDF features...")
    booster = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100),
        ],
    )

    best_iter = int(booster.best_iteration) if booster.best_iteration is not None else None
    print("Best iteration:", best_iter)

    # Predict and evaluate on validation and test sets
    y_val_pred = booster.predict(X_val, num_iteration=best_iter)
    y_test_pred = booster.predict(X_test, num_iteration=best_iter)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_spearman = spearmanr(y_val, y_val_pred).correlation
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_spearman = spearmanr(y_test, y_test_pred).correlation

    # Save model (LightGBM booster text file)
    model_file = os.path.join(MODEL_DIR, "predictor_sharpr_lgbm_tfidf.txt")
    booster.save_model(model_file)
    print("Saved LightGBM model to", model_file)

    elapsed = time.time() - start
    report.update({
        "model": "lightgbm_tfidf",
        "model_file": model_file,
        "best_iteration": best_iter,
        "val_mae": float(val_mae),
        "val_spearman": float(val_spearman),
        "test_mae": float(test_mae),
        "test_spearman": float(test_spearman),
        "elapsed_sec": elapsed,
    })
else:
    # Fallback: train SGD on TF-IDF features (dense) -- but prefer LGB if available
    from sklearn.linear_model import SGDRegressor
    print("Training SGDRegressor fallback on TF-IDF features (this is slower)...")
    sgd = SGDRegressor(loss="squared_loss", penalty="l2", alpha=1e-3, max_iter=2000, tol=1e-4, random_state=42)
    sgd.fit(X_train, y_train)
    y_val_pred = sgd.predict(X_val)
    y_test_pred = sgd.predict(X_test)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_spearman = spearmanr(y_val, y_val_pred).correlation
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_spearman = spearmanr(y_test, y_test_pred).correlation
    model_file = os.path.join(MODEL_DIR, "predictor_sharpr_sgd_tfidf.joblib")
    joblib.dump(sgd, model_file)
    report.update({
        "model": "sgd_tfidf",
        "model_file": model_file,
        "val_mae": float(val_mae),
        "val_spearman": float(val_spearman),
        "test_mae": float(test_mae),
        "test_spearman": float(test_spearman),
        "elapsed_sec": time.time() - start,
    })

# Save report & top features (correlations) for TF-IDF
report_path = os.path.join(REPORT_DIR, "tfidf_lightgbm_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print("Saved TF-IDF report to", report_path)

# Compute per-kmer Pearson correlations on TF-IDF matrix (proxy importance)
print("Computing per-kmer Pearson correlations on TF-IDF matrix...")
n = X_tfidf.shape[0]
y_vec = y.astype(float)
mean_y = y_vec.mean()
std_y = y_vec.std(ddof=0)
if std_y == 0:
    print("Warning: zero std in target")

col_sum = np.asarray(X_tfidf.sum(axis=0)).ravel()
mean_x = col_sum / n
col_sq_sum = np.asarray(X_tfidf.multiply(X_tfidf).sum(axis=0)).ravel()
E_x2 = col_sq_sum / n
var_x = E_x2 - mean_x * mean_x
var_x[var_x < 0] = 0.0
std_x = np.sqrt(var_x)
xy_sum = np.asarray(X_tfidf.T.dot(y_vec)).ravel()
E_xy = xy_sum / n
cov_xy = E_xy - mean_x * mean_y
with np.errstate(divide='ignore', invalid='ignore'):
    corr = cov_xy / (std_x * std_y)
corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

# Save correlations and top k-mers
corr_path = os.path.join(DATA_DIR, "feature_corrs_tfidf.npy")
np.save(corr_path, corr)
print("Saved TF-IDF feature correlations to", corr_path)

# Top positives/negatives
top_k = 50
order_pos = np.argsort(corr)[-top_k:][::-1]
order_neg = np.argsort(corr)[:top_k]
with open(os.path.join(REPORT_DIR, "top_kmers_tfidf_pos.txt"), "w") as fpos:
    for i in order_pos:
        fpos.write(f"{vocab[i]}\t{corr[i]:.6f}\n")
with open(os.path.join(REPORT_DIR, "top_kmers_tfidf_neg.txt"), "w") as fneg:
    for i in order_neg:
        fneg.write(f"{vocab[i]}\t{corr[i]:.6f}\n")

print("Saved top k-mer lists to reports folder.")

# Print short summary
print("\n=== TF-IDF LightGBM summary ===")
for k, v in report.items():
    print(k, v)

print("\nDone.")
