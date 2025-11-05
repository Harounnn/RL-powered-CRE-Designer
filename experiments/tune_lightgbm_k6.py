import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy import sparse
from itertools import product
import time
import os

# paths
X_PATH = "data/processed/sharpr/X_k6.npz"
Y_PATH = "data/processed/sharpr/y.npy"
OUT_DIR = "data/processed/sharpr/reports"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Load data
print("Loading data...")
X = sparse.load_npz(X_PATH)    # possibly integer dtype
y = np.load(Y_PATH)

print("Raw dtypes:", X.dtype, y.dtype)
# Cast features/labels to float32 for LightGBM compatibility
if not np.issubdtype(X.dtype, np.floating) or X.dtype != np.float32:
    print("Casting X to float32...")
    X = X.astype(np.float32)
if not np.issubdtype(y.dtype, np.floating) or y.dtype != np.float32:
    print("Casting y to float32...")
    y = y.astype(np.float32)

print(f"Feature matrix: {X.shape}, Labels: {y.shape}, dtypes: X={X.dtype}, y={y.dtype}")

# Split: use a hold-out validation (10%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# -------------------------------------------------------------------
# Parameter grid (kept small)
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [63, 127, 255],
    "min_data_in_leaf": [20, 100],
    "feature_fraction": [0.8, 1.0],
    "lambda_l1": [0, 0.1],
    "lambda_l2": [0, 0.1],
}

BASE_PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}

results = []
start_time = time.time()

# iterate
for (lr, nl, mdl, ff, l1, l2) in product(
    param_grid["learning_rate"],
    param_grid["num_leaves"],
    param_grid["min_data_in_leaf"],
    param_grid["feature_fraction"],
    param_grid["lambda_l1"],
    param_grid["lambda_l2"],
):
    params = BASE_PARAMS.copy()
    params.update(
        dict(
            learning_rate=lr,
            num_leaves=nl,
            min_data_in_leaf=mdl,
            feature_fraction=ff,
            lambda_l1=l1,
            lambda_l2=l2,
        )
    )

    print(f"\nTraining: lr={lr}, num_leaves={nl}, min_data_in_leaf={mdl}, ff={ff}, l1={l1}, l2={l2}")
    # Create Datasets (LightGBM accepts scipy CSR with float32)
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200),
        ],
    )

    y_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    mae = mean_absolute_error(y_val, y_pred)
    spearman = spearmanr(y_val, y_pred).correlation

    print(f" -> MAE={mae:.4f}, Spearman={spearman:.4f}")

    results.append(
        dict(
            lr=lr,
            num_leaves=nl,
            min_data_in_leaf=mdl,
            feature_fraction=ff,
            lambda_l1=l1,
            lambda_l2=l2,
            mae=mae,
            spearman=spearman,
            best_iter=int(booster.best_iteration) if booster.best_iteration is not None else None,
        )
    )

# -------------------------------------------------------------------
res_df = pd.DataFrame(results).sort_values("spearman", ascending=False)
print("\n=== Top 10 Results ===")
print(res_df.head(10).to_string(index=False))

out_csv = os.path.join(OUT_DIR, "lgbm_grid_results.csv")
res_df.to_csv(out_csv, index=False)
print(f"\nSaved all results in {out_csv}")
print(f"Total runtime: {(time.time() - start_time)/60:.2f} min")
