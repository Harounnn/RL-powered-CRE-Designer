import os
import time
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import lightgbm as lgb

DATA_DIR = "data/processed/sharpr"
REPORT_DIR = os.path.join(DATA_DIR, "reports")
MODEL_DIR = "models"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

X_PATH = os.path.join(DATA_DIR, "X_k6k7.npz")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
REPORT_PATH = os.path.join(REPORT_DIR, "c2_svd_report.txt")

SVD_COMPONENTS = [200, 500]  
TEST_SIZE = 0.2
RANDOM_STATE = 42

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
    "seed": RANDOM_STATE,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100

def train_and_eval(X, y, model_path, desc):
    """Train LightGBM on given dense features and evaluate."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

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

    best_iter = booster.best_iteration or NUM_BOOST_ROUND
    y_pred = booster.predict(X_test, num_iteration=best_iter)

    mae = mean_absolute_error(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred).correlation
    booster.save_model(model_path)

    return dict(
        desc=desc,
        best_iter=best_iter,
        mae=float(mae),
        spearman=float(spearman),
        elapsed_sec=float(elapsed),
        model=model_path,
    )

def main():
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"Missing file: {X_PATH}")
    if not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Missing file: {Y_PATH}")

    print("Loading hybrid feature matrix...")
    X_sparse = sparse.load_npz(X_PATH).astype(np.float32)
    y = np.load(Y_PATH).astype(np.float32)
    print(f"Matrix shape: {X_sparse.shape}, y: {y.shape}")

    results = []

    for n_comp in SVD_COMPONENTS:
        print(f"\n=== Running TruncatedSVD with n_components={n_comp} ===")
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        X_svd = svd.fit_transform(X_sparse)
        print(f"SVD done. Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
        print(f"Reduced shape: {X_svd.shape}")

        model_path = os.path.join(MODEL_DIR, f"predictor_sharpr_lgbm_svd{n_comp}.txt")
        res = train_and_eval(X_svd, y, model_path, desc=f"SVD-{n_comp}")
        results.append(res)
        print(f"Result ({n_comp}): MAE={res['mae']:.4f}, Spearman={res['spearman']:.4f}, time={res['elapsed_sec']:.1f}s")

    with open(REPORT_PATH, "w") as f:
        f.write("=== C2 SVD + LightGBM Results ===\n")
        for r in results:
            for k, v in r.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    print("\nAll results written to", REPORT_PATH)

if __name__ == "__main__":
    main()
