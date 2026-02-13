"""
Compute raw model predictions over the full k6 feature matrix and save:
 - models/predictor_raw_preds.npy     (raw predictions per sequence)
 - models/predictor_percentiles.npy   (sorted array used to map pred -> percentile)

This script assumes:
 - LightGBM model saved as a text model: data path in metadata or set below
 - X_k6.npz exists (sparse CSR) and rows align with y.npy / filtered CSV
"""
import os
import json
import numpy as np
import scipy.sparse as sp

METADATA_PATH = "data/processed/sharpr/reports/final_model_metadata.json"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def load_metadata(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    meta = load_metadata(METADATA_PATH)
    model_path = meta.get("model_file", "models/predictor_sharpr_lightgbm.txt")
    X_path = meta.get("feature_matrix", "data/processed/sharpr/X_k6.npz")

    print("Using model:", model_path)
    print("Using feature matrix:", X_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LightGBM model file not found: {model_path}")
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Feature matrix not found: {X_path}")

    print("Loading X_k6 sparse matrix...")
    X = sp.load_npz(X_path).astype(np.float32)
    print("X shape:", X.shape, "dtype:", X.dtype)

    print("Loading LightGBM...")
    try:
        import lightgbm as lgb
    except Exception as e:
        raise RuntimeError("LightGBM import failed: " + str(e))

    booster = lgb.Booster(model_file=model_path)
    best_iter = meta.get("best_iteration", None)
    if best_iter is not None:
        print("Using best_iteration from metadata:", best_iter)
    else:
        print("No best_iteration in metadata; will let booster.predict use default (all iters).")

    print("Predicting raw scores over full dataset (this may take a bit)...")
    if best_iter is not None:
        preds = booster.predict(X, num_iteration=best_iter)
    else:
        preds = booster.predict(X)

    preds = np.asarray(preds, dtype=np.float32)
    print("Preds shape:", preds.shape)

    raw_path = os.path.join(OUT_DIR, "predictor_raw_preds.npy")
    np.save(raw_path, preds)
    print("Saved raw predictions to:", raw_path)

    sorted_preds = np.sort(preds)
    sorted_path = os.path.join(OUT_DIR, "predictor_percentiles.npy")
    np.save(sorted_path, sorted_preds)
    print("Saved sorted predictions (for percentile mapping) to:", sorted_path)


    p10 = np.percentile(preds, 10)
    p50 = np.percentile(preds, 50)
    p90 = np.percentile(preds, 90)
    print(f"Summary raw preds: 10th={p10:.4f}, median={p50:.4f}, 90th={p90:.4f}")

    print("\nExample mapping function (raw -> percentile):")
    print("Given a raw score 's', compute:")
    print("  pct = np.searchsorted(sorted_preds, s, side='right') / len(sorted_preds)")
    print("This yields a value in [0,1].")

    print("\nDone.")

if __name__ == '__main__':
    main()
