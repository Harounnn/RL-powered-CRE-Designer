import os
import random
import time
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# Config 
DATA_DIR = "data/processed/sharpr"
CSV_PATH = os.path.join(DATA_DIR, "processed_sharpr_sequence_activity.csv")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
MODEL_DIR = "models"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
MIN_COVERAGE = 10   
DEFAULT_BATCH = 128
DEFAULT_EPOCHS = 30
PATIENCE = 6  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utilities
base2idx = {b"A":0, b"C":1, b"G":2, b"T":3}
complement_map = {b"A":b"T", b"T":b"A", b"C":b"G", b"G":b"C", b"N":b"N"}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rc_seq_bytes(s: bytes) -> bytes:
    return b"".join(complement_map.get(ch, b"N") for ch in s[::-1])

def center_pad_truncate(seq: bytes, L: int) -> bytes:
    if len(seq) == L:
        return seq
    if len(seq) > L:
        start = (len(seq) - L) // 2
        return seq[start:start+L]
    # pad
    pad_left = (L - len(seq)) // 2
    pad_right = L - len(seq) - pad_left
    return b"N"*pad_left + seq + b"N"*pad_right

def one_hot_encode_bytes(seq: bytes, L: int) -> np.ndarray:
    seq = center_pad_truncate(seq, L)
    arr = np.zeros((4, L), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = base2idx.get(bytes([ch]), None)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr

# Dataset
class SharprSequenceDataset(Dataset):
    def __init__(self, sequences: List[str], targets: np.ndarray, seq_len: int,
                 augment_rc: bool = False):
        assert len(sequences) == len(targets)
        self.seq_bytes = [s.encode("ascii") for s in sequences]
        self.targets = targets.astype(np.float32)
        self.seq_len = seq_len
        self.augment_rc = augment_rc

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        s = self.seq_bytes[idx]
        # augmentation
        if self.augment_rc and random.random() < 0.5:
            s = rc_seq_bytes(s)
        x = one_hot_encode_bytes(s, self.seq_len)  # (4, L)
        y = self.targets[idx]
        return x, y

# Model
class SmallCNN(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)

        # compute final size after pooling
        L = seq_len
        L = L // 2  # after pool1
        L = L // 2  # after pool2
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # reduces to (batch, channels, 1)

        self.fc1 = nn.Linear(64, 128)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)  # (batch, channels)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x).squeeze(-1)
        return x

# Training / evaluation helpers
def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae = float(np.mean(np.abs(preds - trues)))
    rho = spearmanr(trues, preds).correlation
    return mae, rho, preds, trues

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--pilot", type=int, default=0,
                        help="If >0, use this many samples (random) for quick pilot")
    parser.add_argument("--seq_len", type=int, default=0,
                        help="If 0, auto-detect from first sequence; otherwise enforce length")
    parser.add_argument("--augment_rc", action="store_true", help="Use reverse-complement augmentation")
    parser.add_argument("--min_coverage", type=int, default=MIN_COVERAGE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    set_seed(SEED)
    print("Device:", DEVICE)
    print("Loading processed CSV and y.npy ...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("Missing processed CSV: " + CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    df['coverage'] = df['rna_total'].fillna(0).astype(int) + df['dna_total'].fillna(0).astype(int)
    df = df[df['coverage'] >= args.min_coverage].copy()
    seqs = df['sequence'].astype(str).tolist()

    y = np.load(Y_PATH)
    if len(seqs) != len(y):
        raise RuntimeError(f"Length mismatch: sequences {len(seqs)} vs y {len(y)}. Ensure y.npy matches filtered CSV.")
    print("Total sequences after coverage filter:", len(seqs))

    # pilot subset
    indices = np.arange(len(seqs))
    if args.pilot and args.pilot < len(seqs):
        indices = np.random.choice(indices, size=args.pilot, replace=False)
        seqs = [seqs[i] for i in indices]
        y = y[indices]
        print(f"Pilot mode: using {len(seqs)} samples")

    # sequence length handling
    if args.seq_len and args.seq_len > 0:
        L = int(args.seq_len)
    else:
        # auto-detect modal length and use it
        lengths = [len(s) for s in seqs]
        L = int(np.median(lengths))
        print(f"Auto-detected seq_len (median): {L}")

    # Build dataset and dataloaders
    dataset = SharprSequenceDataset(seqs, y, seq_len=L, augment_rc=args.augment_rc)
    # train/val/test split (80/10/10)
    n = len(dataset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = SmallCNN(seq_len=L).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss() 

    best_val_spearman = -999.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for xb_np, yb in train_loader:
            xb = torch.tensor(xb_np, dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(yb, dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1

        train_loss = running_loss / max(1, batches)
        val_mae, val_spear, _, _ = evaluate_model(model, val_loader, DEVICE)
        test_mae, test_spear, _, _ = evaluate_model(model, test_loader, DEVICE)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_mae={val_mae:.4f} val_spearman={val_spear:.4f} | "
              f"test_mae={test_mae:.4f} test_spearman={test_spear:.4f} | elapsed={elapsed/60:.2f}m")

        # Early stopping on val spearman
        if val_spear > best_val_spearman + 1e-4:
            best_val_spearman = val_spear
            best_epoch = epoch
            best_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_state, os.path.join(MODEL_DIR, "predictor_sharpr_cnn_best.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered (no improvement).")
                break

    total_time = time.time() - start_time
    print(f"Training complete. Best val spearman={best_val_spearman:.4f} at epoch {best_epoch}. Total time {total_time/60:.2f}m")

    # Save final model & report
    final_model_path = os.path.join(MODEL_DIR, "predictor_sharpr_cnn.pth")
    if best_state is not None:
        torch.save(best_state, final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)

    report = {
        "device": str(DEVICE),
        "seq_len": L,
        "n_samples": len(dataset),
        "batch_size": args.batch,
        "epochs_ran": epoch,
        "best_epoch": best_epoch,
        "best_val_spearman": float(best_val_spearman),
        "test_spearman_at_best": float(test_spear),
        "total_time_sec": float(total_time),
        "pilot": int(args.pilot),
    }
    report_path = os.path.join(REPORT_DIR, "c3_cnn_report.json")
    import json
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("Saved report to", report_path)
    print("Saved model to", final_model_path)

if __name__ == "__main__":
    main()
