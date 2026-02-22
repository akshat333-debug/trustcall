"""
Fast RawNet training with pre-cached audio for speed.
Pre-loads all audio into RAM as numpy arrays, then trains fast.

Usage:
    python train_rawnet.py --data_path "data/ASVspoof 2019 Dataset 2/LA/LA" \
                           --epochs 10 --batch_size 32
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import librosa
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from model import RawNet

SAMPLE_RATE = 16000
MAX_LEN     = 64000   # 4 seconds at 16kHz


def pad_or_trim(x, max_len=MAX_LEN):
    if len(x) >= max_len:
        return x[:max_len]
    return np.tile(x, int(max_len / len(x)) + 1)[:max_len]


def load_protocol(data_path, split):
    split_map = {
        'train': ('ASVspoof2019_LA_train', 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),
        'dev':   ('ASVspoof2019_LA_dev',   'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),
    }
    audio_dir_name, proto_name = split_map[split]
    audio_dir  = os.path.join(data_path, audio_dir_name, 'flac')
    proto_path = os.path.join(data_path, proto_name)

    samples, labels = [], []
    with open(proto_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id    = parts[1]
            label_str = parts[4]
            label     = 0 if label_str == 'bonafide' else 1
            fpath     = os.path.join(audio_dir, utt_id + '.flac')
            if os.path.exists(fpath):
                samples.append(fpath)
                labels.append(label)
    return samples, labels


def stratified_sample(samples, labels, n_real, n_fake, seed=42):
    rng = random.Random(seed)
    real_idx = [i for i, l in enumerate(labels) if l == 0]
    fake_idx = [i for i, l in enumerate(labels) if l == 1]
    chosen_real = rng.sample(real_idx, min(n_real, len(real_idx)))
    chosen_fake = rng.sample(fake_idx, min(n_fake, len(fake_idx)))
    chosen = chosen_real + chosen_fake
    rng.shuffle(chosen)
    return [samples[i] for i in chosen], [labels[i] for i in chosen]


def preload_audio(file_paths, labels, desc='Loading'):
    """Pre-load all audio into RAM as numpy float32 arrays."""
    X, Y = [], []
    for fpath, label in tqdm(zip(file_paths, labels), total=len(file_paths), desc=f'  {desc}'):
        try:
            y, sr = librosa.load(fpath, sr=None, mono=True)
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            y = pad_or_trim(y)
            X.append(y)
            Y.append(label)
        except Exception as e:
            pass  # skip corrupt files
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


def train_epoch(model, X_t, Y_t, optimizer, criterion, device, batch_size=32):
    model.train()
    n = len(X_t)
    idx = torch.randperm(n)
    total_loss, correct = 0.0, 0

    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        x = X_t[batch_idx].to(device)
        y = Y_t[batch_idx].to(device)

        optimizer.zero_grad()
        out_binary, out_multi = model(x)
        # ASVspoof provides binary labels only (0=real, 1=fake).
        # Use the binary head as the supervised objective.
        loss = criterion(out_binary, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out_binary.argmax(1) == y).sum().item()

    return total_loss / (n / batch_size), correct / n


@torch.no_grad()
def eval_epoch(model, X_t, Y_t, criterion, device, batch_size=64):
    model.eval()
    n = len(X_t)
    total_loss, correct = 0.0, 0
    all_scores, all_labels = [], []

    for start in range(0, n, batch_size):
        x = X_t[start:start + batch_size].to(device)
        y = Y_t[start:start + batch_size].to(device)
        out_binary, _ = model(x)
        loss = criterion(out_binary, y)
        total_loss += loss.item()
        correct    += (out_binary.argmax(1) == y).sum().item()
        probs = torch.exp(out_binary)
        all_scores.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = correct / n
    # EER
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return total_loss / (n / batch_size), acc, eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   default='data/ASVspoof 2019 Dataset 2/LA/LA')
    parser.add_argument('--config',      default='model_config_RawNet.yaml')
    parser.add_argument('--out_dir',     default='outputs')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=0.0001)
    parser.add_argument('--n_real',      type=int,   default=500,  help='# real train samples')
    parser.add_argument('--n_fake',      type=int,   default=1500, help='# fake train samples')
    parser.add_argument('--n_dev_real',  type=int,   default=200)
    parser.add_argument('--n_dev_fake',  type=int,   default=600)
    parser.add_argument('--seed',        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use MPS if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*55}")
    print(f"  TrustCall — RawNet Fast Training")
    print(f"{'='*55}")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Train  : {args.n_real} real + {args.n_fake} fake = {args.n_real+args.n_fake} total")
    print(f"  Dev    : {args.n_dev_real} real + {args.n_dev_fake} fake")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("\n  Selecting samples...")
    tr_files, tr_labels = load_protocol(args.data_path, 'train')
    dv_files, dv_labels = load_protocol(args.data_path, 'dev')

    tr_files, tr_labels = stratified_sample(tr_files, tr_labels, args.n_real, args.n_fake, args.seed)
    dv_files, dv_labels = stratified_sample(dv_files, dv_labels, args.n_dev_real, args.n_dev_fake, args.seed)

    print(f"\n  Pre-loading train audio ({len(tr_files)} files)...")
    X_train, Y_train = preload_audio(tr_files, tr_labels, 'Train')
    print(f"  Pre-loading dev audio ({len(dv_files)} files)...")
    X_dev, Y_dev     = preload_audio(dv_files, dv_labels, 'Dev  ')

    # Move to tensors (keep on CPU, move batches to device during training)
    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    X_dev   = torch.tensor(X_dev)
    Y_dev   = torch.tensor(Y_dev)

    print(f"\n  Train tensor: {X_train.shape}  Dev tensor: {X_dev.shape}")

    model = RawNet(config['model'], device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.NLLLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_eer  = float('inf')
    best_path = os.path.join(args.out_dir, 'best_model.pth')

    print(f"\n  {'Epoch':>5}  {'Tr Loss':>8}  {'Tr Acc':>7}  {'Dv Loss':>8}  {'Dv Acc':>7}  {'EER':>6}")
    print(f"  {'-'*50}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, X_train, Y_train, optimizer, criterion, device, args.batch_size)
        dv_loss, dv_acc, eer = eval_epoch(model, X_dev, Y_dev, criterion, device, args.batch_size)
        scheduler.step()

        marker = ' ✓' if eer < best_eer else ''
        print(f"  {epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {dv_loss:>8.4f}  {dv_acc:>7.4f}  {eer:>6.4f}{marker}")

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), best_path)

    print(f"\n  ✅ Training complete! Best EER: {best_eer:.4f}")
    print(f"  Model saved: {best_path}")


if __name__ == '__main__':
    main()
