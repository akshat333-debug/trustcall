"""
Fast RawNet training with pre-cached audio for speed.
(Updated for memory-constrained environments like 8GB M3)
Streams audio dynamically from disk on-the-fly to prevent RAM overflow.

Usage:
    python train_rawnet.py --data_path "data/ASVspoof 2019 Dataset 2/LA/LA" \
                           --epochs 100 --batch_size 32
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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


class ASVspoofDataset(Dataset):
    def __init__(self, file_paths, labels, max_len=MAX_LEN):
        self.file_paths = file_paths
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = self.file_paths[idx]
        label = self.labels[idx]
        try:
            y, sr = librosa.load(fpath, sr=None, mono=True)
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            y = pad_or_trim(y, self.max_len)
        except Exception:
            # Fallback for corrupt files
            y = np.zeros(self.max_len, dtype=np.float32)
        
        return torch.tensor(y, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out_binary, out_multi = model(x)
        loss = criterion(out_binary, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        correct += (out_binary.argmax(1) == y).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_scores, all_labels = [], []

    pbar = tqdm(dataloader, desc="  Evaluating", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out_binary, _ = model(x)
        loss = criterion(out_binary, y)
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        correct += (out_binary.argmax(1) == y).sum().item()
        total += batch_size
        
        probs = torch.exp(out_binary)
        all_scores.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = correct / total
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return total_loss / total, acc, eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   default='data/ASVspoof 2019 Dataset 2/LA/LA')
    parser.add_argument('--config',      default='model_config_RawNet.yaml')
    parser.add_argument('--out_dir',     default='outputs')
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=0.0001)
    parser.add_argument('--n_real',      type=int,   default=None,  help='Max real train samples (None=all)')
    parser.add_argument('--n_fake',      type=int,   default=None,  help='Max fake train samples (None=all)')
    parser.add_argument('--n_dev_real',  type=int,   default=None)
    parser.add_argument('--n_dev_fake',  type=int,   default=None)
    parser.add_argument('--seed',        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*55}")
    print(f"  TrustCall — RawNet Full On-the-Fly Training")
    print(f"{'='*55}")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("\n  Indexing files...")
    tr_files, tr_labels = load_protocol(args.data_path, 'train')
    dv_files, dv_labels = load_protocol(args.data_path, 'dev')

    if args.n_real is not None or args.n_fake is not None:
        tr_files, tr_labels = stratified_sample(tr_files, tr_labels, args.n_real or 999999, args.n_fake or 999999, args.seed)
    if args.n_dev_real is not None or args.n_dev_fake is not None:
        dv_files, dv_labels = stratified_sample(dv_files, dv_labels, args.n_dev_real or 999999, args.n_dev_fake or 999999, args.seed)

    print(f"  Train : {len(tr_files)} samples")
    print(f"  Dev   : {len(dv_files)} samples")

    tr_dataset = ASVspoofDataset(tr_files, tr_labels)
    dv_dataset = ASVspoofDataset(dv_files, dv_labels)

    # num_workers=0 is crucial for 8GB RAM to prevent multiprocessing memory leaks
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dv_loader = DataLoader(dv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RawNet(config['model'], device).to(device)
    
    # Optional: Load best model checkpoint if it exists to resume
    best_path = os.path.join(args.out_dir, 'best_model.pth')
    if os.path.exists(best_path):
        print(f"  Existing checkpoint found: {best_path} (Starting fresh to avoid overwriting unless intended, use carefully!)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.NLLLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_eer  = float('inf')

    print(f"\n  {'Epoch':>5}  {'Tr Loss':>8}  {'Tr Acc':>7}  {'Dv Loss':>8}  {'Dv Acc':>7}  {'EER':>6}")
    print(f"  {'-'*50}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, device)
        dv_loss, dv_acc, eer = eval_epoch(model, dv_loader, criterion, device)
        scheduler.step()

        marker = ' ✓' if eer < best_eer else ''
        print(f"  {epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {dv_loss:>8.4f}  {dv_acc:>7.4f}  {eer:>6.4f}{marker}")

        # Always explicitly save the latest epoch as a backup
        epoch_path = os.path.join(args.out_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), epoch_path)

        if eer < best_eer:
            best_eer = eer
            # Overwrite the best_model.pth
            torch.save(model.state_dict(), best_path)

    print(f"\n  ✅ Training complete! Best EER: {best_eer:.4f}")
    print(f"  Model saved: {best_path}")


if __name__ == '__main__':
    main()
