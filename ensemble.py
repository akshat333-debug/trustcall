"""
Extension 9: LFCC-LCNN Model + Ensemble with RawNet
Implements a lightweight LCNN classifier on LFCC features,
then ensembles its predictions with RawNet for improved accuracy.

Usage:
    # Train LFCC-LCNN:
    python ensemble.py --mode train_lfcc \
        --dataset asvspoof \
        --data_path "data/ASVspoof 2019 Dataset 2/LA/LA" \
        --lfcc_path outputs/lfcc_model.pth

    # Evaluate ensemble:
    python ensemble.py --mode eval_ensemble \
        --rawnet_path outputs/best_model.pth \
        --lfcc_path outputs/lfcc_model.pth \
        --audio path/to/sample.wav
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import librosa
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from asvspoof_dataset import Dataset_ASVspoof2019
from main import Dataset_LibriSeVoc
from model import RawNet

SAMPLE_RATE = 24000
MAX_LEN     = 96000
N_LFCC      = 60
N_FFT       = 512
HOP_LENGTH  = 160
LFCC_T_TARGET = 300
BENIGN_MISSING_SINC = {'Sinc_conv.low_hz_', 'Sinc_conv.band_hz_', 'Sinc_conv.window_', 'Sinc_conv.n_'}


# ── LFCC Feature Extraction ───────────────────────────────────────────────────

def extract_lfcc(waveform_np, sr=SAMPLE_RATE, n_lfcc=N_LFCC):
    """
    Extract Linear Frequency Cepstral Coefficients (LFCC).
    Unlike MFCC (Mel scale), LFCC uses a linear filter bank —
    better at capturing vocoder artifacts at high frequencies.
    """
    # Linear filter bank
    n_filters = n_lfcc * 2
    fft_freqs = np.linspace(0, sr / 2, N_FFT // 2 + 1)
    linear_freqs = np.linspace(0, sr / 2, n_filters + 2)

    # Build filter bank matrix
    fb = np.zeros((n_filters, N_FFT // 2 + 1))
    for i in range(n_filters):
        f_left   = linear_freqs[i]
        f_center = linear_freqs[i + 1]
        f_right  = linear_freqs[i + 2]
        for j, f in enumerate(fft_freqs):
            if f_left <= f <= f_center:
                fb[i, j] = (f - f_left) / (f_center - f_left + 1e-9)
            elif f_center < f <= f_right:
                fb[i, j] = (f_right - f) / (f_right - f_center + 1e-9)

    # STFT
    stft = np.abs(librosa.stft(waveform_np, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2

    # Apply filter bank
    filtered = np.dot(fb, stft)  # (n_filters, T)
    log_filtered = np.log(filtered + 1e-9)

    # DCT to get cepstral coefficients
    from scipy.fft import dct
    lfcc = dct(log_filtered, axis=0, norm='ortho')[:n_lfcc]  # (n_lfcc, T)

    # Delta and delta-delta
    delta  = librosa.feature.delta(lfcc)
    delta2 = librosa.feature.delta(lfcc, order=2)

    features = np.concatenate([lfcc, delta, delta2], axis=0)  # (3*n_lfcc, T)
    return features.astype(np.float32)


# ── LCNN Architecture ─────────────────────────────────────────────────────────

class MaxFeatureMap(nn.Module):
    """Max Feature Map activation — halves channels, takes element-wise max."""
    def forward(self, x):
        assert x.size(1) % 2 == 0
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


class LCNN(nn.Module):
    """
    Light CNN (LCNN) for LFCC-based deepfake detection.
    Uses Max Feature Map activations for compact representation.
    Input: (batch, 1, n_lfcc*3, T)
    Output: (batch, 2) log-softmax scores
    """

    def __init__(self, input_channels=1, n_lfcc=N_LFCC):
        super().__init__()
        in_feat = n_lfcc * 3  # LFCC + delta + delta2

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            MaxFeatureMap(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 64, kernel_size=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 96, kernel_size=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 256),
            MaxFeatureMap(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return self.logsoftmax(x)


# ── LFCC Dataset + Training ────────────────────────────────────────────────────

def pad_lfcc_time(lfcc, t_target=LFCC_T_TARGET):
    if lfcc.shape[1] >= t_target:
        return lfcc[:, :t_target]
    pad = t_target - lfcc.shape[1]
    return np.pad(lfcc, ((0, 0), (0, pad)), mode='wrap')


class LFCCDataset(Dataset):
    """Wrap waveform datasets and expose LFCC tensors for LCNN training."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            x, _, y = sample
        else:
            x, y = sample

        waveform_np = x.numpy() if isinstance(x, torch.Tensor) else np.array(x)
        lfcc = extract_lfcc(waveform_np)
        lfcc = pad_lfcc_time(lfcc)

        x_lfcc = torch.tensor(lfcc, dtype=torch.float32).unsqueeze(0)  # (1, 3*n_lfcc, T)
        y_bin = torch.tensor(int(y), dtype=torch.int64)
        return x_lfcc, y_bin


def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def build_base_dataset(dataset_name, data_path, split):
    if dataset_name == "asvspoof":
        return Dataset_ASVspoof2019(data_path, split=split, resample_to=SAMPLE_RATE)
    if dataset_name == "librisevoc":
        return Dataset_LibriSeVoc(data_path, split=split)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def maybe_subset(ds, max_samples):
    if max_samples is None:
        return ds
    n = min(max_samples, len(ds))
    return Subset(ds, list(range(n)))


def train_lfcc_model(args):
    if not args.data_path:
        raise SystemExit("--data_path is required for --mode train_lfcc")

    if not os.path.isdir(args.data_path):
        raise SystemExit(f"Dataset path not found: {args.data_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Building LFCC datasets from {args.dataset}...")

    train_base = build_base_dataset(args.dataset, args.data_path, args.train_split)
    dev_base = build_base_dataset(args.dataset, args.data_path, args.dev_split)
    train_base = maybe_subset(train_base, args.max_train_samples)
    dev_base = maybe_subset(dev_base, args.max_dev_samples)

    train_ds = LFCCDataset(train_base)
    dev_ds = LFCCDataset(dev_base)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = LCNN(n_lfcc=N_LFCC).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.NLLLoss()

    best_eer = float("inf")
    out_path = args.lfcc_path
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"\n{'='*55}")
    print("  Training LFCC-LCNN")
    print(f"{'='*55}")
    print(
        f"  Train={len(train_ds)} Dev={len(dev_ds)} "
        f"Epochs={args.epochs} Batch={args.batch_size} LR={args.lr}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for x, y in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * x.size(0)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total += x.size(0)

        model.eval()
        dv_loss = 0.0
        dv_correct = 0
        dv_total = 0
        y_true = []
        y_score = []
        with torch.no_grad():
            for x, y in dev_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                loss = criterion(out, y)
                dv_loss += loss.item() * x.size(0)
                dv_correct += (out.argmax(1) == y).sum().item()
                dv_total += x.size(0)
                probs = torch.exp(out)[:, 1]
                y_true.extend(y.cpu().numpy().tolist())
                y_score.extend(probs.cpu().numpy().tolist())

        tr_loss /= max(tr_total, 1)
        dv_loss /= max(dv_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)
        dv_acc = dv_correct / max(dv_total, 1)
        dv_eer = compute_eer(np.array(y_true), np.array(y_score))

        marker = ""
        if np.isfinite(dv_eer) and dv_eer < best_eer:
            best_eer = dv_eer
            torch.save(model.state_dict(), out_path)
            marker = " ✓"

        print(
            f"Epoch {epoch:02d} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
            f"dv_loss={dv_loss:.4f} dv_acc={dv_acc:.4f} dv_eer={dv_eer:.4f}{marker}"
        )

    if not os.path.exists(out_path):
        # Fallback save when EER is NaN (e.g., subset with one class only).
        torch.save(model.state_dict(), out_path)
    print(f"LFCC model saved: {out_path}")


# ── Ensemble ──────────────────────────────────────────────────────────────────

class TrustCallEnsemble:
    """
    Ensemble of RawNet (waveform-based) + LCNN (LFCC-based).
    Combines predictions via weighted averaging of probabilities.
    """

    def __init__(self, rawnet_model, lcnn_model, device,
                 rawnet_weight=0.6, lcnn_weight=0.4):
        self.rawnet = rawnet_model
        self.lcnn   = lcnn_model
        self.device = device
        self.w_raw  = rawnet_weight
        self.w_lcnn = lcnn_weight

        self.rawnet.eval()
        self.lcnn.eval()

    @torch.no_grad()
    def predict(self, waveform_np):
        """
        Args:
            waveform_np: numpy array of shape (T,) at SAMPLE_RATE
        Returns:
            prob_fake: float in [0, 1]
            details: dict with individual model scores
        """
        # RawNet prediction
        x_raw = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        out_binary, _ = self.rawnet(x_raw)
        prob_fake_raw = torch.exp(out_binary[0, 1]).item()

        # LFCC-LCNN prediction
        lfcc = extract_lfcc(waveform_np)
        lfcc = pad_lfcc_time(lfcc)

        x_lfcc = torch.tensor(lfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        out_lcnn = self.lcnn(x_lfcc)
        prob_fake_lcnn = torch.exp(out_lcnn[0, 1]).item()

        # Weighted ensemble
        prob_fake_ensemble = self.w_raw * prob_fake_raw + self.w_lcnn * prob_fake_lcnn

        return prob_fake_ensemble, {
            'rawnet_score': round(prob_fake_raw, 4),
            'lcnn_score':   round(prob_fake_lcnn, 4),
            'ensemble_score': round(prob_fake_ensemble, 4),
            'weights': {'rawnet': self.w_raw, 'lcnn': self.w_lcnn},
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def pad_audio(x, max_len=MAX_LEN):
    if len(x) >= max_len:
        return x[:max_len]
    return np.tile(x, int(max_len / len(x)) + 1)[:max_len]


def main():
    parser = argparse.ArgumentParser(description='TrustCall Ensemble')
    parser.add_argument('--mode', choices=['eval_ensemble', 'show_lfcc', 'train_lfcc'],
                        default='eval_ensemble')
    parser.add_argument('--data_path', default=None, help='Dataset root (required for train_lfcc)')
    parser.add_argument('--dataset', choices=['asvspoof', 'librisevoc'], default='asvspoof')
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--dev_split', default='dev')
    parser.add_argument('--rawnet_path', default='outputs/best_model.pth')
    parser.add_argument('--lfcc_path',   default='outputs/lfcc_model.pth')
    parser.add_argument('--config',      default='model_config_RawNet.yaml')
    parser.add_argument('--audio',       default=None)
    parser.add_argument('--epochs',      type=int, default=5)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_dev_samples',   type=int, default=None)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--rawnet_weight', type=float, default=0.6)
    parser.add_argument('--lcnn_weight',   type=float, default=0.4)
    args = parser.parse_args()

    if args.mode == 'train_lfcc':
        train_lfcc_model(args)
        return

    if not args.audio:
        raise SystemExit('--audio is required for --mode eval_ensemble or --mode show_lfcc')

    device = torch.device('cpu')

    # Load audio
    y, sr = librosa.load(args.audio, sr=None, mono=True)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    y = pad_audio(y)

    if args.mode == 'show_lfcc':
        print("Extracting LFCC features...")
        lfcc = extract_lfcc(y)
        print(f"LFCC shape: {lfcc.shape}  (features x time frames)")
        return

    # Load RawNet
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    rawnet = RawNet(config['model'], device)
    if os.path.exists(args.rawnet_path):
        state_dict = torch.load(args.rawnet_path, map_location=device)
        missing, unexpected = rawnet.load_state_dict(state_dict, strict=False)
        print(f"  RawNet loaded: {args.rawnet_path}")
        non_benign_missing = [k for k in missing if k not in BENIGN_MISSING_SINC]
        if non_benign_missing or unexpected:
            print(f"  RawNet partial load: missing={len(non_benign_missing)} unexpected={len(unexpected)}")
    rawnet.to(device)

    # Load / init LCNN
    lcnn = LCNN(n_lfcc=N_LFCC)
    if os.path.exists(args.lfcc_path):
        lcnn.load_state_dict(torch.load(args.lfcc_path, map_location=device), strict=False)
        print(f"  LCNN loaded: {args.lfcc_path}")
    else:
        print(f"  LCNN: no checkpoint found, using random weights")
    lcnn.to(device)

    # Run ensemble
    ensemble = TrustCallEnsemble(rawnet, lcnn, device,
                                  rawnet_weight=args.rawnet_weight,
                                  lcnn_weight=args.lcnn_weight)
    prob_fake, details = ensemble.predict(y)

    print(f"\n{'='*45}")
    print(f"  TrustCall Ensemble Result")
    print(f"{'='*45}")
    print(f"  RawNet score  : {details['rawnet_score']:.4f}")
    print(f"  LCNN score    : {details['lcnn_score']:.4f}")
    print(f"  Ensemble score: {details['ensemble_score']:.4f}")
    print(f"  Verdict       : {'🚨 DEEPFAKE' if prob_fake > 0.5 else '✅ GENUINE'}")
    print(json.dumps(details, indent=2))


if __name__ == '__main__':
    main()
