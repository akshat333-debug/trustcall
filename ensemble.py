"""
Extension 9: LFCC-LCNN Model + Ensemble with RawNet
Implements a lightweight LCNN classifier on LFCC features,
then ensembles its predictions with RawNet for improved accuracy.

Usage:
    # Train LFCC-LCNN:
    python ensemble.py --mode train_lfcc \
        --data_path /path/to/data --model_save_path outputs/lfcc_model.pth

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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import librosa

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model import RawNet

SAMPLE_RATE = 24000
MAX_LEN     = 96000
N_LFCC      = 60
N_FFT       = 512
HOP_LENGTH  = 160


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
        # Pad/trim time dimension to fixed length
        T_target = 300
        if lfcc.shape[1] >= T_target:
            lfcc = lfcc[:, :T_target]
        else:
            pad = T_target - lfcc.shape[1]
            lfcc = np.pad(lfcc, ((0, 0), (0, pad)), mode='wrap')

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
    parser.add_argument('--mode', choices=['eval_ensemble', 'show_lfcc'],
                        default='eval_ensemble')
    parser.add_argument('--rawnet_path', default='outputs/best_model.pth')
    parser.add_argument('--lfcc_path',   default='outputs/lfcc_model.pth')
    parser.add_argument('--config',      default='model_config_RawNet.yaml')
    parser.add_argument('--audio',       required=True)
    parser.add_argument('--rawnet_weight', type=float, default=0.6)
    parser.add_argument('--lcnn_weight',   type=float, default=0.4)
    args = parser.parse_args()

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
        rawnet.load_state_dict(torch.load(args.rawnet_path, map_location=device))
        print(f"  RawNet loaded: {args.rawnet_path}")
    rawnet.to(device)

    # Load / init LCNN
    lcnn = LCNN(n_lfcc=N_LFCC)
    if os.path.exists(args.lfcc_path):
        lcnn.load_state_dict(torch.load(args.lfcc_path, map_location=device))
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
