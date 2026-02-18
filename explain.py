"""
Extension 6: SincConv Filter Visualization & Gradient-based Saliency
Visualizes what frequency bands the model focuses on.

Usage:
    python explain.py --model_path outputs/best_model.pth --audio path/to/sample.wav
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from model import RawNet

SAMPLE_RATE = 24000
MAX_LEN     = 96000


def pad_audio(x, max_len=MAX_LEN):
    if len(x) >= max_len:
        return x[:max_len]
    return np.tile(x, int(max_len / len(x)) + 1)[:max_len]


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return pad_audio(y)


def plot_sinc_filters(model, out_path, n_filters=20):
    """
    Visualize the learned SincConv bandpass filters in frequency domain.
    Shows which frequency bands the first layer is sensitive to.
    """
    sinc = model.Sinc_conv
    mel_freqs = sinc.mel  # shape: (n_filters + 1,)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('SincConv Learned Bandpass Filters', fontsize=14, fontweight='bold')

    # Plot 1: Filter bank frequency response
    ax = axes[0]
    n_show = min(n_filters, len(mel_freqs) - 1)
    cmap = cm.get_cmap('plasma', n_show)

    for i in range(n_show):
        fmin = mel_freqs[i]
        fmax = mel_freqs[i + 1]
        freqs = np.linspace(0, SAMPLE_RATE // 2, 1000)
        # Ideal bandpass response
        response = ((freqs >= fmin) & (freqs <= fmax)).astype(float)
        ax.plot(freqs, response + i * 0.05, color=cmap(i), alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Filter Index (stacked)', fontsize=11)
    ax.set_title('Mel-spaced Bandpass Filter Bank', fontsize=12)
    ax.set_xlim(0, SAMPLE_RATE // 2)
    ax.grid(True, alpha=0.3)

    # Plot 2: Center frequencies and bandwidths
    ax2 = axes[1]
    centers = [(mel_freqs[i] + mel_freqs[i+1]) / 2 for i in range(len(mel_freqs)-1)]
    bandwidths = [mel_freqs[i+1] - mel_freqs[i] for i in range(len(mel_freqs)-1)]
    ax2.bar(range(len(centers)), centers, yerr=bandwidths,
            color='#4c8bf5', alpha=0.7, capsize=2)
    ax2.set_xlabel('Filter Index', fontsize=11)
    ax2.set_ylabel('Center Frequency (Hz)', fontsize=11)
    ax2.set_title('Filter Center Frequencies with Bandwidth', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def compute_input_gradient_saliency(model, device, audio_np):
    """
    Compute gradient of output w.r.t. input waveform.
    High gradient = model is sensitive to that part of the audio.
    """
    x = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(device)
    x.requires_grad_(True)

    out_binary, _ = model(x)
    # Gradient w.r.t. fake class score
    fake_score = out_binary[0, 1]
    model.zero_grad()
    fake_score.backward()

    saliency = x.grad.data.abs().squeeze().cpu().numpy()
    return saliency


def plot_saliency(audio_np, saliency, out_path):
    """Plot waveform with saliency overlay and mel spectrogram."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Input Saliency Analysis', fontsize=14, fontweight='bold')
    t = np.linspace(0, len(audio_np) / SAMPLE_RATE, len(audio_np))

    # Waveform
    ax = axes[0]
    ax.plot(t, audio_np, color='#4c8bf5', linewidth=0.5, alpha=0.8)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title('Input Waveform', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Saliency
    ax = axes[1]
    # Smooth saliency for readability
    from numpy.lib.stride_tricks import sliding_window_view
    window = 512
    if len(saliency) > window:
        smoothed = np.array([saliency[max(0,i-window//2):i+window//2].mean()
                             for i in range(len(saliency))])
    else:
        smoothed = saliency
    ax.fill_between(t, smoothed, alpha=0.7, color='#ff4b4b')
    ax.set_ylabel('|Gradient|', fontsize=10)
    ax.set_title('Input Saliency (where model focuses)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mel spectrogram
    ax = axes[2]
    mel = librosa.feature.melspectrogram(y=audio_np, sr=SAMPLE_RATE, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=SAMPLE_RATE, hop_length=256,
                                   x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title('Mel Spectrogram', fontsize=11)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='TrustCall Explainability')
    parser.add_argument('--model_path', default='outputs/best_model.pth')
    parser.add_argument('--config',     default='model_config_RawNet.yaml')
    parser.add_argument('--audio',      required=True, help='Path to audio file')
    parser.add_argument('--out_dir',    default='outputs/explanations')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cpu')

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = RawNet(config['model'], device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"  Loaded: {args.model_path}")
    model.eval()

    print("\n[1/2] Visualizing SincConv filter bank...")
    plot_sinc_filters(model,
                      os.path.join(args.out_dir, 'sinc_filters.png'))

    print("[2/2] Computing input gradient saliency...")
    audio = load_audio(args.audio)
    saliency = compute_input_gradient_saliency(model, device, audio)
    plot_saliency(audio, saliency,
                  os.path.join(args.out_dir, 'saliency.png'))

    print(f"\n  Done. Outputs in: {args.out_dir}/")


if __name__ == '__main__':
    main()
