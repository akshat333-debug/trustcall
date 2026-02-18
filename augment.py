"""
Extension 8: Data Augmentation for Robust Training
Adds noise, room reverb, codec compression, and pitch shift augmentations.
Integrates as a drop-in wrapper around any existing Dataset.

Usage:
    from augment import AugmentedDataset
    base_dataset = Dataset_LibriSeVoc(path, split='train')
    aug_dataset  = AugmentedDataset(base_dataset, p=0.5)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import random


# ── Individual Augmentations ──────────────────────────────────────────────────

def add_gaussian_noise(x, snr_db_range=(10, 40)):
    """Add white Gaussian noise at a random SNR."""
    snr_db = random.uniform(*snr_db_range)
    signal_power = np.mean(x ** 2) + 1e-9
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(x)) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


def apply_room_reverb(x, sample_rate=24000, room_scale_range=(0.1, 0.9)):
    """
    Simulate room reverb using a simple exponential decay impulse response.
    Avoids heavy dependency on pyroomacoustics.
    """
    room_scale = random.uniform(*room_scale_range)
    decay_rate = 5.0 / (room_scale * sample_rate)
    ir_len = int(room_scale * sample_rate * 0.5)
    t = np.arange(ir_len)
    ir = np.exp(-decay_rate * t) * np.random.randn(ir_len)
    ir = ir / (np.sum(np.abs(ir)) + 1e-9)
    reverbed = np.convolve(x, ir, mode='full')[:len(x)]
    return reverbed.astype(np.float32)


def apply_codec_compression(x, sample_rate=24000, bitrate_kbps_range=(8, 32)):
    """
    Simulate codec compression artifacts by quantizing the signal.
    Approximates the distortion of low-bitrate codecs (MP3, AAC, etc.)
    without requiring ffmpeg.
    """
    bits = random.randint(8, 12)  # simulate low-bit quantization
    scale = 2 ** bits
    quantized = np.round(x * scale) / scale
    return quantized.astype(np.float32)


def apply_pitch_shift(x, sample_rate=24000, semitones_range=(-2, 2)):
    """
    Approximate pitch shift via resampling (no librosa dependency at runtime).
    Shifts pitch by resampling then trimming/padding.
    """
    try:
        import librosa
        semitones = random.uniform(*semitones_range)
        shifted = librosa.effects.pitch_shift(x, sr=sample_rate, n_steps=semitones)
        return shifted.astype(np.float32)
    except Exception:
        return x  # fallback: no-op


def apply_time_stretch(x, rate_range=(0.9, 1.1)):
    """
    Approximate time stretch via linear interpolation.
    """
    rate = random.uniform(*rate_range)
    orig_len = len(x)
    new_len = int(orig_len / rate)
    indices = np.linspace(0, orig_len - 1, new_len)
    stretched = np.interp(indices, np.arange(orig_len), x)
    # Trim or pad back to original length
    if len(stretched) >= orig_len:
        return stretched[:orig_len].astype(np.float32)
    pad = orig_len - len(stretched)
    return np.pad(stretched, (0, pad), mode='wrap').astype(np.float32)


def apply_volume_perturbation(x, gain_range=(0.5, 1.5)):
    """Random volume scaling."""
    gain = random.uniform(*gain_range)
    return (x * gain).astype(np.float32)


# ── Augmentation Pipeline ─────────────────────────────────────────────────────

AUGMENTATIONS = [
    ('gaussian_noise',    add_gaussian_noise),
    ('room_reverb',       apply_room_reverb),
    ('codec_compression', apply_codec_compression),
    ('pitch_shift',       apply_pitch_shift),
    ('time_stretch',      apply_time_stretch),
    ('volume_perturb',    apply_volume_perturbation),
]


class AugmentedDataset(Dataset):
    """
    Wraps any dataset that returns (waveform_tensor, label) and applies
    random augmentations with probability p.

    Args:
        base_dataset:  Dataset returning (Tensor[T], int)
        p:             Probability of applying each augmentation
        sample_rate:   Audio sample rate (for time-aware augmentations)
        aug_names:     List of augmentation names to use (default: all)
    """

    def __init__(self, base_dataset, p=0.5, sample_rate=24000, aug_names=None):
        self.base_dataset = base_dataset
        self.p = p
        self.sample_rate = sample_rate

        if aug_names is None:
            self.augs = AUGMENTATIONS
        else:
            self.augs = [(n, fn) for n, fn in AUGMENTATIONS if n in aug_names]

        print(f"[AugmentedDataset] {len(base_dataset)} samples | "
              f"p={p} | augmentations: {[n for n, _ in self.augs]}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, label = self.base_dataset[idx]

        # Convert to numpy for augmentation
        x_np = x.numpy() if isinstance(x, torch.Tensor) else np.array(x)

        for name, aug_fn in self.augs:
            if random.random() < self.p:
                try:
                    x_np = aug_fn(x_np, self.sample_rate) \
                        if name in ('room_reverb', 'pitch_shift') \
                        else aug_fn(x_np)
                except Exception:
                    pass  # skip failed augmentations silently

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(x_np)) + 1e-9
        x_np = x_np / max_val

        return torch.tensor(x_np, dtype=torch.float32), label


# ── Demo / Test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import librosa

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True, help='Test audio file')
    parser.add_argument('--out_dir', default='outputs/augmentation_demo')
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    y, sr = librosa.load(args.audio, sr=24000, mono=True)
    y = y[:96000]  # 4 seconds

    print(f"Testing augmentations on: {args.audio}")
    for name, fn in AUGMENTATIONS:
        try:
            if name in ('room_reverb', 'pitch_shift'):
                aug = fn(y, sr)
            else:
                aug = fn(y)
            print(f"  ✓ {name}: shape={aug.shape}, max={aug.max():.3f}")
            import soundfile as sf
            sf.write(os.path.join(args.out_dir, f'{name}.wav'), aug, sr)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
