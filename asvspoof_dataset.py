"""
Extension 4: ASVspoof 2019 LA Dataset Loader
Adds support for the ASVspoof 2019 Logical Access dataset alongside LibriSeVoc.

Usage (in main.py):
    from asvspoof_dataset import Dataset_ASVspoof2019
    dataset = Dataset_ASVspoof2019(dataset_path='/path/to/ASVspoof2019', split='train')
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

SAMPLE_RATE = 16000   # ASVspoof uses 16kHz (vs LibriSeVoc's 24kHz)
MAX_LEN = 64000       # 4 seconds at 16kHz


def pad_or_trim(x, max_len=MAX_LEN):
    """Pad short audio with repetition, trim long audio from center."""
    if len(x) >= max_len:
        start = (len(x) - max_len) // 2
        return x[start:start + max_len]
    repeats = int(max_len / len(x)) + 1
    return np.tile(x, repeats)[:max_len]


class Dataset_ASVspoof2019(Dataset):
    """
    ASVspoof 2019 Logical Access dataset loader.

    Directory structure expected:
        dataset_path/
            ASVspoof2019_LA_train/
                flac/  (or wav/)
            ASVspoof2019_LA_dev/
                flac/
            ASVspoof2019_LA_eval/
                flac/
            ASVspoof2019_LA_cm_protocols/
                ASVspoof2019.LA.cm.train.trn.txt
                ASVspoof2019.LA.cm.dev.trl.txt
                ASVspoof2019.LA.cm.eval.trl.txt

    Labels: 0 = bonafide (real), 1 = spoof (fake)
    """

    SPLIT_MAP = {
        'train': ('ASVspoof2019_LA_train', 'ASVspoof2019.LA.cm.train.trn.txt'),
        'dev':   ('ASVspoof2019_LA_dev',   'ASVspoof2019.LA.cm.dev.trl.txt'),
        'eval':  ('ASVspoof2019_LA_eval',  'ASVspoof2019.LA.cm.eval.trl.txt'),
    }

    def __init__(self, dataset_path, split='train', resample_to=None, max_len=MAX_LEN):
        assert split in self.SPLIT_MAP, f"split must be one of {list(self.SPLIT_MAP.keys())}"
        self.max_len = max_len
        self.resample_to = resample_to  # if set, resample to this rate

        audio_dir_name, protocol_name = self.SPLIT_MAP[split]
        audio_dir = os.path.join(dataset_path, audio_dir_name, 'flac')
        if not os.path.isdir(audio_dir):
            audio_dir = os.path.join(dataset_path, audio_dir_name, 'wav')

        protocol_path = os.path.join(
            dataset_path, 'ASVspoof2019_LA_cm_protocols', protocol_name
        )

        self.samples = []
        self.labels  = []

        if not os.path.exists(protocol_path):
            raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # Format: SPEAKER_ID UTTERANCE_ID - SYSTEM_ID LABEL
                utt_id = parts[1]
                label_str = parts[4]  # 'bonafide' or 'spoof'
                label = 0 if label_str == 'bonafide' else 1

                # Try .flac first, then .wav
                for ext in ['.flac', '.wav']:
                    fpath = os.path.join(audio_dir, utt_id + ext)
                    if os.path.exists(fpath):
                        self.samples.append(fpath)
                        self.labels.append(label)
                        break

        print(f"[ASVspoof2019] {split}: {len(self.samples)} samples "
              f"({self.labels.count(0)} real, {self.labels.count(1)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath = self.samples[idx]
        label = self.labels[idx]

        y, sr = librosa.load(fpath, sr=None, mono=True)

        # Optionally resample (e.g. to 24kHz to match LibriSeVoc)
        target_sr = self.resample_to or sr
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        y = pad_or_trim(y, self.max_len)
        return torch.tensor(y, dtype=torch.float32), label


def get_asvspoof_loaders(dataset_path, batch_size=32, num_workers=4, resample_to=None):
    """Convenience function: returns train, dev, eval DataLoaders."""
    from torch.utils.data import DataLoader

    loaders = {}
    for split in ['train', 'dev', 'eval']:
        try:
            ds = Dataset_ASVspoof2019(dataset_path, split=split, resample_to=resample_to)
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, shuffle=(split == 'train'),
                num_workers=num_workers, pin_memory=True
            )
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
    return loaders


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--split', default='dev')
    args = parser.parse_args()

    ds = Dataset_ASVspoof2019(args.data_path, split=args.split)
    x, y = ds[0]
    print(f"Sample shape: {x.shape}, label: {y} ({'real' if y==0 else 'fake'})")
