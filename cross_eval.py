"""
Extension 7: Cross-Dataset Evaluation
Trains on one dataset (LibriSeVoc or ASVspoof) and evaluates on the other.
Measures generalization gap — a key research contribution.

Usage:
    python cross_eval.py \
        --model_path outputs/best_model.pth \
        --config model_config_RawNet.yaml \
        --librisevoc_path /path/to/LibriSeVoc \
        --asvspoof_path /path/to/ASVspoof2019 \
        --out_dir outputs/cross_eval
"""

import argparse
import json
import os
import sys
import tempfile
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model import RawNet
from asvspoof_dataset import Dataset_ASVspoof2019
from main import Dataset_LibriSeVoc

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "trustcall-cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "trustcall-mpl"))
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

SAMPLE_RATE = 24000
MAX_LEN     = 96000
BENIGN_MISSING_SINC = {'Sinc_conv.low_hz_', 'Sinc_conv.band_hz_', 'Sinc_conv.window_', 'Sinc_conv.n_'}


def pad_audio(x, max_len=MAX_LEN):
    if len(x) >= max_len:
        return x[:max_len]
    return np.tile(x, int(max_len / len(x)) + 1)[:max_len]


def compute_eer(labels, scores):
    """Compute Equal Error Rate."""
    from sklearn.metrics import roc_curve
    if len(np.unique(labels)) < 2:
        return float("nan"), 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    diff = np.abs(fnr - fpr)
    if np.all(np.isnan(diff)):
        return float("nan"), 0.5
    idx = np.nanargmin(diff)
    return (fpr[idx] + fnr[idx]) / 2, thresholds[idx]


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Run model on a DataLoader, return labels and fake scores."""
    all_labels, all_scores = [], []
    model.eval()
    for batch in tqdm(loader, desc='  Evaluating', leave=False):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, _, y = batch
        else:
            x, y = batch
        x = x.to(device)
        y = y.to(device)
        out_binary, _ = model(x)
        probs = torch.exp(out_binary)
        all_scores.extend(probs[:, 1].cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_scores)


def run_cross_eval(model, device, datasets: dict, out_dir: str):
    """
    Evaluate model on all provided datasets.
    datasets: {'name': DataLoader, ...}
    Returns results dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for name, loader in datasets.items():
        print(f"\n  Evaluating on: {name}")
        labels, scores = evaluate_loader(model, loader, device)
        eer, eer_thresh = compute_eer(labels, scores)

        from sklearn.metrics import roc_auc_score, accuracy_score
        threshold = eer_thresh if np.isfinite(eer_thresh) else 0.5
        preds = (scores >= threshold).astype(int)
        acc   = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")

        results[name] = {
            'eer':       round(float(eer), 4) if np.isfinite(eer) else None,
            'auc':       round(float(auc), 4) if np.isfinite(auc) else None,
            'accuracy':  round(float(acc), 4),
            'n_samples': int(len(labels)),
            'n_real':    int((labels == 0).sum()),
            'n_fake':    int((labels == 1).sum()),
        }
        print(f"    EER={eer:.4f}  AUC={auc:.4f}  Acc={acc:.4f}  N={len(labels)}")

    # Save JSON
    out_json = os.path.join(out_dir, 'cross_eval_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_json}")

    # Plot comparison bar chart
    if HAS_MPL and results:
        names = list(results.keys())
        eers  = [((results[n]['eer'] if results[n]['eer'] is not None else np.nan) * 100) for n in names]
        aucs  = [(results[n]['auc'] if results[n]['auc'] is not None else np.nan) for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Cross-Dataset Generalization', fontsize=14, fontweight='bold')

        colors = ['#4c8bf5', '#ff4b4b', '#21c354', '#ffa500'][:len(names)]
        ax1.bar(names, eers, color=colors, alpha=0.8)
        ax1.set_ylabel('EER (%)', fontsize=11)
        ax1.set_title('Equal Error Rate (lower = better)', fontsize=11)
        for i, v in enumerate(eers):
            label = "NA" if np.isnan(v) else f'{v:.2f}%'
            y_text = 0.2 if np.isnan(v) else v + 0.2
            ax1.text(i, y_text, label, ha='center', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        ax2.bar(names, aucs, color=colors, alpha=0.8)
        ax2.set_ylabel('AUC-ROC', fontsize=11)
        ax2.set_title('AUC-ROC (higher = better)', fontsize=11)
        ax2.set_ylim(0, 1.05)
        for i, v in enumerate(aucs):
            label = "NA" if np.isnan(v) else f'{v:.3f}'
            y_text = 0.01 if np.isnan(v) else v + 0.01
            ax2.text(i, y_text, label, ha='center', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(out_dir, 'cross_eval_chart.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Chart saved: {plot_path}")

    return results


def maybe_subset(ds, max_samples):
    if max_samples is None:
        return ds
    n = min(max_samples, len(ds))
    return Subset(ds, list(range(n)))


def main():
    parser = argparse.ArgumentParser(description='TrustCall Cross-Dataset Evaluation')
    parser.add_argument('--model_path',      default='outputs/best_model.pth')
    parser.add_argument('--config',          default='model_config_RawNet.yaml')
    parser.add_argument('--asvspoof_path',   default=None, help='Path to ASVspoof 2019 LA')
    parser.add_argument('--librisevoc_path', default=None, help='Path to LibriSeVoc')
    parser.add_argument('--batch_size',      type=int, default=16)
    parser.add_argument('--num_workers',     type=int, default=0)
    parser.add_argument('--max_samples_per_set', type=int, default=None,
                        help='Optional cap per evaluation split for quick smoke runs')
    parser.add_argument('--out_dir',         default='outputs/cross_eval')
    args = parser.parse_args()

    device = torch.device('cpu')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = RawNet(config['model'], device)
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded: {args.model_path}")
        non_benign_missing = [k for k in missing if k not in BENIGN_MISSING_SINC]
        if non_benign_missing or unexpected:
            print(f"  Partial load: missing={len(non_benign_missing)} unexpected={len(unexpected)}")
    else:
        print(f"  Warning: no checkpoint, using random weights")
    model.to(device)

    datasets = {}

    if args.asvspoof_path and os.path.isdir(args.asvspoof_path):
        for split in ['dev', 'eval']:
            try:
                ds = Dataset_ASVspoof2019(args.asvspoof_path, split=split,
                                          resample_to=SAMPLE_RATE)
                ds = maybe_subset(ds, args.max_samples_per_set)
                datasets[f'ASVspoof2019_{split}'] = DataLoader(
                    ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            except Exception as e:
                print(f"  Skipping ASVspoof {split}: {e}")

    if args.librisevoc_path and os.path.isdir(args.librisevoc_path):
        try:
            ds = Dataset_LibriSeVoc(args.librisevoc_path, split='test')
            ds = maybe_subset(ds, args.max_samples_per_set)
            datasets['LibriSeVoc_test'] = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )
        except Exception as e:
            print(f"  Skipping LibriSeVoc test: {e}")

    if not datasets:
        print("  No datasets found. Provide --asvspoof_path and/or --librisevoc_path")
        return

    run_cross_eval(model, device, datasets, args.out_dir)


if __name__ == '__main__':
    main()
