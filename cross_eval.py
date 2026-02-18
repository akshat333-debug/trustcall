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
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model import RawNet
from asvspoof_dataset import Dataset_ASVspoof2019

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

SAMPLE_RATE = 24000
MAX_LEN     = 96000


def pad_audio(x, max_len=MAX_LEN):
    if len(x) >= max_len:
        return x[:max_len]
    return np.tile(x, int(max_len / len(x)) + 1)[:max_len]


def compute_eer(labels, scores):
    """Compute Equal Error Rate."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2, thresholds[idx]


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Run model on a DataLoader, return labels and fake scores."""
    all_labels, all_scores = [], []
    model.eval()
    for batch in tqdm(loader, desc='  Evaluating', leave=False):
        x, y = batch
        x = x.to(device)
        out_binary, _ = model(x)
        probs = torch.exp(out_binary)
        all_scores.extend(probs[:, 1].cpu().numpy().tolist())
        all_labels.extend(y.numpy().tolist())
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
        preds = (scores >= eer_thresh).astype(int)
        acc   = accuracy_score(labels, preds)
        auc   = roc_auc_score(labels, scores)

        results[name] = {
            'eer':       round(float(eer), 4),
            'auc':       round(float(auc), 4),
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
        eers  = [results[n]['eer'] * 100 for n in names]
        aucs  = [results[n]['auc'] for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Cross-Dataset Generalization', fontsize=14, fontweight='bold')

        colors = ['#4c8bf5', '#ff4b4b', '#21c354', '#ffa500'][:len(names)]
        ax1.bar(names, eers, color=colors, alpha=0.8)
        ax1.set_ylabel('EER (%)', fontsize=11)
        ax1.set_title('Equal Error Rate (lower = better)', fontsize=11)
        for i, v in enumerate(eers):
            ax1.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        ax2.bar(names, aucs, color=colors, alpha=0.8)
        ax2.set_ylabel('AUC-ROC', fontsize=11)
        ax2.set_title('AUC-ROC (higher = better)', fontsize=11)
        ax2.set_ylim(0, 1.05)
        for i, v in enumerate(aucs):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(out_dir, 'cross_eval_chart.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Chart saved: {plot_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='TrustCall Cross-Dataset Evaluation')
    parser.add_argument('--model_path',      default='outputs/best_model.pth')
    parser.add_argument('--config',          default='model_config_RawNet.yaml')
    parser.add_argument('--asvspoof_path',   default=None, help='Path to ASVspoof 2019 LA')
    parser.add_argument('--librisevoc_path', default=None, help='Path to LibriSeVoc')
    parser.add_argument('--batch_size',      type=int, default=16)
    parser.add_argument('--out_dir',         default='outputs/cross_eval')
    args = parser.parse_args()

    device = torch.device('cpu')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = RawNet(config['model'], device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"  Loaded: {args.model_path}")
    else:
        print(f"  Warning: no checkpoint, using random weights")
    model.to(device)

    datasets = {}

    if args.asvspoof_path and os.path.isdir(args.asvspoof_path):
        for split in ['dev', 'eval']:
            try:
                ds = Dataset_ASVspoof2019(args.asvspoof_path, split=split,
                                          resample_to=SAMPLE_RATE)
                datasets[f'ASVspoof2019_{split}'] = DataLoader(
                    ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
            except Exception as e:
                print(f"  Skipping ASVspoof {split}: {e}")

    if not datasets:
        print("  No datasets found. Provide --asvspoof_path and/or --librisevoc_path")
        return

    run_cross_eval(model, device, datasets, args.out_dir)


if __name__ == '__main__':
    main()
