"""
Extension 2: Evaluation Visualizations
Generates confusion matrix, ROC curve, and EER plot from model predictions.

Usage:
    python visualize.py --preds_json outputs/predictions.json --out_dir outputs/plots/
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay, classification_report
)


def compute_eer(y_true, y_score):
    """Equal Error Rate: point where FAR == FRR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold, fpr, tpr


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_roc_curve(fpr, tpr, roc_auc, eer, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#4c8bf5', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.scatter([eer], [1 - eer], color='red', zorder=5, s=80,
               label=f'EER = {eer:.3f}')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_score_distribution(y_true, y_score, out_path):
    """Histogram of model scores for real vs fake."""
    real_scores = [s for s, l in zip(y_score, y_true) if l == 0]
    fake_scores = [s for s, l in zip(y_score, y_true) if l == 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real_scores, bins=50, alpha=0.6, color='#21c354', label='Real', density=True)
    ax.hist(fake_scores, bins=50, alpha=0.6, color='#ff4b4b', label='Fake', density=True)
    ax.set_xlabel('Fake Probability Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution: Real vs Fake', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def generate_all_plots(preds_json, out_dir, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)

    with open(preds_json, 'r') as f:
        data = json.load(f)

    y_true  = np.array(data['labels'])
    y_score = np.array(data['scores'])
    y_pred  = (y_score >= threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"  TrustCall Evaluation Report")
    print(f"{'='*50}")
    print(f"  Samples: {len(y_true)} | Threshold: {threshold}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

    eer, eer_thresh, fpr, tpr = compute_eer(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"  EER:     {eer:.4f} (threshold={eer_thresh:.4f})")
    print(f"  AUC-ROC: {roc_auc:.4f}")

    plot_confusion_matrix(y_true, y_pred,
                          os.path.join(out_dir, 'confusion_matrix.png'))
    plot_roc_curve(fpr, tpr, roc_auc, eer,
                   os.path.join(out_dir, 'roc_curve.png'))
    plot_score_distribution(y_true, y_score,
                            os.path.join(out_dir, 'score_distribution.png'))

    summary = {
        'eer': round(eer, 4),
        'auc': round(roc_auc, 4),
        'eer_threshold': round(float(eer_thresh), 4),
        'n_samples': int(len(y_true)),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {out_dir}/summary.json")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrustCall Evaluation Visualizer')
    parser.add_argument('--preds_json', required=True,
                        help='Path to predictions JSON (keys: labels, scores)')
    parser.add_argument('--out_dir', default='outputs/plots',
                        help='Output directory for plots')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    generate_all_plots(args.preds_json, args.out_dir, args.threshold)
