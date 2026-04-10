"""Evaluate a trained RawNet model on ASVspoof 2019 LA or LibriSeVoc.

Computes accuracy, EER, and AUC-ROC on the specified split.

Usage:
    python eval.py --dataset asvspoof \
        --data_path "data/ASVspoof 2019 Dataset 2/LA/LA" \
        --split dev \
        --model_path outputs/best_model.pth

    python eval.py --dataset librisevoc \
        --data_path /path/to/LibriSeVoc \
        --split test \
        --model_path outputs/best_model.pth
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model import load_model
from asvspoof_dataset import Dataset_ASVspoof2019
from main import Dataset_LibriSeVoc

SAMPLE_RATE = 16000


def build_loader(dataset_name, data_path, split, batch_size, num_workers, max_samples, seed):
    """Build a DataLoader for the requested dataset and split."""
    if dataset_name == "asvspoof":
        ds = Dataset_ASVspoof2019(data_path, split=split)
    elif dataset_name == "librisevoc":
        ds = Dataset_LibriSeVoc(data_path, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if max_samples is not None:
        max_n = min(max_samples, len(ds))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=max_n, replace=False).tolist()
        ds = Subset(ds, idx)

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def compute_eer(y_true, y_score):
    """Equal Error Rate: point where FAR == FRR."""
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return float("nan"), 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


@torch.no_grad()
def evaluate(model, loader, device):
    """Run model on a DataLoader, return labels, scores, and accuracy."""
    all_labels, all_scores = [], []
    correct, total = 0, 0
    model.eval()

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, _, y = batch
        else:
            x, y = batch

        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.int64)

        out_binary, _ = model(x)
        probs = torch.exp(out_binary)

        preds = out_binary.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        all_scores.extend(probs[:, 1].cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    accuracy = correct / max(total, 1)
    return np.array(all_labels), np.array(all_scores), accuracy


def main():
    parser = argparse.ArgumentParser(description="TrustCall Model Evaluation")
    parser.add_argument("--dataset", choices=["asvspoof", "librisevoc"], required=True)
    parser.add_argument("--data_path", required=True, help="Dataset root path")
    parser.add_argument("--split", required=True,
                        help="Split (asvspoof: train/dev/eval, librisevoc: train/dev/test)")
    parser.add_argument("--model_path", default="outputs/best_model.pth")
    parser.add_argument("--config", default="model_config_RawNet.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on number of evaluation samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", default=None,
                        help="Optional path to save results JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  TrustCall — Model Evaluation")
    print(f"{'='*55}")
    print(f"  Device  : {device}")
    print(f"  Dataset : {args.dataset} ({args.split})")
    print(f"  Model   : {args.model_path}")

    model = load_model(args.model_path, args.config, device=device)
    loader = build_loader(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    labels, scores, accuracy = evaluate(model, loader, device)
    eer, eer_thresh = compute_eer(labels, scores)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())

    print(f"\n  Results:")
    print(f"  {'─'*40}")
    print(f"  Samples   : {len(labels)} ({n_real} real, {n_fake} fake)")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  EER       : {eer:.4f} ({eer*100:.2f}%)")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  EER Thresh: {eer_thresh:.4f}")

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "model_path": args.model_path,
        "n_samples": int(len(labels)),
        "n_real": n_real,
        "n_fake": n_fake,
        "accuracy": round(float(accuracy), 4),
        "eer": round(float(eer), 4) if np.isfinite(eer) else None,
        "auc": round(float(auc), 4) if np.isfinite(auc) else None,
        "eer_threshold": round(float(eer_thresh), 4),
    }

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {args.out_json}")

    print(f"\n  ✅ Evaluation complete.")


if __name__ == "__main__":
    main()
