"""Generate dataset-level prediction scores for visualization and analysis.

Output JSON format is compatible with visualize.py:
{
  "labels": [0, 1, ...],
  "scores": [0.02, 0.91, ...],  # fake-class probability (class 1)
  "meta": {...}
}
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from asvspoof_dataset import Dataset_ASVspoof2019
from eval import load_model
from main import Dataset_LibriSeVoc


def build_loader(dataset_name, data_path, split, batch_size, num_workers, max_samples, seed):
    if dataset_name == "asvspoof":
        ds = Dataset_ASVspoof2019(data_path, split=split, resample_to=24000)
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


@torch.no_grad()
def score_loader(model, loader, device):
    labels = []
    scores = []
    model.eval()

    for batch in tqdm(loader, desc="Scoring", total=len(loader)):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, _, y = batch
        else:
            x, y = batch

        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.int64)

        out_binary, _ = model(x)
        probs = torch.exp(out_binary)
        scores.extend(probs[:, 1].cpu().numpy().tolist())  # class 1 = fake
        labels.extend(y.cpu().numpy().tolist())

    return np.array(labels), np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="TrustCall Dataset Prediction Export")
    parser.add_argument("--dataset", choices=["asvspoof", "librisevoc"], required=True)
    parser.add_argument("--data_path", required=True, help="Dataset root path")
    parser.add_argument("--split", required=True, help="Split (asvspoof: train/dev/eval, librisevoc: train/dev/test)")
    parser.add_argument("--model_path", default="outputs/best_model.pth")
    parser.add_argument("--config", default="model_config_RawNet.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", required=True, help="Output JSON path for visualize.py")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    labels, scores = score_loader(model, loader, device)
    out = {
        "labels": labels.astype(int).tolist(),
        "scores": scores.astype(float).tolist(),
        "meta": {
            "dataset": args.dataset,
            "split": args.split,
            "n_samples": int(len(labels)),
            "model_path": args.model_path,
            "timestamp": datetime.now().isoformat(),
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved predictions: {args.out_json}")
    print(f"Samples: {len(labels)} | Real={(labels == 0).sum()} | Fake={(labels == 1).sum()}")
    if len(np.unique(labels)) < 2:
        print("WARNING: Only one class present in exported labels. EER/ROC may be undefined.")


if __name__ == "__main__":
    main()
