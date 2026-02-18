"""
Extension 5: Inference Speed Benchmark
Measures model throughput (samples/sec) and latency (ms/sample) across batch sizes.

Usage:
    python benchmark.py --model_path outputs/best_model.pth --config model_config_RawNet.yaml
"""

import argparse
import time
import json
import os
import sys
import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from model import RawNet

SAMPLE_RATE = 24000
AUDIO_LEN   = 96000   # 4 seconds


def benchmark_model(model, device, batch_sizes, n_runs=50, warmup=10):
    """
    Benchmark inference latency and throughput for various batch sizes.
    Returns a list of result dicts.
    """
    results = []
    model.eval()

    for bs in batch_sizes:
        dummy = torch.randn(bs, AUDIO_LEN).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)

        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)  # ms

        mean_ms   = np.mean(latencies)
        std_ms    = np.std(latencies)
        p95_ms    = np.percentile(latencies, 95)
        throughput = (bs * 1000) / mean_ms  # samples/sec

        results.append({
            'batch_size':        bs,
            'mean_latency_ms':   round(mean_ms, 2),
            'std_latency_ms':    round(std_ms, 2),
            'p95_latency_ms':    round(p95_ms, 2),
            'throughput_sps':    round(throughput, 1),
            'ms_per_sample':     round(mean_ms / bs, 3),
        })

        print(f"  BS={bs:3d} | {mean_ms:7.2f}ms ± {std_ms:.2f}ms | "
              f"p95={p95_ms:.2f}ms | {throughput:.1f} samples/s")

    return results


def count_parameters(model):
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description='TrustCall Inference Benchmark')
    parser.add_argument('--model_path', default='outputs/best_model.pth')
    parser.add_argument('--config',     default='model_config_RawNet.yaml')
    parser.add_argument('--device',     default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_sizes', nargs='+', type=int,
                        default=[1, 4, 8, 16, 32])
    parser.add_argument('--n_runs',     type=int, default=50)
    parser.add_argument('--out',        default='outputs/benchmark.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    print(f"\n{'='*55}")
    print(f"  TrustCall Inference Benchmark")
    print(f"{'='*55}")
    print(f"  Device : {device}")
    print(f"  Runs   : {args.n_runs} per batch size")
    print(f"  Audio  : {AUDIO_LEN/SAMPLE_RATE:.1f}s @ {SAMPLE_RATE}Hz\n")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = RawNet(config['model'], device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"  Loaded checkpoint: {args.model_path}")
    else:
        print(f"  Warning: checkpoint not found, using random weights")
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"  Parameters: {total_params:,} total | {trainable_params:,} trainable\n")
    print(f"  {'BS':>4}  {'Mean(ms)':>10}  {'Std(ms)':>8}  {'p95(ms)':>8}  {'Samples/s':>10}")
    print(f"  {'-'*50}")

    results = benchmark_model(model, device, args.batch_sizes, args.n_runs)

    summary = {
        'device':       str(device),
        'model_path':   args.model_path,
        'total_params': total_params,
        'audio_len_s':  AUDIO_LEN / SAMPLE_RATE,
        'results':      results,
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {args.out}")
    print(f"  Best throughput : {max(r['throughput_sps'] for r in results):.1f} samples/s")


if __name__ == '__main__':
    main()
