import argparse
import os
import numpy as np
import torch
from torch import Tensor
import yaml
from model import RawNet
import librosa
import json
from datetime import datetime

VOCODER_NAMES = ['gt', 'wavegrad', 'diffwave', 'parallel_wave_gan', 'wavernn', 'wavenet', 'melgan']
BENIGN_MISSING_SINC = {
    'Sinc_conv.low_hz_',
    'Sinc_conv.band_hz_',
    'Sinc_conv.window_',
    'Sinc_conv.n_',
}


def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def load_sample(sample_path, max_len=96000):
    y, sr = librosa.load(sample_path, sr=None)
    if sr != 24000:
        y = librosa.resample(y, orig_sr=sr, target_sr=24000)

    if len(y) <= max_len:
        return [Tensor(pad(y, max_len))]

    y_list = []
    n_seg = int(np.ceil(len(y) / max_len))
    for i in range(n_seg):
        y_seg = y[i * max_len: (i + 1) * max_len]
        y_list.append(Tensor(pad(y_seg, max_len)))
    return y_list


def load_model(model_path, config_path='model_config_RawNet.yaml', device='cpu'):
    """Load RawNet model, gracefully handling missing or incompatible checkpoints."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = RawNet(config['model'], device)
    model = model.to(device)
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f'Model loaded: {model_path}')
            non_benign_missing = [k for k in missing if k not in BENIGN_MISSING_SINC]
            if non_benign_missing:
                print(f'  Missing keys: {len(non_benign_missing)} (e.g., {non_benign_missing[:3]})')
            elif missing:
                print('  Checkpoint compatibility: using initialized SincConv parameters.')
            if unexpected:
                print(f'  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})')
        except RuntimeError as e:
            print(f'WARNING: Could not load checkpoint (architecture mismatch): {e}')
            print('  Using randomly initialized weights.')
    else:
        print(f'WARNING: Checkpoint not found at "{model_path}". Using random weights.')
    model.eval()
    return model
    

def run_eval(input_path, model_path, config_path='model_config_RawNet.yaml'):
    """Run evaluation on a single audio file. Returns result dict."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = load_model(model_path, config_path, device)

    out_list_multi  = []
    out_list_binary = []
    for m_batch in load_sample(input_path):
        m_batch = m_batch.to(device=device, dtype=torch.float).unsqueeze(0)
        logits, multi_logits = model(m_batch)
        probs = torch.exp(logits)
        probs_multi = torch.exp(multi_logits)
        out_list_multi.append(probs_multi.tolist()[0])
        out_list_binary.append(probs.tolist()[0])

    result_multi  = np.average(out_list_multi,  axis=0).tolist()
    result_binary = np.average(out_list_binary, axis=0).tolist()

    # Unified convention across RawNet scripts: class 0=real, class 1=fake.
    verdict = 'FAKE' if result_binary[1] > 0.5 else 'REAL'
    print(f'\nVerdict: {verdict}')
    print(f'Binary  — real: {result_binary[0]:.4f}  fake: {result_binary[1]:.4f}')
    print('Vocoder — ' + '  '.join(f'{n}:{v:.3f}' for n, v in zip(VOCODER_NAMES, result_multi)))

    return {
        'verdict':      verdict,
        'real_prob':    round(result_binary[0], 4),
        'fake_prob':    round(result_binary[1], 4),
        'vocoder':      {n: round(v, 4) for n, v in zip(VOCODER_NAMES, result_multi)},
        'file':         os.path.basename(input_path),
        'timestamp':    datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description='TrustCall — Single Audio Evaluation')
    parser.add_argument('--input_path',  required=True,  help='Path to audio file (WAV/FLAC)')
    parser.add_argument('--model_path',  default='outputs/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--config',      default='model_config_RawNet.yaml')
    parser.add_argument('--output_json', default=None,   help='Optional: save result to JSON')
    args = parser.parse_args()

    result = run_eval(args.input_path, args.model_path, args.config)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Result saved: {args.output_json}')


if __name__ == '__main__':
    main()
