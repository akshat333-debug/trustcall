import numpy as np
import soundfile as sf
import os
import torch

def generate_dummy_flac(path, duration=3.0, sr=16000):
    # Sine wave
    t = np.linspace(0, duration, int(sr*duration))
    # Random freq
    freq = np.random.uniform(200, 800)
    waveform = 0.5 * np.sin(2 * np.pi * freq * t)
    
    sf.write(path, waveform, sr)

# Files to generate based on protocols created
files = {
    "data/asvspoof2019/LA/ASVspoof2019_LA_train/flac": ["LA_T_10001.flac", "LA_T_10002.flac"],
    "data/asvspoof2019/LA/ASVspoof2019_LA_dev/flac": ["LA_D_10001.flac", "LA_D_10002.flac"],
    "data/asvspoof2019/LA/ASVspoof2019_LA_eval/flac": ["LA_E_10001.flac", "LA_E_10002.flac"]
}

for folder, filenames in files.items():
    if not os.path.exists(folder):
        os.makedirs(folder)
    for f in filenames:
        p = os.path.join(folder, f)
        generate_dummy_flac(p)
        print(f"Generated {p}")
