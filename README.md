# 🛡️ TrustCall — Deepfake Voice Detection

> **Course Project** — Implementation and extension of neural vocoder artifact-based deepfake audio detection.

---

## 📌 Overview

TrustCall is a deepfake voice detection system that identifies AI-synthesized audio by exploiting **artifacts left behind by neural vocoders**. Unlike traditional approaches that rely on spectral features, this system works at the raw waveform level using a modified **RawNet2** architecture.

The core idea: when a neural vocoder synthesizes speech, it leaves subtle signal-level artifacts that are invisible to the human ear but detectable by a trained model.

---

## 🧠 Architecture

The model is based on **RawNet2** with the following components:

- **SincConv Layer** — Learnable sinc-function filters applied directly to raw waveforms (no hand-crafted features)
- **Residual Blocks** — Deep feature extraction with skip connections
- **GRU Layer** — Temporal modeling across the audio sequence
- **Multi-Loss Training** — Binary cross-entropy + auxiliary loss for improved convergence

```
Raw Waveform → SincConv → ResBlocks → GRU → FC → Real/Fake
```

---

## 📄 Based On

This project implements and extends the method from:

> Sun et al., *"AI-Synthesized Voice Detection Using Neural Vocoder Artifacts"*, CVPRW 2023  
> [Paper Link](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Sun_AI-Synthesized_Voice_Detection_Using_Neural_Vocoder_Artifacts_CVPRW_2023_paper.html)

---

## 🆕 My Extensions

| Feature | Description |
|--------|-------------|
| **ASVspoof 2019 Support** | Added dataset loader for ASVspoof 2019 LA (in addition to LibriSeVoc) |
| **CLI Evaluation** | Single-file evaluation via `eval.py` with JSON output |
| **Configurable via YAML** | All hyperparameters controlled via `model_config_RawNet.yaml` |

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python main.py --data_path /path/to/LibriSeVoc/ --model_save_path ./outputs/
```

---

## 🧪 Evaluation

```bash
python eval.py --input_path /path/to/sample.wav --model_path ./outputs/model.pth
```

---

## 📦 Dataset

- **LibriSeVoc** — Self-vocoded samples from 6 neural vocoders: [Download](https://drive.google.com/file/d/1Zh6b51S1WIsFjdCDRTQhYW61CQ0Ue1lk/view?usp=sharing)
- **ASVspoof 2019 LA** — Standard anti-spoofing benchmark

---

## 📊 Results

| Dataset | EER (Baseline) | EER (This Impl.) |
|---------|---------------|-----------------|
| ASVspoof 2019 | 6.10% | ~4.54% |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
