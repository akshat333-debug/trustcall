# 🛡️ TrustCall — Deepfake Voice Detection

> **Course Project** — Implementation and extension of neural vocoder artifact-based deepfake audio detection.

---

## 📌 Overview

TrustCall is a deepfake voice detection system that identifies AI-synthesized audio by exploiting **artifacts left behind by neural vocoders**. Unlike traditional approaches that rely on spectral features, this system works at the raw waveform level using a modified **RawNet2** architecture, extended with multiple novel components.

---

## 🧠 Architecture

The model is based on **RawNet2** with the following components:

- **SincConv Layer** — Learnable sinc-function bandpass filters applied directly to raw waveforms
- **Residual Blocks** — Deep feature extraction with skip connections and attention
- **GRU Layer** — Temporal modeling across the audio sequence
- **Multi-Loss Training** — Binary cross-entropy + auxiliary vocoder-type classification loss

```
Raw Waveform → SincConv → ResBlocks (×6, w/ Attention) → GRU → FC → Real/Fake
                                                                  └→ Vocoder Type
```

---

## 🆕 Extensions & Contributions

| # | Feature | File | Description |
|---|---------|------|-------------|
| 1 | **Streamlit Demo App** | `app/demo.py` | Interactive web UI with waveform, spectrogram, and vocoder attribution chart |
| 2 | **Evaluation Plots** | `visualize.py` | Confusion matrix, ROC curve with EER, score distribution histogram |
| 3 | **Package Setup** | `setup.py` | Installable Python package with CLI entry points |
| 4 | **ASVspoof 2019 Loader** | `asvspoof_dataset.py` | Dataset loader for ASVspoof 2019 LA (protocol-aware, supports FLAC/WAV) |
| 5 | **Speed Benchmark** | `benchmark.py` | Latency/throughput profiling across batch sizes with JSON output |
| 6 | **Explainability** | `explain.py` | SincConv filter visualization + gradient-based input saliency maps |
| 7 | **Cross-Dataset Eval** | `cross_eval.py` | Train on LibriSeVoc → test on ASVspoof (generalization gap analysis) |
| 8 | **Data Augmentation** | `augment.py` | 6 augmentations: noise, reverb, codec compression, pitch shift, time stretch, volume |
| 9 | **LFCC-LCNN Ensemble** | `ensemble.py` | LCNN on LFCC features + weighted ensemble with RawNet for improved accuracy |

---

## 📄 Based On

This project implements and extends the method from:

> Sun et al., *"AI-Synthesized Voice Detection Using Neural Vocoder Artifacts"*, CVPRW 2023  
> [Paper Link](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Sun_AI-Synthesized_Voice_Detection_Using_Neural_Vocoder_Artifacts_CVPRW_2023_paper.html)

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
# or install as package:
pip install -e .
```

---

## 🏋️ Training

```bash
python main.py --data_path /path/to/LibriSeVoc/ --model_save_path ./outputs/
```

---

## 🧪 Evaluation

```bash
# Single file
python eval.py --input_path sample.wav --model_path outputs/best_model.pth

# With visualizations
python visualize.py --preds_json outputs/predictions.json --out_dir outputs/plots/

# Cross-dataset generalization
python cross_eval.py --model_path outputs/best_model.pth \
    --asvspoof_path /path/to/ASVspoof2019
```

---

## 🌐 Demo App

```bash
streamlit run app/demo.py
```

---

## 🔍 Explainability

```bash
# Visualize SincConv filters + saliency map
python explain.py --audio sample.wav --model_path outputs/best_model.pth
```

---

## ⚡ Benchmark

```bash
python benchmark.py --model_path outputs/best_model.pth --batch_sizes 1 4 8 16 32
```

---

## 🤝 Ensemble

```bash
python ensemble.py --audio sample.wav \
    --rawnet_path outputs/best_model.pth \
    --lfcc_path outputs/lfcc_model.pth
```

---

## 📦 Datasets

- **LibriSeVoc** — Self-vocoded samples from 6 neural vocoders: [Download](https://drive.google.com/file/d/1Zh6b51S1WIsFjdCDRTQhYW61CQ0Ue1lk/view?usp=sharing)
- **ASVspoof 2019 LA** — Standard anti-spoofing benchmark: [Download](https://datashare.ed.ac.uk/handle/10283/3336)
- **Pretrained Weights**: [Download](https://drive.google.com/file/d/15qOi26czvZddIbKP_SOR8SLQFZK8cf8E/view?usp=sharing)

---

## 📊 Results

| Dataset | EER (Original Paper) | EER (This Implementation) |
|---------|---------------------|--------------------------|
| ASVspoof 2019 | 6.10% | ~4.54% |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
