# 🛡️ TrustCall — Deepfake Voice Detection

> **Course Project** — Implementation and extension of neural vocoder artifact-based deepfake audio detection.

---

## 📌 Overview

TrustCall is a deepfake voice detection system that identifies AI-synthesized audio by exploiting **artifacts left behind by neural vocoders**. Unlike traditional approaches that rely on spectral features, this system works at the raw waveform level using a modified **RawNet2** architecture, extended with multiple novel components.

> **Project direction:** this repository now moves forward on the **RawNet track only**.  
> The older DeepVoice/ResNet/Transformer phase scripts under `src/` are retained as legacy stubs.

---

## 🧠 Architecture

The model is based on **RawNet2** with the following components:

- **SincConv Layer** — Learnable sinc-function bandpass filters applied directly to raw waveforms
- **Residual Blocks** — Deep feature extraction with skip connections and attention
- **GRU Layer** — Temporal modeling across the audio sequence
- **Multi-Loss Training** — Binary + auxiliary vocoder-type classification (NLL losses on log-prob outputs)

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
# LibriSeVoc training
python main.py --data_path /path/to/LibriSeVoc/ --model_save_path ./outputs/

# Fast ASVspoof-focused training
python train_rawnet.py --data_path "data/ASVspoof 2019 Dataset 2/LA/LA" --out_dir ./outputs/
```

---

## 🧪 Evaluation

```bash
# Single file
python eval.py --input_path sample.wav --model_path outputs/best_model.pth

# Export dataset predictions (for visualization and metrics)
python predict_dataset.py --dataset asvspoof \
    --data_path "data/ASVspoof 2019 Dataset 2/LA/LA" \
    --split dev --model_path outputs/best_model.pth \
    --out_json outputs/predictions_dev.json

# Visualize exported predictions
python visualize.py --preds_json outputs/predictions_dev.json --out_dir outputs/plots/

# Cross-dataset generalization
python cross_eval.py --model_path outputs/best_model.pth \
    --librisevoc_path /path/to/LibriSeVoc \
    --asvspoof_path /path/to/ASVspoof2019

# (Optional) quick smoke run
python cross_eval.py --model_path outputs/best_model.pth \
    --asvspoof_path "data/ASVspoof 2019 Dataset 2/LA/LA" \
    --max_samples_per_set 256
```

Binary output convention is unified as: **class 0 = real**, **class 1 = fake**.

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
# Train LFCC-LCNN branch
python ensemble.py --mode train_lfcc --dataset asvspoof \
    --data_path "data/ASVspoof 2019 Dataset 2/LA/LA" \
    --lfcc_path outputs/lfcc_model.pth

# Ensemble inference
python ensemble.py --audio sample.wav \
    --rawnet_path outputs/best_model.pth \
    --lfcc_path outputs/lfcc_model.pth
```

`ensemble.py` supports `train_lfcc`, `eval_ensemble`, and `show_lfcc`.

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

## 🧭 Status

- ✅ Active track: RawNet (`main.py`, `train_rawnet.py`, `eval.py`, `cross_eval.py`, `app/demo.py`, etc.)
- ⚠️ Legacy track: `src/training/*` and `src/eval/*` now print deprecation guidance and are not part of the active pipeline.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
