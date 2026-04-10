# 🛡️ TrustCall — Project Definition

## Vision
AI-powered deepfake voice detection system that provides **explainable, real-time** analysis of audio authenticity using raw waveform neural processing.

## Product Goal
Enable users (researchers, security analysts, developers) to detect AI-synthesized speech and identify the specific neural vocoder used, with full XAI transparency — via a polished Streamlit dashboard or programmatic API.

## Core Architecture
- **Model**: Modified RawNet2 (SincConv → 6 ResBlocks → 3-layer GRU → Dual Output Heads)
- **Primary Dataset**: ASVspoof 2019 LA (16kHz, 25k train / 24k dev / 71k eval)
- **Secondary Dataset**: LibriSeVoc (7-class vocoder classification)
- **Frontend**: Streamlit dashboard with live mic, augmentation controls, XAI visualizations
- **Ensemble**: LFCC-LCNN secondary model with weighted averaging (60/40)

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Audio DSP | Librosa, SoundFile |
| Frontend | Streamlit |
| Transcription | OpenAI Whisper (local, base model) |
| Evaluation | scikit-learn (EER, ROC, AUC) |
| Visualization | Matplotlib |
| Config | PyYAML |

## Key Performance Metrics
| Metric | Value |
|--------|-------|
| Test EER | 4.19% (vs 4.54% benchmark) |
| Test Accuracy | 95.20% |
| Dev EER | 1.25% |
| Total Parameters | ~18.7M |

## Author
Akshat Agrawal — Academic project for Soft Computing examination
