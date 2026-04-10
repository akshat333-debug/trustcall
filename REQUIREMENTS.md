# 📋 TrustCall — Requirements (v1)

## ✅ Implemented Features

### Core ML Pipeline
- [x] RawNet2 architecture with SincConv + ResBlocks + GRU
- [x] Dual output heads (binary Real/Fake + 7-class vocoder attribution)
- [x] Multi-loss training (λ=0.5 binary + vocoder)
- [x] ASVspoof 2019 LA dataset loader with protocol parsing
- [x] LibriSeVoc dataset loader with train/dev/test splits
- [x] Training script with cosine annealing, gradient clipping, stratified sampling
- [x] Checkpoint save/resume support
- [x] Model config via YAML

### Streamlit Dashboard
- [x] File upload (WAV/FLAC) and live microphone recording
- [x] Real-time waveform and Mel spectrogram visualization
- [x] Binary deepfake probability display with configurable threshold
- [x] Vocoder attribution bar chart (7 vocoders)
- [x] OpenAI Whisper transcription integration
- [x] Lottie animation during inference
- [x] Dark theme styling

### Robustness & Augmentation
- [x] Gaussian noise injection
- [x] Room reverb simulation (RIR convolution)
- [x] Codec compression simulation
- [x] Pitch shift augmentation
- [x] Time stretch
- [x] Volume perturbation
- [x] AugmentedDataset wrapper for training-time augmentation
- [x] Interactive augmentation testing in Streamlit sidebar

### Explainability (XAI)
- [x] SincConv filter bank frequency response visualization
- [x] Input gradient saliency map computation and plotting
- [x] Mel spectrogram overlay in saliency view
- [x] Integrated into Streamlit Tab 2

### Evaluation & Benchmarking
- [x] EER, AUC-ROC, accuracy metrics
- [x] Confusion matrix, ROC curve, score distribution plots
- [x] Cross-dataset evaluation (ASVspoof ↔ LibriSeVoc)
- [x] Inference speed benchmark (latency/throughput profiling)
- [x] Dataset-level prediction JSON export (for visualize.py)

### Ensemble
- [x] LFCC feature extraction (linear frequency cepstral coefficients)
- [x] LCNN architecture with MaxFeatureMap activations
- [x] LFCC-LCNN training pipeline
- [x] TrustCallEnsemble weighted prediction combiner
- [x] CLI for ensemble evaluation

### Documentation
- [x] README.md (professional with badges, demo screenshot, project structure)
- [x] PROJECT_OVERVIEW.md (detailed architecture description)
- [x] VIVA_PREP.md (comprehensive technical deep-dive Q&A)
- [x] setup.py with console_scripts entry points
- [x] requirements.txt

---

## ❌ Missing / Incomplete Features

### Critical Gaps
- [ ] **No demo screenshot** — `assets/demo.png` referenced in README but `assets/` directory is empty
- [ ] **No `eval.py`** — referenced in `setup.py` entry points (`trustcall-eval`) but file doesn't exist
- [ ] **No LFCC model checkpoint** — `outputs/lfcc_model.pth` doesn't exist; ensemble is untrained
- [ ] **No unit tests** — zero test coverage
- [ ] **No CI/CD** — no GitHub Actions or pre-commit hooks

### Architecture Gaps
- [ ] **Sample rate inconsistency** — `explain.py`, `benchmark.py`, `augment.py` use `SAMPLE_RATE=24000` while the actual model and demo use `16000`
- [ ] **Missing Whisper in requirements.txt** — `openai-whisper` not listed despite being imported
- [ ] **Missing `streamlit-lottie` in requirements.txt** — used in demo.py but not declared
- [ ] **`configs/robust_training.yaml`** references `resnet_bilstm` model not implemented anywhere

### Quality-of-Life
- [ ] **No `.env` or config management** — model/config paths hardcoded
- [ ] **No Docker support** — no Dockerfile for reproducible deployment
- [ ] **No REST API** — mentioned in README "Future Improvements" but not started
- [ ] **No production error handling** — demo.py crashes on bad audio silently
