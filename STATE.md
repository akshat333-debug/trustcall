# 📊 TrustCall — Current State (Snapshot)

**Last Updated**: 2026-04-10

## Project Status: 🟡 85% Complete

### What Works
- ✅ Full RawNet2 model training pipeline (ASVspoof 2019)
- ✅ Trained checkpoint: `outputs/best_model.pth` (74.7 MB)
- ✅ Streamlit dashboard: upload, record, visualize, detect, explain
- ✅ Whisper transcription integrated
- ✅ Cross-dataset evaluation pipeline
- ✅ LFCC-LCNN ensemble architecture (code complete)
- ✅ Data augmentation pipeline (6 augmentation types)
- ✅ Comprehensive viva preparation document
- ✅ Git repo: 33 commits on `main`, pushed to `origin/main`

### What's Broken / Missing
- ❌ **`eval.py` missing** — setup.py references it but it doesn't exist → `pip install -e .` will create broken `trustcall-eval` console command
- ❌ **`assets/demo.png` missing** — README references a demo screenshot that doesn't exist
- ❌ **Sample rate mismatch** — explain.py/benchmark.py/augment.py hardcode 24kHz but model expects 16kHz
- ❌ **Incomplete requirements.txt** — missing `openai-whisper`, `streamlit-lottie`, `requests`
- ❌ **LFCC model untrained** — ensemble predict will use random LCNN weights

### Key Decisions Made
1. Using `legacy_resblock: true` for checkpoint compatibility with csun22 weights
2. Using `learnable_sinc: false` in config for pre-trained checkpoint loading
3. Binary head is the sole verdict authority; vocoder head is supplementary
4. Label convention: 0=real, 1=fake throughout all scripts
5. Augmentations are test-time interactive only (not training-time by default)

### Training Outputs
```
outputs/
├── best_model.pth          (74.7MB — main RawNet checkpoint)
├── model_epoch_1.pth       (74.7MB — epoch 1 backup)
├── benchmark.json          (speed profiling results)
├── preds_smoke.json        (small test prediction export)
├── run_result.json         (metadata)
├── training_log.txt        (2.7MB — full training logs)
├── streamlit_log.txt       (40KB — Streamlit session log)
├── cross_eval/             (cross-dataset results & charts)
├── plots/                  (evaluation visualization outputs)
└── ...various run dirs...
```
