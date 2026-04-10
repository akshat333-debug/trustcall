# 🗺️ TrustCall — Remaining Roadmap

## Phase 1: Critical Bug Fixes & Consistency ⚡ (Priority: HIGH)
> Fix breaking issues that prevent clean installation and usage

- [ ] **1.1** Create missing `eval.py` script (referenced by `setup.py`)
- [ ] **1.2** Fix sample rate inconsistency: update `explain.py`, `benchmark.py`, `augment.py` from 24kHz → 16kHz
- [ ] **1.3** Fix `requirements.txt`: add `openai-whisper`, `streamlit-lottie`, `requests`
- [ ] **1.4** Remove `robust_training.yaml` orphan config (references non-existent `resnet_bilstm` architecture)

## Phase 2: Demo & Presentation Polish 🎨 (Priority: HIGH)
> Make the project submission-ready with visual assets and clean documentation

- [ ] **2.1** Capture Streamlit demo screenshot → save as `assets/demo.png`
- [ ] **2.2** Record a short demo video (upload file → detect → explain) → `assets/demo_video.webp`
- [ ] **2.3** Verify and clean up `outputs/` directory (remove redundant epoch checkpoints)
- [ ] **2.4** Run final Streamlit smoke test to confirm all 3 tabs work end-to-end

## Phase 3: Ensemble Completion 🔧 (Priority: MEDIUM)
> Train the LFCC-LCNN model and enable true ensemble detection

- [ ] **3.1** Train LFCC-LCNN on ASVspoof 2019 dev split (quick 5-epoch run)
- [ ] **3.2** Save `outputs/lfcc_model.pth` checkpoint
- [ ] **3.3** Add ensemble toggle to Streamlit sidebar
- [ ] **3.4** Display ensemble scores alongside RawNet-only scores in UI

## Phase 4: Testing & Reliability 🧪 (Priority: MEDIUM)
> Add automated testing to prevent regressions

- [ ] **4.1** Create `tests/` directory with pytest structure
- [ ] **4.2** Unit test: model forward pass (shape validation)
- [ ] **4.3** Unit test: augmentation functions (output shape, dtype, range)
- [ ] **4.4** Unit test: LFCC extraction (feature dimensions)
- [ ] **4.5** Integration test: end-to-end inference on sample audio

## Phase 5: Deployment & API (Priority: LOW — Future)
> Post-submission enhancements

- [ ] **5.1** FastAPI REST endpoint for programmatic audio analysis
- [ ] **5.2** Dockerfile for containerized deployment
- [ ] **5.3** Streamlit Community Cloud deployment
- [ ] **5.4** ASVspoof 2021 dataset integration for cross-lingual evaluation
