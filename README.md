# TrustCall: Deepfake Voice + Social-Engineering Risk Detector (Scam Shield)

TrustCall is an end-to-end system designed to detect spoofed audio (using ASVspoof 2019 data) and provide interpretable risk explanations using a novel Neuro-Fuzzy decision layer.

## Project Structure
```
trustcall-deepfake-voice-shield/
├── data/               # Dataset storage (ASVspoof 2019)
├── configs/            # YAML configuration files
├── scripts/            # Helper scripts (verify dataset, instructions)
├── src/                # Source code
│   ├── data/           # Data loading and processing
│   ├── models/         # Model definitions (CNN, ResNet-BiLSTM, Fuzzy)
│   ├── training/       # Training scripts
│   ├── eval/           # Evaluation metric and plotting
│   └── utils/          # Utilities (config, seeding, logging)
├── app/                # Streamlit demo application
├── docs/               # Documentation and reports
├── outputs/            # Experiment outputs (logs, checkpoints)
└── tests/              # Unit tests
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/trustcall-deepfake-voice-shield.git
   cd trustcall-deepfake-voice-shield
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Setup:**
   - Please refer to `scripts/prepare_data_instructions.md` for detailed instructions on downloading and placing the ASVspoof 2019 dataset.
   - Run verification: `python scripts/verify_dataset.py`

## Usage

### Training Baselines
```bash
# MFCC + Traditional ML
python src/training/train_baseline_mfcc.py

# Basic CNN
python src/training/train_baseline_cnn.py
```

### Training Proposed Model
```bash
python src/training/train_proposed.py
```

### Evaluation
```bash
python src/eval/evaluate.py --model_path outputs/checkpoints/best_model.pth --config configs/proposed_resnet_bilstm.yaml
```

### Dashboard / Demo
```bash
streamlit run app/streamlit_app.py
```

## Features
- **Binary Classification**: Bonafide vs Spoof
- **Neuro-Fuzzy Layer**: "White-box" explanation of risk (Low/Medium/High) based on spoof probability, signal artifacts, and prosody stability.
- **SpecAugment**: Data augmentation for robustness.
- **Streamlit App**: Interactive demo with batch processing.
