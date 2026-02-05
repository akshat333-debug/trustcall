# TrustCall: Deepfake Voice + Social-Engineering Risk Detector
**Project Review Report**

**Team Members:** (Enter Names Here)  
**Course Code:** CSI3003 Soft Computing  
**Slot:** G2  
**Professor:** Dr. Ilanthenral Kandasamy  

---

## 1. Introduction
Voice spoofing attacks (TTS, Voice Conversion) are becoming increasingly sophisticated, posing threats to biometric systems and social engineering security. TrustCall bridges the gap between black-box Deep Learning detection and human-interpretable risk assessment by integrating state-of-the-art CNN/RNN architectures with a Neuro-Fuzzy decision layer.

## 2. Literature Review Summary Table
| Paper Title | Year | Method | Dataset | Metric | Limitation | TrustCall Improvement |
|---|---|---|---|---|---|---|
| ASVspoof 2019 Baseline | 2019 | GMM/LFCC | ASVspoof 2019 | EER | High EER on LA | Uses Deep Learning (ResNet-BiLSTM) |
| RawNet2 | 2021 | SincNet+ResNet | ASVspoof 2019 | EER ~2% | Black box | Adds Fuzzy Explanation |
| LCNN-LSTM | 2020 | LCNN | ASVspoof 2019 | EER ~1% | Artifacts ignored | Explicit Artifact/Prosody Features |
| ... | ... | ... | ... | ... | ... | ... |
(Add 4 more rows)

## 3. Objective
To develop a robust, explainable audio deepfake detection system that not only classifies audio but provides semantic reasoning (e.g., "High Flatness", "Unstable Prosody") suitable for non-expert users.

## 4. Innovation Component
**System Design:**
[Input Audio] -> [Feature Extraction (LogMel + Handcrafted)] 
   |-> [DL Backbone (ResNet-BiLSTM)] -> [Spoof Prob]
   |-> [Signal Analyzer] -> [Artifact Score]
   |-> [Prosody Analyzer] -> [Stability Score]
         \      |      /
          \     |     /
           [Neuro-Fuzzy Layer (ANFIS)]
                  |
             [Risk Level + Explanation]

**Key Innovations:**
1. **Hybrid Architecture:** Combining data-driven DL with logic-driven Fuzzy systems.
2. **Differentiable Fuzzy Layer:** Allows end-to-end tuning of membership functions.
3. **Interpretability:** Outputs "Why" the audio was flagged.

## 5. Work Done and Implementation

### a. Methodology Adapted
**Dataset:** ASVspoof 2019 LA (Logical Access), covering TTS and VC attacks. 
**Preprocessing:** Resampling to 16kHz, Log-Mel Spectrograms, Time/Freq masking (SpecAugment).
**Model:** 
- **Backbone:** ResNet-18 modified for 1-channel audio.
- **Sequential:** BiLSTM for temporal context.
- **Decision:** Neuro-Fuzzy (3 inputs, 27 rules).

**Experimental Setup:**
- Optimizer: Adam
- Tuning: Optuna (LR, Hidden Size, Dropout)
- Loss: CrossEntropy

### c. Tools Used
- **PyTorch/Torchaudio:** Model development.
- **Librosa:** Signal feature extraction.
- **Streamlit:** Interactive demonstration.
- **Optuna:** Auto-tuning.
- **Matplotlib/Seaborn:** Visualization.

### d. Visualization
(Include screenshot of Streamlit App here)
[GitHub Repository Link Here]

## 6. Results and Discussions
**Metrics on Eval Set:**
- **Accuracy:** ...
- **EER:** ...
- **AUC:** ...

**Confusion Matrix:**
(Insert CM Plot)

**Comparative Analysis:**
TrustCall achieves comparable EER to pure DL baselines but significantly outperforms in *interpretability*.

## 7. Conclusions
TrustCall successfully demonstrates that soft-computing techniques (Fuzzy Logic) can enhance Deep Learning systems by adding a layer of transparency critical for security applications.

## 8. References
1. Todisco, M., et al. "ASVspoof 2019: Future horizons in spoofed and fake audio detection." *Interspeech*. 2019.
2. ...
3. ...
