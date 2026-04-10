# 🛡️ TrustCall — Project Overview

## 1. Project Description
TrustCall is an open-source AI system designed to detect synthetic and deepfake audio. As generative voice models (like WaveNet, ElevenLabs clones, etc.) become more advanced and easily accessible, the risk of voice-based impersonation and fraud increases. TrustCall addresses this by providing a reproducible and explainable pipeline for verifying the authenticity of audio recordings.

It is a binary classification tool at its core—determining whether a given audio clip is human (bonafide) or AI-generated (spoofed)—but goes a step further to identify the specific neural vocoder used to generate the spoofed speech (via a 7-class multi-label model).

## 2. Key Features
- **Raw Waveform Analysis**: Analyzes the raw audio waveforms directly, bypassing traditional Mel Spectrogram constraints that lose high-frequency deepfake artifacts.
- **Explainable AI (XAI)**: Provides full transparency using SincConv bandpass filters and saliency maps, showing exactly *which* parts and frequencies of the audio the model found suspicious.
- **Live Detection & UI**: Features an intuitive Streamlit dashboard for real-time live microphone recordings or file uploads.
- **Audio Augmentation for Robustness**: Supports interactive testing against background noise, room reverb, codec compression (MP3/Opus), and pitch shifts.
- **Contextual Transcriptions**: Uses OpenAI Whisper to provide automated dictation context alongside its deepfake probability metric.
- **Ensemble mode**: Employs an LFCC-LCNN complementary secondary model to vote alongside the primary RawNet model.

## 3. How It Works (Architecture)

TrustCall avoids traditional models that rely on compressed feature inputs. Instead, it utilizes a custom variation of **RawNet2**, trained natively on the ASVspoof 2019 LA dataset via Multi-Loss (Binary & Vocoder Types).

**The Pipeline:**
1. **Input:** The raw audio waveform series `[1 x 64000 samples]` accurately representing 4 seconds of audio at a sample rate of 16,000 Hz.
2. **SincConv Layer:** Acts as learnable bandpass filters that cut the audio into frequency bands. Unlike standard CNNs, these filters explicitly learn which frequency ranges are suspicious in deepfakes (often bridging the 3-8kHz spectrum).
3. **ResBlocks:** A progressive sequence of residual blocks with batch normalization, LeakyReLU, and Convolutions to learn abstract high-level patterns.
4. **GRU Layer:** A 3-layer Gated Recurrent Unit (GRU) models the temporal sequence dependencies across the 4-second audio window.
5. **Output Heads:** 
   - **Binary Head:** Resolves the basic overarching inquiry to evaluate Real vs. Fake probabilities (controls the official system verdict).
   - **Multi-Label Head:** Qualitatively infers the type of neural synthesizer architecture used (e.g. WaveNet, MelGAN, DiffWave).

## 4. End-to-End Inference Flow

When a user interacts with the TrustCall system via the visual dashboard, the following chronological end-to-end flow executes:

1. **Submission:** User uploads an audio file (`.wav` or `.flac`) or records live directly from the browser microphone.
2. **Preprocessing:** Audio is decoded into a fast `numpy` array, resampled strictly to 16,000 Hz if necessary, and precisely padded/trimmed to a uniform length of 64,000 samples.
3. **Augmentations (Optional):** Custom corruptions (i.e. noise overlay or room reverb) are dynamically applied based on user sidebar settings to test operational robustness.
4. **Transcription:** The OpenAI Whisper model dictates the holistic context and speech-to-text semantic transcription.
5. **Inference:** The finalized model independently evaluates the raw timeline sequence extracting probabilities directly linked to deepfake traits.
6. **Visualization Generation:** The Streamlit dashboard natively renders the final analytical assessment matrix including:
   - **Dark Blue Waveform:** Highlighting generalized raw amplitude variance sequences.
   - **Magma Mel Spectrogram Heatmap:** Structurally visualizing voice harmonic integrity and disclosing high-frequency compression artifacts.
   - **Probability Scores Matrix:** Presenting the exponential conversion log-softmax outputs explicitly as numerical `prob_real` vs `prob_fake` proportions.
   - **Vocoder Signature Chart:** Mapping out the probabilistic multi-class forensic fingerprint mapping the likely vocal clone software configuration utilized.

## 5. Result Verification
A standardized detection threshold (baseline defaults at: 0.50) safely maps the continuous probability output into categorical outcomes immediately relayed to the final user:
- **If [prob_fake > threshold]** -> 🚨 DEEPFAKE DETECTED
- **If [prob_fake <= threshold]** -> ✅ LIKELY GENUINE

*Note: While the multi-label head provides valuable forensic inferences exploring the likely synthetic origin mechanics, the Binary Head's assessment logic guarantees the singular, authoritative determination for TrustCall's final fraud verdict.*
