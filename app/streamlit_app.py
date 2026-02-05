import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import os
import sys
import io
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

# Fix import path when running from app/ directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.models.resnet_bilstm import ResNetBiLSTM
from src.models.anfis_fuzzy import NeuroFuzzyLayer
from src.data.features import compute_artifact_scores, compute_prosody_scores
from src.models.artifact_prosody import ArtifactProsodyScorer

# Page Config
st.set_page_config(page_title="TrustCall Scam Shield", layout="wide")

st.title("🛡️ TrustCall: Deepfake Voice & Scam Detector")
st.markdown("### Advanced Audio Deepfake Detection with Explainable AI (Neuro-Fuzzy)")

# Sidebar
st.sidebar.header("Configuration")
default_model = os.path.join(PROJECT_ROOT, "outputs/deepvoice_proposed/best_model.pth")
default_config = os.path.join(PROJECT_ROOT, "configs/deepvoice.yaml")
model_path = st.sidebar.text_input("Model Checkpoint Path", default_model)
config_path = st.sidebar.text_input("Config Path", default_config)
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5)

# Feature Extractor using librosa (no torchaudio dependency)
class LibrosaFeatureExtractor:
    def __init__(self, config):
        self.sample_rate = config['data']['sample_rate']
        self.n_fft = config['data'].get('n_fft', 1024)
        self.hop_length = config['data'].get('hop_length', 256)
        self.n_mels = config['data'].get('n_mels', 80)
        
    def extract(self, waveform_np, sr):
        # Resample if needed
        if sr != self.sample_rate:
            waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=self.sample_rate)
        
        # Compute log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform_np, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        log_mel = np.log(mel_spec + 1e-6)
        return torch.tensor(log_mel, dtype=torch.float32)

# Load System
@st.cache_resource
def load_system(config_file, checkpoint):
    config = load_config(config_file)
    device = torch.device('cpu')
    
    # DL Model
    model = ResNetBiLSTM(config)
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        st.warning(f"Checkpoint not found at {checkpoint}. Using initialized weights (Random).")
    model.eval()
    
    # Fuzzy Layer
    fuzzy = NeuroFuzzyLayer(config)
    fuzzy.eval()
    
    scorer = ArtifactProsodyScorer(config)
    extractor = LibrosaFeatureExtractor(config)
    
    return model, fuzzy, scorer, extractor, config

try:
    model, fuzzy, scorer, extractor, config = load_system(config_path, model_path)
    st.sidebar.success("System Loaded")
except Exception as e:
    st.sidebar.error(f"Failed to load system: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Audio Analysis", "🎤 Live Recording", "Batch Processing"])

with tab1:
    uploaded_file = st.file_uploader("Upload Audio (FLAC/WAV)", type=["flac", "wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Load audio using librosa
        uploaded_file.seek(0)
        waveform_np, sr = librosa.load(uploaded_file, sr=None, mono=True)
        
        # Display Spectrogram
        st.subheader("Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Waveform**")
            fig_wav, ax_wav = plt.subplots(figsize=(10, 2))
            ax_wav.plot(waveform_np)
            st.pyplot(fig_wav)
            
        with col2:
            st.markdown("**Log-Mel Spectrogram**")
            features = extractor.extract(waveform_np, sr)
            # Ensure shape for model (1, F, T)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            
            fig_spec, ax_spec = plt.subplots(figsize=(10, 2))
            ax_spec.imshow(features.squeeze().numpy(), aspect='auto', origin='lower')
            st.pyplot(fig_spec)

        # DL Inference with error handling
        try:
            with st.spinner("Running model inference..."):
                with torch.no_grad():
                    logits = model(features.unsqueeze(0))  # Add batch dim
                    prob_bonafide = torch.softmax(logits, dim=1)[0, 1].item()
                    prob_spoof = 1.0 - prob_bonafide
                    
                # Artifact/Prosody Scores
                raw_art = compute_artifact_scores(waveform_np, sr)
                raw_pros = compute_prosody_scores(waveform_np, sr)
                
                score_art = scorer.normalize_artifact(raw_art)
                score_pros = scorer.normalize_prosody(raw_pros)
                
                # Fuzzy Inference
                fuzzy_input = torch.tensor([[prob_spoof, score_art, score_pros]])
                risk_score, firing_strengths, memberships = fuzzy(fuzzy_input)
                risk_val = risk_score.item()
        
            # Results
            st.divider()
            st.subheader("🛡️ Detection Results")
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("DL Spoof Probability", f"{prob_spoof:.2%}", delta_color="inverse")
            c2.metric("Artifact Score", f"{score_art:.2f}")
            c3.metric("Prosody Stability", f"{score_pros:.2f}")
            
            st.markdown("### Risk Assessment")
            
            if risk_val < 0.33:
                risk_label = "LOW RISK (Likely Real)"
                color = "green"
            elif risk_val < 0.66:
                risk_label = "MEDIUM RISK (Suspicious)"
                color = "orange"
            else:
                risk_label = "HIGH RISK (Likely Deepfake)"
                color = "red"
                
            st.markdown(f"<h2 style='color:{color}'>{risk_label} (Score: {risk_val:.2f})</h2>", unsafe_allow_html=True)
            
            # Explanation
            with st.expander("Show Neuro-Fuzzy Explanation"):
                st.write("Top Fired Rules:")
                strengths = firing_strengths[0].detach().numpy()
                indices = np.argsort(strengths)[::-1]
                
                for i in range(3):
                    idx = indices[i]
                    val = strengths[idx]
                    if val > 0.01:
                        rule_desc = fuzzy.get_rule_interpretation(idx)
                        st.write(f"- **Rule {idx+1}** ({val:.2f}): IF {rule_desc} THEN Risk contributes.")
                
                st.write("---")
                st.write("Interpretation:")
                if prob_spoof > 0.8:
                    st.write("- Deep Learning model strongly detects spoofing patterns.")
                if score_art > 0.7:
                    st.write("- High signal artifacts detected (flatness/variance anomalies).")
                if score_pros < 0.3:
                    st.write("- Unnatural prosody stability (monotone or jittery).")
                    
        except Exception as e:
            st.error(f"Error during analysis: {e}")

with tab2:
    st.header("🎤 Live Recording")
    st.write("Click the microphone button below to record audio directly from your device.")
    st.info("**Tip**: Record for at least 3-5 seconds for accurate detection.")
    
    # Audio recorder component
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="2x",
        sample_rate=16000
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        try:
            with st.spinner("Analyzing recorded audio..."):
                # Convert bytes to numpy array
                audio_buffer = io.BytesIO(audio_bytes)
                waveform_np, sr = librosa.load(audio_buffer, sr=16000, mono=True)
                
                # Feature extraction
                features = extractor.extract(waveform_np, sr)
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Waveform**")
                    fig_wav, ax_wav = plt.subplots(figsize=(10, 2))
                    ax_wav.plot(waveform_np)
                    ax_wav.set_xlim([0, len(waveform_np)])
                    st.pyplot(fig_wav)
                    
                with col2:
                    st.markdown("**Log-Mel Spectrogram**")
                    fig_spec, ax_spec = plt.subplots(figsize=(10, 2))
                    ax_spec.imshow(features.squeeze().numpy(), aspect='auto', origin='lower')
                    st.pyplot(fig_spec)
                
                # DL Inference
                with torch.no_grad():
                    logits = model(features.unsqueeze(0))
                    prob_bonafide = torch.softmax(logits, dim=1)[0, 1].item()
                    prob_spoof = 1.0 - prob_bonafide
                
                # Artifact/Prosody Scores
                raw_art = compute_artifact_scores(waveform_np, sr)
                raw_pros = compute_prosody_scores(waveform_np, sr)
                score_art = scorer.normalize_artifact(raw_art)
                score_pros = scorer.normalize_prosody(raw_pros)
                
                # Fuzzy Inference
                fuzzy_input = torch.tensor([[prob_spoof, score_art, score_pros]])
                risk_score, firing_strengths, memberships = fuzzy(fuzzy_input)
                risk_val = risk_score.item()
                
                # Results
                st.divider()
                st.subheader("🛡️ Live Detection Results")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("DL Spoof Probability", f"{prob_spoof:.2%}", delta_color="inverse")
                c2.metric("Artifact Score", f"{score_art:.2f}")
                c3.metric("Prosody Stability", f"{score_pros:.2f}")
                
                st.markdown("### Risk Assessment")
                
                if risk_val < 0.33:
                    risk_label = "LOW RISK (Likely Real)"
                    color = "green"
                elif risk_val < 0.66:
                    risk_label = "MEDIUM RISK (Suspicious)"
                    color = "orange"
                else:
                    risk_label = "HIGH RISK (Likely Deepfake)"
                    color = "red"
                    
                st.markdown(f"<h2 style='color:{color}'>{risk_label} (Score: {risk_val:.2f})</h2>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error analyzing recording: {e}")

with tab3:
    st.header("Batch Processing")
    st.write("Point to a directory of WAV/FLAC files to process them all.")
    batch_dir = st.text_input("Folder Path (Absolute)")
    if st.button("Run Batch"):
        if os.path.exists(batch_dir):
            files = [f for f in os.listdir(batch_dir) if f.endswith('.flac') or f.endswith('.wav')]
            st.write(f"Found {len(files)} files.")
            
            if len(files) > 0:
                results = []
                progress = st.progress(0)
                
                with torch.no_grad():
                    for i, fname in enumerate(files):
                        fpath = os.path.join(batch_dir, fname)
                        try:
                            # Load using librosa
                            waveform_np, sr = librosa.load(fpath, sr=None, mono=True)
                            features = extractor.extract(waveform_np, sr)
                            if features.dim() == 2:
                                features = features.unsqueeze(0)
                            
                            # DL
                            logits = model(features.unsqueeze(0))
                            prob_spoof = 1.0 - torch.softmax(logits, dim=1)[0, 1].item()
                            
                            # Fuzzy Attributes
                            raw_art = compute_artifact_scores(waveform_np, sr)
                            raw_pros = compute_prosody_scores(waveform_np, sr)
                            
                            score_art = scorer.normalize_artifact(raw_art)
                            score_pros = scorer.normalize_prosody(raw_pros)
                            
                            # Fuzzy Score
                            fuzzy_input = torch.tensor([[prob_spoof, score_art, score_pros]])
                            risk_score, _, _ = fuzzy(fuzzy_input)
                            
                            results.append({
                                "Filename": fname,
                                "Spoof Prob": round(prob_spoof, 4),
                                "Artifact Score": round(score_art, 4),
                                "Prosody Score": round(score_pros, 4),
                                "Risk Score": round(risk_score.item(), 4),
                                "Decision": "Fake" if risk_score.item() > 0.5 else "Real"
                            })
                        except Exception as e:
                            st.warning(f"Error processing {fname}: {e}")
                        
                        progress.progress((i + 1) / len(files))
                
                st.success("Batch Processing Complete")
                st.dataframe(results)
                
                # CSV Download
                import pandas as pd
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "trustcall_results.csv", "text/csv")
        else:
            st.error("Directory not found.")
