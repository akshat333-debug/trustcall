"""
Extension 1: Streamlit Demo App for TrustCall
Run: streamlit run app/demo.py
"""

import streamlit as st
import torch
import numpy as np
import librosa
import yaml
import os
import sys
import io
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "trustcall-cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "trustcall-mpl"))
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import RawNet

st.set_page_config(page_title="TrustCall — Deepfake Voice Detector", layout="wide", page_icon="🛡️")
BENIGN_MISSING_SINC = {'Sinc_conv.low_hz_', 'Sinc_conv.band_hz_', 'Sinc_conv.window_', 'Sinc_conv.n_'}

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0e1117; }
    .risk-high  { color: #ff4b4b; font-size: 2rem; font-weight: 700; }
    .risk-low   { color: #21c354; font-size: 2rem; font-weight: 700; }
    .risk-med   { color: #ffa500; font-size: 2rem; font-weight: 700; }
    .metric-box { background: #1e2130; border-radius: 10px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ TrustCall — Deepfake Voice Detector")
st.markdown("*Detecting AI-synthesized voices via neural vocoder artifact analysis*")

SAMPLE_RATE = 24000
MAX_LEN = 96000

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cpu')
    model = RawNet(config['model'], device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        st.sidebar.success("✅ Model loaded")
        non_benign_missing = [k for k in missing if k not in BENIGN_MISSING_SINC]
        if non_benign_missing or unexpected:
            st.sidebar.info(
                f"Checkpoint partially loaded (missing={len(non_benign_missing)}, unexpected={len(unexpected)})"
            )
    else:
        st.sidebar.warning("⚠️ No checkpoint found — using random weights")
    model.eval()
    return model, device

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
default_model = os.path.join(PROJECT_ROOT, "outputs", "best_model.pth")
default_config = os.path.join(PROJECT_ROOT, "model_config_RawNet.yaml")
model_path = st.sidebar.text_input("Model Path", default_model)
config_path = st.sidebar.text_input("Config Path", default_config)
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.01)

model, device = load_model(model_path, config_path)

# ── Helpers ───────────────────────────────────────────────────────────────────
def pad_audio(x, max_len=MAX_LEN):
    if len(x) >= max_len:
        return x[:max_len]
    repeats = int(max_len / len(x)) + 1
    return np.tile(x, repeats)[:max_len]

def run_inference(waveform_np, sr):
    if sr != SAMPLE_RATE:
        waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=SAMPLE_RATE)
    waveform_np = pad_audio(waveform_np)
    x = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out_binary, out_multi = model(x)
    prob_fake = torch.exp(out_binary[0, 1]).item()
    vocoder_probs = torch.exp(out_multi[0]).cpu().numpy()
    return prob_fake, vocoder_probs

VOCODER_NAMES = ["gt", "wavegrad", "diffwave", "parallel_wave_gan", "wavernn", "wavenet", "melgan"]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎵 Upload Audio", "📊 About"])

with tab1:
    uploaded = st.file_uploader("Upload a WAV or FLAC file", type=["wav", "flac"])

    if uploaded:
        st.audio(uploaded)
        uploaded.seek(0)
        waveform_np, sr = librosa.load(io.BytesIO(uploaded.read()), sr=None, mono=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Waveform**")
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.plot(waveform_np, color='#4c8bf5', linewidth=0.5)
            ax.set_facecolor('#1e2130'); fig.patch.set_facecolor('#1e2130')
            ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("**Mel Spectrogram**")
            mel = librosa.feature.melspectrogram(y=waveform_np, sr=sr, n_mels=80)
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.imshow(np.log(mel + 1e-6), aspect='auto', origin='lower', cmap='magma')
            ax.set_facecolor('#1e2130'); fig.patch.set_facecolor('#1e2130')
            st.pyplot(fig); plt.close()

        with st.spinner("🔍 Analyzing..."):
            prob_fake, vocoder_probs = run_inference(waveform_np, sr)

        st.divider()
        st.subheader("🛡️ Detection Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Fake Probability", f"{prob_fake:.1%}")
        c2.metric("Real Probability", f"{1-prob_fake:.1%}")
        c3.metric("Threshold", f"{threshold:.2f}")

        if prob_fake > threshold:
            st.markdown(f'<p class="risk-high">🚨 DEEPFAKE DETECTED ({prob_fake:.1%})</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="risk-low">✅ LIKELY GENUINE ({1-prob_fake:.1%})</p>', unsafe_allow_html=True)

        st.markdown("**Vocoder Attribution (which synthesizer?)**")
        fig, ax = plt.subplots(figsize=(10, 3))
        colors = ['#21c354' if i == 0 else '#ff4b4b' for i in range(7)]
        bars = ax.bar(VOCODER_NAMES, vocoder_probs, color=colors)
        ax.set_ylabel("Probability", color='white')
        ax.set_facecolor('#1e2130'); fig.patch.set_facecolor('#1e2130')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
        for bar, val in zip(bars, vocoder_probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', color='white', fontsize=9)
        st.pyplot(fig); plt.close()

with tab2:
    st.markdown("""
    ## About TrustCall

    TrustCall detects AI-synthesized (deepfake) voices by analyzing **neural vocoder artifacts** —
    subtle signal-level distortions left behind when a neural vocoder synthesizes speech.

    ### How it works
    1. Raw audio is passed through **SincConv** learnable bandpass filters
    2. **Residual blocks** extract deep features
    3. A **GRU** models temporal patterns
    4. **Multi-loss** training: binary (real/fake) + vocoder type classification

    ### Reference
    > Sun et al., *AI-Synthesized Voice Detection Using Neural Vocoder Artifacts*, CVPRW 2023
    """)
