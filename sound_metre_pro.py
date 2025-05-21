# sound_metre_pro.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import av
import matplotlib.pyplot as plt
import threading
import queue
import time
import scipy.signal
from datetime import datetime

# ---------------------------
# CONFIGURATION
# ---------------------------
import streamlit as st

# Page configuration
st.set_page_config(page_title="Sound Metre Pro", layout="wide")

# Styled title and content
st.markdown(
    """
    <h1 style='text-align: center; font-size: 65px;'>
        🎧🔊🎶 <b>SOUND METRE PRO</b> 🎶🔊🎧
    </h1>

    <h3 style='text-align: center; font-size: 28px;'>
        ✨🎛️ <b>High-Tech Real-Time Audio Analyzer</b> 🎛️✨
    </h3>

    <hr style='border-top: 2px solid #bbb;'>

    <blockquote style='text-align: center; font-size: 20px; color: #555;'>
        🧘‍♂️ “The quieter you become, the more you can hear.” – Ram Dass
    </blockquote>

    <blockquote style='text-align: center; font-size: 20px; color: #555;'>
        🎼 “Where words fail, music speaks.” – Hans Christian Andersen
    </blockquote>

    <blockquote style='text-align: center; font-size: 20px; color: #555;'>
        🎧 “Sound is the vocabulary of nature.” – Pierre Schaeffer
    </blockquote>

    <blockquote style='text-align: center; font-size: 20px; color: #555;'>
        🔊 “If you want to find the secrets of the universe, think in terms of energy, frequency and vibration.” – Nikola Tesla
    </blockquote>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# SESSION STATE INITIALIZATION
# ---------------------------
if 'db_level' not in st.session_state:
    st.session_state['db_level'] = 0.0
if 'audio_buffer' not in st.session_state:
    st.session_state['audio_buffer'] = np.zeros(1024)
if 'timestamp' not in st.session_state:
    st.session_state['timestamp'] = "--:--:--"

# ---------------------------
# AUDIO PROCESSING FUNCTIONS
# ---------------------------
def rms(samples):
    return np.sqrt(np.mean(np.square(samples)))

def rms_to_db(rms_val, ref=1.0):
    if rms_val <= 0:
        return -100.0
    return 20 * np.log10(rms_val / ref)

def apply_a_weighting(signal, rate):
    b, a = scipy.signal.butter(4 , [100 / (0.5 * rate), 5000 / (0.5 * rate)], btype='band')
    return scipy.signal.lfilter(b, a, signal)

# ---------------------------
# AUDIO PROCESSOR CLASS
# ---------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.sample_rate = 44100
        self.buffer = queue.Queue()
        self.lock = threading.Lock()
        self.audio_data = np.zeros(1024)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0  # Normalize 16-bit PCM
        with self.lock:
            self.audio_data = audio
        return frame

    def get_audio_data(self):
        with self.lock:
            return self.audio_data

# ---------------------------
# STREAMLIT WEBRTC STREAMER
# ---------------------------
webrtc_ctx = webrtc_streamer(
    key="sound-metere-pro",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# ---------------------------
# MAIN APPLICATION DISPLAY
# ---------------------------
if webrtc_ctx.audio_processor:
    audio_processor = webrtc_ctx.audio_processor

    # Sidebar Settings
    st.sidebar.header("⚙️ Settings")
    use_weighting = st.sidebar.checkbox("Apply A-Weighting", value=True)
    alert_threshold = st.sidebar.slider("Alert Threshold (dB)", 40, 120, 85)
    show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
    show_fft = st.sidebar.checkbox("Show Frequency Spectrum", value=True)

    # Real-time update container
    container = st.container()

    # Get latest audio data
    audio_data = audio_processor.get_audio_data()
    if use_weighting:
        audio_data = apply_a_weighting(audio_data, 44100)

    current_rms = rms(audio_data)
    current_db = rms_to_db(current_rms)
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Calculate extra parameters
    peak_amplitude = np.max(np.abs(audio_data))
    fft_vals = np.abs(np.fft.rfft(audio_data))
    fft_freqs = np.fft.rfftfreq(len(audio_data), 1 / 44100)
    dominant_freq_index = np.argmax(fft_vals)
    dominant_freq = fft_freqs[dominant_freq_index]
    period = 1 / dominant_freq if dominant_freq > 0 else 0

    # Save to session state
    st.session_state['db_level'] = current_db
    st.session_state['audio_buffer'] = audio_data
    st.session_state['timestamp'] = timestamp

    # Display metrics
    col1, col2, col3, col4 = container.columns(4)
    col1.metric("Current dB", f"{current_db:.2f} dB")
    col2.metric("Time", timestamp)
    col3.metric("Peak Amplitude", f"{peak_amplitude:.3f}")
    col4.metric("Dominant Freq (Hz)", f"{dominant_freq:.2f}")

    if current_db > alert_threshold:
        st.error("🚨 Alert: Sound is too loud!")
    else:
        st.success("✅ Safe Sound Level")

    st.markdown(f"**Period (s)**: `{period:.6f}`")

    # Visualizations
    if show_waveform:
        st.subheader("📉 Waveform")
        fig, ax = plt.subplots()
        ax.plot(audio_data, color='green')
        ax.set_title("Audio Waveform")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    if show_fft:
        st.subheader("🔬 Frequency Spectrum")
        fig2, ax2 = plt.subplots()
        ax2.semilogx(fft_freqs, fft_vals, color='purple')
        ax2.set_title("FFT Spectrum")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        st.pyplot(fig2)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("### ✨ Developed by **PRAISE ADEYEYE** ✨")
st.markdown("> \"Sound is the vocabulary of nature.\" – Pierre Schaeffer")
