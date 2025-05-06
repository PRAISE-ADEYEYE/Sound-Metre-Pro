# SOUND METRE PRO - Streamlit-Based Real-Time Sound Meter

import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import pandas as pd
import wave
import os
import time
import threading
import scipy.signal
from datetime import datetime
import queue

# Page Configuration
st.set_page_config(page_title="Sound Metre Pro", layout="wide")
st.markdown("""
    <style>
        .reportview-container { background: #f4f4f4; }
        .sidebar .sidebar-content { background: #d1e7dd; }
        h1, h5 { text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🔊 SOUND METRE PRO")
st.subheader("High-Tech Real-Time Audio Analyzer")

# Constants
SAMPLE_RATE = 44100
CHANNELS = 1
FRAME_DURATION = 0.5
BUFFER_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

# File and folder setup
os.makedirs("recordings", exist_ok=True)

# ---------------------------
# SESSION STATE INITIALIZATION
# ---------------------------
for key, val in {
    "recording": b"",
    "stream_thread": None,
    "log_buffer": [],
    "latest_audio": np.zeros(BUFFER_SIZE),
    "current_db": 0.0,
    "current_timestamp": "--:--:--",
    "running": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

audio_queue = queue.Queue()


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
    b, a = scipy.signal.butter(4, [100 / (0.5 * rate), 5000 / (0.5 * rate)], btype='band')
    return scipy.signal.lfilter(b, a, signal)


# ---------------------------
# SIDEBAR CONFIGURATION
# ---------------------------
st.sidebar.header("⚙️ Settings")
use_weighting = st.sidebar.checkbox("Apply A-Weighting", value=True)
alert_threshold = st.sidebar.slider("Alert Threshold (dB)", 40, 120, 85)
record_audio = st.sidebar.checkbox("Record Audio", value=False)
show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
show_fft = st.sidebar.checkbox("Show Frequency Spectrum", value=True)
log_data = st.sidebar.checkbox("Log dB Data", value=False)

# ---------------------------
# AUDIO CALLBACK
# ---------------------------
def audio_callback(indata, frames, time_info, status):
    audio_data = indata[:, 0]
    if use_weighting:
        audio_data = apply_a_weighting(audio_data, SAMPLE_RATE)

    audio_queue.put(audio_data)


# ---------------------------
# STREAM HANDLER THREAD
# ---------------------------
def audio_stream_thread():
    with sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE):
        while st.session_state.running:
            time.sleep(FRAME_DURATION)


# ---------------------------
# START AUDIO STREAM
# ---------------------------
if not st.session_state.stream_thread or not st.session_state.running:
    if st.button("▶️ Start Monitoring"):
        st.session_state.running = True
        st.session_state.stream_thread = threading.Thread(target=audio_stream_thread, daemon=True)
        st.session_state.stream_thread.start()

# ---------------------------
# STOP STREAM
# ---------------------------
if st.session_state.running and st.button("⏹️ Stop Monitoring"):
    st.session_state.running = False
    time.sleep(0.5)
    st.rerun()


# ---------------------------
# DATA PROCESSING LOOP
# ---------------------------
if st.session_state.running and not audio_queue.empty():
    audio = audio_queue.get()
    current_rms = rms(audio)
    current_db = rms_to_db(current_rms)
    timestamp = datetime.now().strftime("%H:%M:%S")

    st.session_state.latest_audio = audio
    st.session_state.current_db = current_db
    st.session_state.current_timestamp = timestamp

    if log_data:
        st.session_state.log_buffer.append({"Timestamp": timestamp, "dB Level": current_db})

    if record_audio:
        st.session_state.recording += audio.astype(np.float32).tobytes()


# ---------------------------
# DISPLAY METRICS
# ---------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Current dB", f"{st.session_state.current_db:.2f} dB")
col2.metric("Time", st.session_state.current_timestamp)
if st.session_state.current_db > alert_threshold:
    col3.error("🚨 Too Loud!")
else:
    col3.success("✅ Safe Level")


# ---------------------------
# VISUALIZATIONS
# ---------------------------
if show_waveform:
    st.subheader("📉 Waveform")
    fig, ax = plt.subplots()
    ax.plot(st.session_state.latest_audio, color='green')
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

if show_fft:
    st.subheader("🔬 Frequency Spectrum")
    fft_vals = np.abs(np.fft.rfft(st.session_state.latest_audio))
    fft_freqs = np.fft.rfftfreq(len(st.session_state.latest_audio), 1 / SAMPLE_RATE)
    fig2, ax2 = plt.subplots()
    ax2.semilogx(fft_freqs, fft_vals, color='purple')
    ax2.set_title("FFT Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    st.pyplot(fig2)


# ---------------------------
# LOGGING AND EXPORT
# ---------------------------
if log_data and st.session_state.log_buffer:
    log_df = pd.DataFrame(st.session_state.log_buffer)
    st.subheader("📊 dB Log")
    st.dataframe(log_df.tail(20))
    csv_data = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Log CSV", csv_data, file_name="sound_log.csv")


# ---------------------------
# AUDIO RECORDING DOWNLOAD
# ---------------------------
if record_audio and st.session_state.recording:
    if st.button("💾 Save Recording"):
        wav_filename = f"recordings/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with wave.open(wav_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(st.session_state.recording)
        st.success(f"Recording saved: {wav_filename}")
        st.audio(wav_filename)


# ---------------------------
# STATS + ANALYSIS
# ---------------------------
if st.session_state.log_buffer:
    db_vals = [entry['dB Level'] for entry in st.session_state.log_buffer]
    st.markdown("---")
    st.subheader("📈 Analysis")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Max dB", f"{np.max(db_vals):.2f}")
    col_b.metric("Min dB", f"{np.min(db_vals):.2f}")
    col_c.metric("Average dB", f"{np.mean(db_vals):.2f}")


# ---------------------------
# RESET + HELP
# ---------------------------
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

with st.expander("❓ Help / About"):
    st.markdown("""
    **Sound Metre Pro** is a full-featured sound analysis tool.

    - Real-time dB measurement
    - Frequency spectrum & waveform plot
    - Optional audio recording & CSV logging
    - Built with ❤️ by Praise Adeyeye using Streamlit
    """)
