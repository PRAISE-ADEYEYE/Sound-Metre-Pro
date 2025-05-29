import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import collections
import threading
from datetime import datetime
import pyaudio
import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import time

# Page config
st.set_page_config(page_title="Sound Metre Pro", layout="wide")

# Title
st.markdown(
    """
    <h1 style='text-align: center; font-size: 65px;'>🎧🔊🎶 <b>SOUND METRE PRO</b> 🎶🔊🎧</h1>
    <h3 style='text-align: center; font-size: 28px;'>✨🎛️ <b>High-Tech Real-Time Audio Analyzer</b> 🎛️✨</h3>
    <hr style='border-top: 2px solid #bbb;'>
    <blockquote style='text-align: center; font-size: 20px; color: #555;'>🧘‍♂️ “The quieter you become, the more you can hear.” – Ram Dass</blockquote>
    <blockquote style='text-align: center; font-size: 20px; color: #555;'>🎼 “Where words fail, music speaks.” – Hans Christian Andersen</blockquote>
    <blockquote style='text-align: center; font-size: 20px; color: #555;'>🎧 “Sound is the vocabulary of nature.” – Pierre Schaeffer</blockquote>
    <blockquote style='text-align: center; font-size: 20px; color: #555;'>🔊 “If you want to find the secrets of the universe, think in terms of energy, frequency and vibration.” – Nikola Tesla</blockquote>
    """,
    unsafe_allow_html=True,
)

# Initialize session states
for key in ['db_history', 'time_history', 'mean_amp_history', 'peak_amp_history', 'freq_history', 'period_history']:
    if key not in st.session_state:
        st.session_state[key] = collections.deque(maxlen=300)


# Audio helpers
def rms(samples):
    return np.sqrt(np.mean(np.square(samples)))


def rms_to_db(rms_val, ref=1.0):
    return 20 * np.log10(rms_val / ref) if rms_val > 1e-10 else 0.0


def apply_a_weighting(signal, rate):
    b, a = scipy.signal.butter(4, [100 / (0.5 * rate), 5000 / (0.5 * rate)], btype='band')
    return scipy.signal.lfilter(b, a, signal)


# Audio processor class
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.sample_rate = 44100
        self.lock = threading.Lock()
        self.audio_data = np.zeros(1024)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        with self.lock:
            self.audio_data = audio
        return frame

    def get_audio_data(self):
        with self.lock:
            return self.audio_data


# Start stream
audio_processor_instance = AudioProcessor()
webrtc_ctx = webrtc_streamer(
    key="sound-metre-pro",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=lambda: audio_processor_instance,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
use_weighting = st.sidebar.checkbox("Apply A-Weighting", value=True)
alert_threshold = st.sidebar.slider("Alert Threshold (dB)", 30, 120, 85)
show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
show_fft = st.sidebar.checkbox("Show Frequency Spectrum", value=True)
show_trend = st.sidebar.checkbox("Show dB Level Over Time", value=True)

if webrtc_ctx.audio_processor:
    audio_processor = webrtc_ctx.audio_processor

    # Placeholder for updating elements
    placeholder = st.empty()

    if webrtc_ctx.state.playing:
        st.session_state.running = True
        while webrtc_ctx.state.playing and st.session_state.running:
            audio_data = audio_processor.get_audio_data()
            if len(audio_data) == 0:
                time.sleep(1)
                continue

            if use_weighting:
                audio_data = apply_a_weighting(audio_data, 44100)

            current_rms = rms(audio_data)
            current_db = rms_to_db(current_rms)
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # FFT Analysis
            windowed = audio_data * np.hanning(len(audio_data))
            fft_vals = np.abs(np.fft.rfft(windowed))
            fft_freqs = np.fft.rfftfreq(len(windowed), 1 / 44100)
            dominant_freq = fft_freqs[np.argmax(fft_vals)] if fft_vals.size > 0 else 0
            period = 1 / dominant_freq if dominant_freq > 0 else 0
            peak_amplitude = np.max(np.abs(audio_data))

            # Update histories
            st.session_state.db_history.append(current_db)
            st.session_state.time_history.append(timestamp)
            st.session_state.mean_amp_history.append(np.mean(np.abs(audio_data)))
            st.session_state.peak_amp_history.append(peak_amplitude)
            st.session_state.freq_history.append(dominant_freq)
            st.session_state.period_history.append(period)

            # Display metrics and visuals
            with placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current dB", f"{current_db:.2f} dB")
                col2.metric("Time", timestamp)
                col3.metric("Peak Amplitude", f"{peak_amplitude:.3f}")
                col4.metric("Dominant Freq (Hz)", f"{dominant_freq:.2f}")

                st.markdown(f"**Period (s):** `{period:.6f}`")
                if current_db > alert_threshold:
                    st.error("🚨 Alert: Sound is too loud!")
                else:
                    st.success("✅ Safe Sound Level")

                if show_waveform:
                    st.subheader("📉 Waveform")
                    fig, ax = plt.subplots()
                    ax.plot(audio_data, color='green')
                    ax.set_title("Audio Waveform")
                    st.pyplot(fig)

                if show_fft:
                    st.subheader("🔬 Frequency Spectrum")
                    fig2, ax2 = plt.subplots()
                    ax2.semilogx(fft_freqs, fft_vals, color='purple')
                    ax2.set_title("FFT Spectrum")
                    st.pyplot(fig2)

                if show_trend:
                    st.subheader("📊 dB Level Over Time (~30 sec)")
                    df = pd.DataFrame({
                        'Time': list(st.session_state.time_history),
                        'dB Level': list(st.session_state.db_history)
                    })
                    df.index = pd.to_datetime(df['Time'], format="%H:%M:%S.%f")
                    st.line_chart(df['dB Level'])
            time.sleep(0.1)

    # Data export section after stream ends or while running
    # 📥 Export Audio Log
    st.markdown("### 📥 Export Audio Log")

    # Convert to DataFrame
    export_df = pd.DataFrame({
        'Timestamp': list(st.session_state.time_history),
        'dB Level': list(st.session_state.db_history),
        'Mean Amplitude': list(st.session_state.mean_amp_history),
        'Peak Amplitude': list(st.session_state.peak_amp_history),
        'Dominant Frequency': list(st.session_state.freq_history),
        'Period': list(st.session_state.period_history)
    })

    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="sound_meter_full_log.csv",
        mime='text/csv'
    )
