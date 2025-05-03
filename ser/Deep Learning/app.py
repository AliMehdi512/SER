import streamlit as st
import numpy as np
import librosa
import keras
import tempfile
from pydub import AudioSegment
import os
import time
import simpleaudio as sa  # ✅ Add simpleaudio

# Initialize session state
if "predicting" not in st.session_state:
    st.session_state.predicting = False
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "sr" not in st.session_state:
    st.session_state.sr = None
if "model" not in st.session_state:
    st.session_state.model = None
if "interval" not in st.session_state:
    st.session_state.interval = 3
if "play_obj" not in st.session_state:
    st.session_state.play_obj = None  # ✅ Track audio playback

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Convert numeric class to emotion label
def convert_class_to_emotion(pred):
    label_conversion = {
        '0': 'neutral', '1': 'calm', '2': 'happy', '3': 'sad',
        '4': 'angry', '5': 'fearful', '6': 'disgust', '7': 'surprised'
    }
    return label_conversion.get(str(pred), "Unknown")

# Live prediction function
def predict_emotions_live(audio_data, sr, model, interval_sec=3, start_index=0):
    interval_samples = int(interval_sec * sr)
    total_samples = len(audio_data)

    for i in range(start_index, total_samples, interval_samples):
        if not st.session_state.predicting:
            break

        segment = audio_data[i:i + interval_samples]
        if len(segment) < interval_samples:
            segment = np.pad(segment, (0, interval_samples - len(segment)))

        mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40).T, axis=0)
        x = np.expand_dims(np.expand_dims(mfccs, axis=1), axis=0)

        pred = model.predict(x, verbose=0)
        label = convert_class_to_emotion(np.argmax(pred))

        start = i / sr
        end = min((i + interval_samples) / sr, total_samples / sr)
        yield f"🎯 Predicted [{start:.2f}s - {end:.2f}s]: **{label}**", i + interval_samples

        time.sleep(interval_sec)

# App UI
st.title("🎧 Real-Time Emotion Recognition with Resume")

audio_file = st.file_uploader("📁 Upload your audio file")
model_path = st.text_input("🧠 Enter path to your trained model (.h5)", "testing-16-200_model.h5")
st.session_state.interval = st.slider("⏱️ Prediction Interval (seconds)", 1, 10, 3)

if audio_file and model_path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        try:
            if audio_file.name.endswith(".mp3"):
                sound = AudioSegment.from_mp3(audio_file)
                sound.export(tmp.name, format="wav")
            else:
                tmp.write(audio_file.read())
            audio_bytes = open(tmp.name, 'rb').read()
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
            st.stop()

    try:
        st.session_state.audio_data, st.session_state.sr = librosa.load(tmp.name)
        st.success(f"🎵 Audio loaded (Sample Rate: {st.session_state.sr})")
    except Exception as e:
        st.error(f"Failed to load audio: {e}")
        os.remove(tmp.name)
        st.stop()

    if st.session_state.model is None:
        st.session_state.model = load_model(model_path)

    if st.session_state.model is None:
        st.stop()

    result_area = st.empty()
    col1, col2 = st.columns([1, 1])

    # ▶ Start/Resume Prediction + Audio
    if col1.button("▶ Start/Resume Prediction"):
        st.session_state.predicting = True

        # ✅ Start audio from current index
        segment_start_sec = st.session_state.current_index / st.session_state.sr
        sound = AudioSegment.from_file(tmp.name)
        trimmed_sound = sound[segment_start_sec * 1000:]  # convert sec to ms

        trimmed_path = os.path.join(tempfile.gettempdir(), "trimmed_playback.wav")
        trimmed_sound.export(trimmed_path, format="wav")

        try:
            wave_obj = sa.WaveObject.from_wave_file(trimmed_path)
            st.session_state.play_obj = wave_obj.play()
        except Exception as e:
            st.error(f"🔇 Audio playback failed: {e}")
            # Optionally fallback to Streamlit's audio player
            st.audio(trimmed_path)

        result_area.write("🕒 Running real-time prediction...")
        for prediction, next_index in predict_emotions_live(
            st.session_state.audio_data,
            st.session_state.sr,
            st.session_state.model,
            interval_sec=st.session_state.interval,
            start_index=st.session_state.current_index,
        ):
            result_area.markdown(prediction)
            st.session_state.current_index = next_index

    # ⏹ Stop Prediction + Audio
    if col2.button("⏹ Stop Prediction"):
        st.session_state.predicting = False
        if st.session_state.play_obj:
            st.session_state.play_obj.stop()  # ✅ Stop audio
        result_area.info("⛔ Prediction paused.")

    os.remove(tmp.name)
