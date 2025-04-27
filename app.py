import streamlit as st
import librosa
import numpy as np

st.title("TB Prediction App (Audio Upload)")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Load the audio file
    y, sr = librosa.load(uploaded_file, sr=None)  # sr=None keeps original sampling rate

    st.success(f"Audio loaded successfully! Sample rate = {sr}, Length = {len(y)} samples.")

    # (Optional) Show audio stats
    st.write(f"Duration: {len(y)/sr:.2f} seconds")

    # Here you can add your prediction model
    # For example:
    # prediction = your_model.predict(y)
    # st.write(f"Prediction: {prediction}")

else:
    st.info("Please upload an audio file.")

