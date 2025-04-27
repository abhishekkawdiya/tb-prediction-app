import streamlit as st
import torchaudio
from transformers import AutoProcessor, AutoModel
import torch

st.title("TB Probability Predictor from Cough Audio")

st.write("Upload a WAV audio file (cough recording) and get the probability of TB.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    processor = AutoProcessor.from_pretrained("google/hear")
    model = AutoModel.from_pretrained("google/hear")

    waveform, sample_rate = torchaudio.load(uploaded_file)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state

    tb_probability = torch.sigmoid(torch.mean(embeddings)).item()

    st.metric(label="TB Probability", value=f"{tb_probability:.2f}")
