# app.py
import streamlit as st
import librosa
import numpy as np
import pickle

# Load trained model
with open('music_mood_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎵 Music Mood Detection")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    y, sr = librosa.load(uploaded_file, duration=30)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    prediction = model.predict([mfccs])
    
    st.success(f"Predicted Mood: {prediction[0].capitalize()}")
