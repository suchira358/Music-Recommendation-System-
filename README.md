**Music Mood Detection System** 🎵

This project is a simple machine learning application that predicts the **mood of a music file** based on its audio features. 
The model is trained using audio files categorized into moods such as **happy, sad, calm, and energetic**. During training, features like **MFCC (Mel-Frequency Cepstral Coefficients)** are extracted from the audio using **LibROSA**, and a classifier from **scikit-learn** (Random Forest) is trained to recognize patterns associated with different moods.

A web interface built with **Streamlit** allows users to upload a `.wav` music file, after which the system processes the audio and predicts its mood in real time. This project demonstrates the use of **machine learning, audio signal processing, and interactive web apps** for analyzing music emotions.
