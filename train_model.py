# train_model.py
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Path to your dataset
# Dataset structure: data/happy/*.wav, data/sad/*.wav, etc.
DATASET_PATH = 'data/'
moods = ['happy', 'sad', 'calm', 'energetic']

features = []
labels = []

print("Extracting features from audio files...")

for mood in moods:
    folder = os.path.join(DATASET_PATH, mood)
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            file_path = os.path.join(folder, file)
            y, sr = librosa.load(file_path, duration=30)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            features.append(mfccs_mean)
            labels.append(mood)

X = np.array(features)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open('music_mood_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as music_mood_model.pkl")
