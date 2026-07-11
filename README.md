#  AI Music Recommendation System

## Overview

A content-based music recommendation system that suggests songs similar to a user's selected track using Spotify audio features and cosine similarity. The application is built with Streamlit for an interactive user experience.

## Features

* Search songs by name
* Recommend Top 5 similar songs
* Content-based recommendation
* Interactive Streamlit interface

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

## Machine Learning

* Feature Scaling (StandardScaler)
* Cosine Similarity
* Content-Based Recommendation

## Project Structure

```
music-recommendation-system/
│
├── app.py
├── recommender.py
├── spotify_songs.csv
└── README.md
```

## Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

