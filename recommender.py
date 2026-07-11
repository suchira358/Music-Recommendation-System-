import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class MusicRecommender:
    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.features = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo"
        ]

   
        self.df = self.df.dropna(subset=self.features)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.df[self.features])

        self.similarity = cosine_similarity(scaled_features)

    def recommend(self, song_name, top_n=5):
  
        song = self.df[
            self.df["track_name"].str.lower() == song_name.lower()
        ]

        if song.empty:
            return None

        index = song.index[0]

        scores = list(enumerate(self.similarity[index]))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for i in scores[1:top_n + 1]:
            row = self.df.iloc[i[0]]

            recommendations.append({
                "Song": row["track_name"],
                "Artist": row["artists"],
                "Genre": row.get("track_genre", "Unknown")
            })

        return recommendations
