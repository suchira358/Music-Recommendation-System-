import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class MusicRecommender:

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.features = [
            "danceability", "energy", "loudness",
            "speechiness", "acousticness",
            "instrumentalness", "liveness",
            "valence", "tempo"
        ]

        self.df = self.df.dropna(subset=self.features)

        scaler = StandardScaler()
        features = scaler.fit_transform(self.df[self.features])

        self.similarity = cosine_similarity(features)


    def recommend(self, song_name, top_n=5):

        songs = self.df[
            self.df["track_name"].str.lower() == song_name.lower()
        ]

        if songs.empty:
            return None

        index = songs.index[0]
        mood = self.df.loc[index, "mood"]

        mood_songs = self.df[
            self.df["mood"].str.lower() == mood.lower()
        ]

        scores = []

        for i in mood_songs.index:

            if i != index:
                scores.append((i, self.similarity[index][i]))

        scores.sort(key=lambda x: x[1], reverse=True)

        recommendations = []

        for i, score in scores[:top_n]:

            row = self.df.loc[i]

            recommendations.append({
                "Song": row["track_name"],
                "Artist": row["artists"],
                "Genre": row["track_genre"],
                "Mood": row["mood"]
            })

        return recommendations
