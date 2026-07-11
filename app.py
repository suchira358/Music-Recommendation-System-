import streamlit as st
from recommender import MusicRecommender


st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 AI Music Recommendation System")

st.write("Enter a song name to get similar song recommendations.")


@st.cache_resource
def load_model():
    return MusicRecommender("spotify_songs.csv")

recommender = load_model()


song_name = st.text_input("Enter Song Name")

if st.button("Recommend Songs"):

    if song_name.strip() == "":
        st.warning("Please enter a song name.")

    else:
        recommendations = recommender.recommend(song_name)

        if recommendations is None:
            st.error("Song not found!")

        else:
            st.success("Top 5 Recommendations")

            for i, song in enumerate(recommendations, start=1):
                st.write(f"### {i}. {song['Song']}")
                st.write(f"**Artist:** {song['Artist']}")
                st.write(f"**Genre:** {song['Genre']}")
                st.write("---")
