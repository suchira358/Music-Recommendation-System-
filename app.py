import streamlit as st
from recommender import MusicRecommender


st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵"
)

st.title("🎵 Music Recommendation System")

st.write(
    "Enter a song"
)


@st.cache_resource
def load_recommender():
    return MusicRecommender("spotify_songs.csv")


recommender = load_recommender()


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

            for i, song in enumerate(recommendations, 1):

                st.write(f"### {i}. {song['Song']}")
                st.write(f"**Artist:** {song['Artist']}")
                st.write(f"**Genre:** {song['Genre']}")
                st.write(f"**Mood:** {song['Mood']}")
                st.write("---")
