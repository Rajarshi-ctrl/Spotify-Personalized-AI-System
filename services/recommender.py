import os
import gc
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DATASET_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
LOCAL_CSV = "data/spotify_tracks.csv"

REGIONAL_ARTISTS = [
    "arijit singh", "shreya ghoshal", "anupam roy", "kishore kumar",
    "hemant kumar", "manna dey", "r. d. burman", "kumar sanu", 
    "alka yagnik", "sonu nigam", "pritam", "shaan", "monali thakur"
]

# 1. Balanced Catalog Ingestion
def load_catalog():
    needed_cols = ["track_name", "artists", "track_id", "track_genre", "energy"]
    
    if os.path.exists(LOCAL_CSV) and os.path.getsize(LOCAL_CSV) > 0:
        full_df = pd.read_csv(LOCAL_CSV, usecols=needed_cols)
    else:
        os.makedirs("data", exist_ok=True)
        full_df = pd.read_csv(DATASET_URL, usecols=needed_cols)
        full_df.to_csv(LOCAL_CSV, index=False)

    full_df = full_df.dropna(subset=["track_name", "artists", "track_id", "track_genre"])
    full_df = full_df.drop_duplicates(subset=["track_name", "artists"])

    # Extract all Indian/Bollywood songs + sample balanced global tracks
    indian_tracks = full_df[full_df["track_genre"].isin(["indian", "pop-film"])]
    other_tracks = full_df[~full_df["track_genre"].isin(["indian", "pop-film"])].sample(
        n=min(2000, len(full_df)), random_state=42
    )

    catalog = pd.concat([indian_tracks, other_tracks], ignore_index=True)

    del full_df
    gc.collect()

    def create_semantic_tag(row):
        artist_lower = str(row["artists"]).lower()
        is_regional = any(a in artist_lower for a in REGIONAL_ARTISTS)

        tags = []
        if row["track_genre"] in ["indian", "pop-film"]:
            tags.extend(["bollywood", "hindi", "indian soundtrack"])
        if is_regional:
            tags.extend(["bengali", "bangla", "regional melody", "romantic acoustic"])

        tag_str = " " + " ".join(tags) if tags else ""
        return f"{row['track_name']} by {row['artists']}. Genre: {row['track_genre']}.{tag_str}"

    catalog["semantic_tag"] = catalog.apply(create_semantic_tag, axis=1)
    return catalog

catalog_df = load_catalog()

# 2. Ultra-Lightweight TF-IDF Vectorizer (Takes ~15MB RAM instead of 500MB)
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
catalog_embeddings = tfidf.fit_transform(catalog_df["semantic_tag"])

# 3. LangChain Groq AI DJ
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.7
    )
    dj_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Spotify AI DJ. "
            "Analyze the user query, their past music taste, and the retrieved tracks. "
            "Provide a concise, engaging 2-sentence explanation of why these tracks match their vibe."
        )),
        ("user", (
            "Query: {query}\n"
            "User Location: {location}\n"
            "Recent Taste History: {history}\n"
            "Retrieved Tracks:\n{tracks}\n\n"
            "DJ Curation Note:"
        ))
    ])
    dj_chain = dj_prompt | llm | StrOutputParser()
else:
    dj_chain = None

# 4. Recommendation Helper Functions
def get_time_profile():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Morning", {"energy_min": 0.35, "energy_max": 0.70}
    elif 12 <= hour < 17:
        return "Afternoon", {"energy_min": 0.55, "energy_max": 1.00}
    elif 17 <= hour < 22:
        return "Evening", {"energy_min": 0.40, "energy_max": 0.80}
    else:
        return "Late Night", {"energy_min": 0.00, "energy_max": 0.50}

def get_cold_start_tracks(location: str):
    time_label, thresholds = get_time_profile()
    filtered = catalog_df[
        (catalog_df["energy"] >= thresholds["energy_min"]) &