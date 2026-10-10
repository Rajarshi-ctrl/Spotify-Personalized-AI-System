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

# Comprehensive list of prominent Indian and Bengali / regional artists
REGIONAL_ARTISTS = [
    "arijit singh", "shreya ghoshal", "anupam roy", "kishore kumar",
    "hemant kumar", "hemanta mukherjee", "manna dey", "r. d. burman",
    "s. d. burman", "kumar sanu", "alka yagnik", "sonu nigam",
    "pritam", "shaan", "monali thakur", "jeet gannguli", "nachiketa",
    "rupankar", "lopamudra", "babul supriyo"
]

# ------------------------------------------------------------------------------
# 1. LIGHTWEIGHT CATALOG INGESTION (MEMORY < 60MB RAM)
# ------------------------------------------------------------------------------
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

    # Extract all Indian / Bollywood tracks
    indian_tracks = full_df[full_df["track_genre"].isin(["indian", "pop-film"])]
    
    # Sample balanced global tracks
    sample_size = min(2000, len(full_df))
    other_tracks = full_df[~full_df["track_genre"].isin(["indian", "pop-film"])].sample(
        n=sample_size, random_state=42
    )

    catalog = pd.concat([indian_tracks, other_tracks], ignore_index=True)

    # Immediately release unneeded raw dataset from memory
    del full_df
    gc.collect()

    def create_semantic_tag(row):
        artist_lower = str(row["artists"]).lower()
        is_regional = any(a in artist_lower for a in REGIONAL_ARTISTS)

        tags = []
        if row["track_genre"] in ["indian", "pop-film"]:
            tags.extend(["bollywood", "hindi", "indian soundtrack", "film song"])
        if is_regional:
            tags.extend(["bengali", "bangla", "regional melody", "romantic acoustic", "tollywood", "bangla gaan"])

        tag_str = " " + " ".join(tags) if tags else ""
        return f"{row['track_name']} by {row['artists']}. Genre: {row['track_genre']}.{tag_str}"

    catalog["semantic_tag"] = catalog.apply(create_semantic_tag, axis=1)
    return catalog

catalog_df = load_catalog()

# ------------------------------------------------------------------------------
# 2. ULTRA-LIGHTWEIGHT TF-IDF VECTOR ENGINE (~15MB RAM)
# ------------------------------------------------------------------------------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
catalog_embeddings = tfidf.fit_transform(catalog_df["semantic_tag"])

# ------------------------------------------------------------------------------
# 3. LANGCHAIN GROQ AI DJ SETUP
# ------------------------------------------------------------------------------
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
            "Analyze the user query, their past music taste, and the candidate tracks. "
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

# ------------------------------------------------------------------------------
# 4. RECOMMENDATION FUNCTIONS
# ------------------------------------------------------------------------------
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
        (catalog_df["energy"] <= thresholds["energy_max"])
    ]
    
    loc_clean = (location or "").lower()
    if any(k in loc_clean for k in ["india", "in", "bengal", "kolkata", "west bengal"]):
        loc_tracks = filtered[filtered["track_genre"].isin(["indian", "pop-film"])]
        if len(loc_tracks) >= 8:
            results = loc_tracks.sample(8)
        else:
            results = filtered.sample(min(8, len(filtered)))
    else:
        results = filtered.sample(min(8, len(filtered)))
        
    return time_label, results.to_dict(orient="records")

def get_warm_start_tracks(past_queries: list, location: str):
    query_corpus = " ".join(past_queries) if past_queries else "music"
    user_vec = tfidf.transform([query_corpus])
    sims = cosine_similarity(user_vec, catalog_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:6]
    history_tracks = catalog_df.iloc[top_indices].to_dict(orient="records")
    
    time_label, time_tracks = get_cold_start_tracks(location)
    return history_tracks, time_label, time_tracks[:4]

def search_catalog_with_ai(query: str, location: str, past_queries: list):
    q_lower = query.lower()
    is_regional_query = any(w in q_lower for w in ["bengali", "bangla", "tollywood", "rabindra", "bollywood", "hindi"])

    # 1. Lexical match (Exact title or artist match)
    exact = catalog_df[
        catalog_df["track_name"].str.contains(query, case=False, na=False) |
        catalog_df["artists"].str.contains(query, case=False, na=False)
    ]

    if len(exact) > 0 and not is_regional_query:
        matched = exact.head(6).to_dict(orient="records")
    else:
        # 2. Semantic query expansion for regional nuances
        enhanced_query = query
        if any(w in q_lower for w in ["bengali", "bangla"]):
            enhanced_query = f"{query} bengali bangla acoustic melody indian regional"
        elif any(w in q_lower for w in ["bollywood", "hindi"]):
            enhanced_query = f"{query} bollywood hindi soundtrack film"

        query_vec = tfidf.transform([enhanced_query])
        sims = cosine_similarity(query_vec, catalog_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:6]
        matched = catalog_df.iloc[top_indices].to_dict(orient="records")

    tracks_formatted = "\n".join([f"- '{t['track_name']}' by {t['artists']} ({t['track_genre']})" for t in matched])
    history_str = ", ".join(past_queries) if past_queries else "None"

    # 3. AI DJ Commentary via Groq
    if dj_chain:
        try:
            dj_commentary = dj_chain.invoke({
                "query": query,
                "location": location or "India",
                "history": history_str,
                "tracks": tracks_formatted
            })
        except Exception:
            dj_commentary = f"Curated selections matching '{query}' based on your recent musical taste."
    else:
        dj_commentary = f"Curated selections matching '{query}' based on your recent musical taste."

    return matched, dj_commentary
