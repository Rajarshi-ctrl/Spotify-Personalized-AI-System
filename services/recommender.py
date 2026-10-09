import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DATASET_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
LOCAL_CSV = "data/spotify_tracks.csv"

# 1. Initialize Vector Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Balanced Catalog Ingestion (All Bollywood + Global Sample)
def load_catalog():
    if os.path.exists(LOCAL_CSV) and os.path.getsize(LOCAL_CSV) > 0:
        print("Loading cached catalog from local CSV...")
        full_df = pd.read_csv(LOCAL_CSV)
    else:
        print("Downloading Spotify dataset from Hugging Face...")
        os.makedirs("data", exist_ok=True)
        full_df = pd.read_csv(DATASET_URL)
        full_df.to_csv(LOCAL_CSV, index=False)

    full_df = full_df.dropna(subset=["track_name", "artists", "track_id", "track_genre"])
    full_df = full_df.drop_duplicates(subset=["track_name", "artists"])

    # Extract all Bollywood / Indian regional tracks
    indian_tracks = full_df[full_df["track_genre"].isin(["indian", "pop-film"])]
    
    # Sample balanced global genres
    other_tracks = full_df[~full_df["track_genre"].isin(["indian", "pop-film"])].sample(
        n=6000, random_state=42
    )

    catalog = pd.concat([indian_tracks, other_tracks], ignore_index=True)

    def create_semantic_tag(row):
        extra = " bollywood hindi indian film soundtrack" if row["track_genre"] in ["indian", "pop-film"] else ""
        return f"{row['track_name']} by {row['artists']}. Genre: {row['track_genre']}.{extra}"

    catalog["semantic_tag"] = catalog.apply(create_semantic_tag, axis=1)
    print(f"Catalog ready: {len(catalog)} tracks loaded.")
    return catalog

catalog_df = load_catalog()
catalog_embeddings = embedder.encode(catalog_df["semantic_tag"].tolist(), show_progress_bar=False)

# 3. LangChain Groq LLM Setup
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
            "Provide a concise, engaging 2-sentence explanation of why these tracks "
            "were curated for their vibe."
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
        (catalog_df["energy"] <= thresholds["energy_max"])
    ]
    
    if location.lower() in ["india", "in"]:
        loc_tracks = filtered[filtered["track_genre"].isin(["indian", "pop-film"])]
        results = loc_tracks.sample(8) if len(loc_tracks) >= 8 else filtered.sample(8)
    else:
        results = filtered.sample(8)
        
    return time_label, results.to_dict(orient="records")

def get_warm_start_tracks(past_queries: list, location: str):
    query_corpus = " ".join(past_queries) if past_queries else "music"
    user_vec = embedder.encode([query_corpus])[0]
    sims = cosine_similarity([user_vec], catalog_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:6]
    history_tracks = catalog_df.iloc[top_indices].to_dict(orient="records")
    
    time_label, time_tracks = get_cold_start_tracks(location)
    return history_tracks, time_label, time_tracks[:4]

def search_catalog_with_ai(query: str, location: str, past_queries: list):
    # Lexical match (Exact title or artist match)
    exact = catalog_df[
        catalog_df["track_name"].str.contains(query, case=False, na=False) |
        catalog_df["artists"].str.contains(query, case=False, na=False)
    ]
    
    if len(exact) > 0:
        matched = exact.head(6).to_dict(orient="records")
    else:
        query_vec = embedder.encode([query])[0]
        sims = cosine_similarity([query_vec], catalog_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:6]
        matched = catalog_df.iloc[top_indices].to_dict(orient="records")

    tracks_formatted = "\n".join([f"- '{t['track_name']}' by {t['artists']} ({t['track_genre']})" for t in matched])
    history_str = ", ".join(past_queries) if past_queries else "None"

    # Generate commentary via Groq
    if dj_chain:
        try:
            dj_commentary = dj_chain.invoke({
                "query": query,
                "location": location,
                "history": history_str,
                "tracks": tracks_formatted
            })
        except Exception:
            dj_commentary = f"Curated selections matching '{query}' based on your recent musical taste."
    else:
        dj_commentary = f"Curated selections matching '{query}' based on your recent musical taste."

    return matched, dj_commentary