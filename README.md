# Spotify: Personalized AI System 🎵🤖

An intelligent music recommendation engine and discovery platform powered by Flask, LangChain, Google Gemini, and a Neo4j Graph Database. The system combines graph-based relationship modeling (artists, genres, audio features) with natural language understanding to deliver personalized song recommendations beyond standard collaborative filtering.

---

## 🌟 Key Features

- **Context-Aware Recommendations:** Interprets natural language prompts using LLM chains.
- **Graph-Powered Relationship Mapping:** Leverages Neo4j to model complex connections across artists, tracks, moods, and musical genres.
- **Dynamic Web Interface:** Clean, responsive UI for exploring track suggestions, browsing metadata, and analyzing audio attributes.
- **Modular Flask Architecture:** Designed with scalable blueprints, dedicated service layers, and separated schema definitions.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, Flask
- **AI & Orchestration:** Google Gemini API, LangChain
- **Graph Database:** Neo4j (Cypher query language)
- **Frontend:** Jinja2 Templates, HTML5, CSS3, JavaScript
- **Data & APIs:** Spotify Web API / Music Audio Features Dataset

---

## 📂 Project Structure

```text
Spotify-Personalized-AI-System/
├── blueprints/          # Modular route controllers
├── services/            # Business logic (Neo4j graph queries, Gemini LLM chain, Spotify API)
├── templates/           # Frontend HTML templates (Jinja2)
├── .gitignore           # Git ignore rules
├── app.py               # Application entry point and server initialization
├── models.py            # Graph and data entity models
└── requirements.txt     # Python project dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Neo4j Desktop or Neo4j AuraDB instance
- Google Gemini API Key
- Spotify Developer Credentials (Client ID & Secret)

---

### Installation

1. Clone the repository and enter the project folder.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your configuration:
   ```env
   FLASK_APP=app.py
   FLASK_ENV=development
   SECRET_KEY=your_secret_key
   NEO4J_URI=your_neo4j_bolt_uri
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   GEMINI_API_KEY=your_gemini_api_key
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   ```

---

### Running the Application

1. Ensure your Neo4j database instance is running.
2. Start the development server:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to `localhost:5000`.

---

## 📄 License

This project is licensed under the MIT License.
