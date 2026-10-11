# Spotify: Personalized AI System 🎵🤖

An intelligent music recommendation engine and discovery platform powered by Flask, LangChain, Google Gemini, and a Neo4j Graph Database. The system combines graph-based relationship modeling (artists, genres, audio features) with natural language understanding to deliver personalized song recommendations beyond standard collaborative filtering.

**[🚀 Live Demo](https://spotify-personalized-ai-system.onrender.com/)** | **[📽️ Project Overview & Showcase / Video Walkthrough](https://drive.google.com/file/d/1eEpVQ2bDar9X4PCb9WyueTsTWklahmNJ/view?usp=sharing)**

---

## 🌟 Key Features

- **Context-Aware Recommendations:** Interprets natural language prompts using LLM chains.
- **Graph-Powered Relationship Mapping:** Leverages Neo4j to model complex connections across artists, tracks, moods, and musical genres.
- **Dynamic Web Interface:** Clean, responsive UI for exploring track suggestions, browsing metadata, and analyzing audio attributes.
- **Modular Flask Architecture:** Designed with scalable blueprints, dedicated service layers, and separated schema definitions.

---

## 🏗️ Architecture & Tech Stack

- **Backend:** Flask, Python 3.10+
- **Database:** Neo4j (Graph DB for music relationships)
- **AI / LLM:** LangChain, Google Gemini API
- **Frontend:** HTML5, CSS3, JavaScript, Jinja2
- **Deployment:** Render

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Neo4j Desktop or Neo4j AuraDB instance
- Google Gemini API Key

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rajarshi-ctrl/Spotify-Personalized-AI-System.git
   cd Spotify-Personalized-AI-System
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   GEMINI_API_KEY=your_gemini_api_key
   SECRET_KEY=your_flask_secret_key
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
