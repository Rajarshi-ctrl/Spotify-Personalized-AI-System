# Spotify: Personalized AI System 🎵🤖

An intelligent music recommendation engine and discovery platform powered by Flask, LangChain, Google Gemini, and Neo4j Graph Database. The system combines graph-based relationship modeling (artists, genres, audio features) with natural language understanding to deliver personalized song recommendations beyond standard collaborative filtering.

---

## 🎯 Problem Statement
Traditional music recommendation systems rely on basic collaborative filtering or static audio metadata, which fail to capture context-rich conversational prompts, multi-dimensional entity connections across artists, tracks, and genres, or complex listener intent.

## 💡 Solution Architecture
Built an end-to-end graph-powered personalized AI recommendation platform:
- **LLM Context Engine:** Implemented LangChain and Google Gemini to parse conversational queries into structured music filtering parameters.
- **Neo4j Knowledge Graph:** Modeled multi-relational graphs linking track metadata, audio features, genres, and artist relationships for graph traversal discovery.
- **Interactive Flask Interface:** Built a responsive web dashboard for query execution, recommendation browsing, and audio attribute exploration.

## 🚀 Impact Highlights
- **Data Analyst:** Built multi-attribute audio feature visualizations and track insights across genres and moods.
- **Data Scientist:** Solved cold-start and context-awareness constraints by integrating LLM intent extraction with graph embedding retrieval.
- **Data Engineer:** Engineered an end-to-end data ingestion pipeline connecting Spotify API endpoints to a structured Neo4j graph database schema.

---

## 🌟 Key Features
- **Context-Aware Recommendations:** Interprets natural language prompts using LLM chains.
- **Graph-Powered Relationship Mapping:** Leverages Neo4j to model complex connections across artists, tracks, moods, and musical genres.
- **Dynamic Web Interface:** Clean, responsive UI for exploring track suggestions, browsing metadata, and analyzing audio attributes.
