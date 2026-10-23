"""Rebuild the ChromaDB index from data/tickets.csv and data/corrections.json.
The app also rebuilds it automatically when tickets.csv changes. Run: python build_vectordb.py"""
from modules.embeddings import ensure_embedding_model
from modules.vector_store import build_index

if __name__ == "__main__":
    ensure_embedding_model()
    build_index()
