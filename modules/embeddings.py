import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_embedding_function = None


def ensure_embedding_model() -> None:
    """Pre-download the embedding model with a longer timeout before ChromaDB uses it."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    from sentence_transformers import SentenceTransformer

    print(f"📥 Loading embedding model: {EMBEDDING_MODEL}")
    SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Embedding model ready")


def get_embedding_function():
    """Return a shared ChromaDB embedding function for build and query."""
    global _embedding_function

    if _embedding_function is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
        try:
            # Cached copy first: no network call, so a slow/unreachable Hugging Face can't hang the app
            _embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL, local_files_only=True
            )
        except Exception:
            print(f"📥 Embedding model not cached yet, downloading {EMBEDDING_MODEL}...")
            _embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    return _embedding_function
