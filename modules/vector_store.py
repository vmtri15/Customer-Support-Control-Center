import chromadb

from modules.embeddings import get_embedding_function

COLLECTION_NAME = "ticket_knowledge_base"
DEFAULT_DB_PATH = "./chroma_db"


def get_chroma_client(path: str = DEFAULT_DB_PATH) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=path)


def get_ticket_collection(
    client: chromadb.PersistentClient,
    recreate: bool = False,
) -> chromadb.Collection:
    if recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )
