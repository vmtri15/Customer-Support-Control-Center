import hashlib
import threading
from pathlib import Path

import chromadb
import pandas as pd

from modules.embeddings import get_embedding_function

COLLECTION_NAME = "tickets"
LEGACY_COLLECTION_NAME = "ticket_knowledge_base"
DEFAULT_DB_PATH = "./chroma_db"
TICKETS_PATH = "data/tickets.csv"
CORRECTIONS_PATH = "data/corrections.json"
FINGERPRINT_FILE = "tickets_fingerprint.txt"  # index được xây từ phiên bản tickets.csv nào
BATCH_SIZE = 1000

_index = None
_index_lock = threading.Lock()


def get_chroma_client(path: str = DEFAULT_DB_PATH) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=path)


def get_ticket_collection(
    client: chromadb.PersistentClient,
    recreate: bool = False,
) -> chromadb.Collection:
    if recreate:
        for name in (COLLECTION_NAME, LEGACY_COLLECTION_NAME):
            try:
                client.delete_collection(name)
            except Exception:  # collection không tồn tại
                pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _fingerprint(path: str = TICKETS_PATH) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def build_index(db_path: str = DEFAULT_DB_PATH, tickets_path: str = TICKETS_PATH,
                corrections_path: str = CORRECTIONS_PATH) -> chromadb.Collection:
    """Xây lại index: mọi vé trong tickets.csv (kèm phản hồi mẫu nếu có) + các chỉnh sửa của nhân viên."""
    from modules.feedback_manager import FeedbackManager

    collection = get_ticket_collection(get_chroma_client(db_path), recreate=True)
    df = pd.read_csv(tickets_path).dropna(subset=["email_content", "category"])
    responses = df["response"].fillna("") if "response" in df.columns else pd.Series("", index=df.index)

    print(f"📦 Indexing {len(df)} tickets into ChromaDB...")
    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start:start + BATCH_SIZE]
        collection.add(
            ids=chunk["id"].astype(str).tolist(),
            documents=chunk["email_content"].astype(str).tolist(),
            metadatas=[
                {"category": str(cat), "source": "dataset", "response": str(resp)}
                for cat, resp in zip(chunk["category"], responses.loc[chunk.index])
            ],
        )
        print(f"   • {min(start + BATCH_SIZE, len(df))}/{len(df)}")

    for ticket_id, c in FeedbackManager(corrections_path).latest_by_ticket().items():
        if c.get("email_content") and c.get("corrected_department"):
            add_correction(ticket_id, c["email_content"], c["corrected_department"], collection)

    (Path(db_path) / FINGERPRINT_FILE).write_text(_fingerprint(tickets_path))
    print("✅ ChromaDB index ready")
    return collection


def get_index(db_path: str = DEFAULT_DB_PATH) -> chromadb.Collection:
    """Index dùng chung; tự xây lại khi chưa có hoặc tickets.csv đã thay đổi."""
    global _index
    with _index_lock:
        if _index is None:
            fp_file = Path(db_path) / FINGERPRINT_FILE
            if fp_file.exists() and fp_file.read_text() == _fingerprint():
                _index = get_ticket_collection(get_chroma_client(db_path))
            else:
                _index = build_index(db_path)
        return _index


def add_correction(ticket_id: str, text: str, department: str, collection=None) -> None:
    (collection or get_index()).upsert(
        ids=[f"correction_{ticket_id}"],
        documents=[text],
        metadatas=[{"category": department, "source": "human_correction", "response": ""}],
    )


def find_similar(texts: list[str], n: int = 3, where: dict | None = None,
                 exclude_ids: list[str] | None = None) -> list[list[dict]]:
    """Top-n vé gần nhất cho mỗi câu; bỏ chính vé đó (và chỉnh sửa của nó) khỏi kết quả."""
    collection = get_index()
    # Lấy dư để còn đủ n sau khi bỏ chính vé đó và các vé trùng nội dung (dữ liệu cũ có nhiều vé lặp)
    result = collection.query(query_texts=texts, n_results=n * 3 + 2, where=where)
    out = []
    for i in range(len(texts)):
        skip = set()
        if exclude_ids and exclude_ids[i]:
            skip = {str(exclude_ids[i]), f"correction_{exclude_ids[i]}"}
        hits, seen_texts = [], set()
        for doc_id, doc, dist, meta in zip(
            result["ids"][i], result["documents"][i], result["distances"][i], result["metadatas"][i]
        ):
            if doc_id in skip or doc in seen_texts:
                continue
            seen_texts.add(doc)
            hits.append({"id": doc_id, "text": doc, "similarity": 1 - dist, **meta})
        out.append(hits[:n])
    return out
