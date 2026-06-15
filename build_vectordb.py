import pandas as pd

from modules.embeddings import ensure_embedding_model
from modules.vector_store import get_chroma_client, get_ticket_collection

BATCH_SIZE = 25


def init_vector_db() -> None:
    print("🚀 Khởi tạo Vector Database (ChromaDB)...")

    ensure_embedding_model()

    client = get_chroma_client()
    collection = get_ticket_collection(client, recreate=True)

    df = pd.read_csv("data/tickets.csv")

    documents = df["email_content"].astype(str).tolist()
    metadatas = [
        {"category": str(row["category"]), "status": str(row["status"])}
        for _, row in df.iterrows()
    ]
    ids = df["id"].astype(str).tolist()

    print(f"📦 Đang đẩy {len(documents)} tickets vào Vector DB...")
    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"   • Đã xử lý {min(end, len(documents))}/{len(documents)}")

    print("✅ Đã đẩy xong! Vector DB đã sẵn sàng để AI 'học hỏi'.")


if __name__ == "__main__":
    init_vector_db()
