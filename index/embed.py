import json
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DB_PATH = Path("db")
BM25_PATH = Path("db/bm25.pkl")
COLLECTION = "flow3d"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks() -> list[dict]:
    lines = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_chroma(chunks: list[dict], model: SentenceTransformer) -> None:
    """Embed chunk text and store vectors. Skip the work if this index is already complete."""
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == len(chunks):
        print(f"chroma already has {collection.count()} chunks")
        return
    if collection.count() not in (0, len(chunks)):
        # A crashed run leaves a partial index. Delete it and start over.
        client.delete_collection(COLLECTION)
        collection = client.get_or_create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metas = [
        {
            "doc_id": chunk["doc_id"],
            "heading": chunk["heading"],
            "source_type": chunk["source_type"],
            "url": chunk["url"],
        }
        for chunk in chunks
    ]

    # bge-small only reads the first 512 tokens. Longer chunks are cut off here.
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=vectors.tolist())
    print(f"chroma count: {collection.count()}")


def build_bm25(chunks: list[dict]) -> None:
    """Keyword index for exact terms like VOF and FAVOR. BM25 returns positions, so ids stay in the same order."""
    ids = [chunk["chunk_id"] for chunk in chunks]
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    print(f"bm25 chunks: {len(ids)}")


def main() -> None:
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_NAME)
    build_chroma(chunks, model)
    build_bm25(chunks)


if __name__ == "__main__":
    main()