import pickle
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer


DB_PATH = Path("db")
BM25_PATH = Path("db/bm25.pkl")
COLLECTION = "flow3d"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K = 20
RRF_K = 60

RERANKER_NAME = "BAAI/bge-reranker-base"
TOP_N = 5


class SearchIndex:
    """Loads Chroma, BM25, and both models once.

    Callers depend on this object. They do not each open the database themselves.
    """

    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        self.collection = client.get_collection(COLLECTION)
        self.model = SentenceTransformer(MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_NAME)
        with BM25_PATH.open("rb") as f:
            saved = pickle.load(f)
        self.bm25 = saved["bm25"]
        self.ids: list[str] = saved["ids"]

    def vector_search(self, query: str, source_type: str | None) -> list[str]:
        """Top chunk ids by cosine similarity. The prefix is query-only."""
        vector = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        kwargs = {"query_embeddings": [vector.tolist()], "n_results": K}
        if source_type:
            kwargs["where"] = {"source_type": source_type}
        found = self.collection.query(**kwargs)
        return found["ids"][0]

    def bm25_search(self, query: str, allowed: set[str] | None) -> list[str]:
        """Top chunk ids by keyword score. Zero scores are not matches."""
        scores = self.bm25.get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0 or (allowed is not None and self.ids[i] not in allowed):
                continue
            hits.append(self.ids[i])
            if len(hits) == K:
                break
        return hits

    def rrf(self, rankings: list[list[str]]) -> list[tuple[str, float]]:
        """Add 1/(60 + rank) across lists. Rank 1 is the best hit in that list."""
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Re-score the fused hits by reading the question and the chunk together.

        Too slow for all 1783 chunks, so it only runs on the short hybrid list.
        """
        scores = self.reranker.predict([(query, hit["text"]) for hit in results])
        for hit, score in zip(results, scores):
            hit["score"] = float(score)
        return sorted(results, key=lambda hit: hit["score"], reverse=True)[:TOP_N]

    def retrieve(self, query: str, source_type: str | None = None, use_reranker: bool = True) -> list[dict]:
        """Hybrid search. source_type limits both lists, not just the vector one.

        Set use_reranker False to compare hybrid search alone.
        """
        allowed = None
        if source_type:
            allowed = set(self.collection.get(where={"source_type": source_type})["ids"])

        fused = self.rrf([
            self.vector_search(query, source_type),
            self.bm25_search(query, allowed),
        ])[:K]

        top_ids = [chunk_id for chunk_id, _ in fused]
        score_by_id = dict(fused)
        got = self.collection.get(ids=top_ids, include=["documents", "metadatas"])
        # Chroma does not return get() results in the order you requested.
        by_id = {
            chunk_id: (doc, meta)
            for chunk_id, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
        }

        results = []
        for chunk_id in top_ids:
            doc, meta = by_id[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "text": doc,
                "heading": meta["heading"],
                "source_type": meta["source_type"],
                "url": meta["url"],
                "score": score_by_id[chunk_id],
            })
        if use_reranker:
            return self.rerank(query, results)
        return results


def main() -> None:
    query = "What is DNS?"
    hits = SearchIndex().retrieve(query)
    for hit in hits:
        print(f"{hit['score']:.4f}  {hit['heading']}")


if __name__ == "__main__":
    main()
