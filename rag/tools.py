import json
from pathlib import Path
from typing import Protocol

from retrieve import SearchIndex

DOCS_PATH = Path("data/processed/docs.jsonl")
MAX_HITS = 5
WINDOW = 400  # characters around the match, not the whole page


class Tool(Protocol):
    """Every tool is only a name and search(). Grep does not need the embedding index."""

    name: str

    def search(self, query: str) -> list[dict]:
        """Return hits in the same shape retrieve() uses."""


class GrepTool:
    """Literal search of docs.jsonl. One hit per document, highest match count first.

    Pass a short token like "DNS" or "VOF", not the full question. docs.jsonl has
    no chunk overlap, and the snippet stays small so the context window does not
    fill with the rest of the page.
    """

    name = "grep"

    def search(self, query: str) -> list[dict]:
        needle = query.lower()
        found = []
        for line in DOCS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            text = doc["text"]
            at = text.lower().find(needle)
            if at < 0:
                continue
            start = max(0, at - WINDOW // 2)
            end = min(len(text), start + WINDOW)
            # Finish the sentence so the snippet is not cut off mid-thought.
            period = text.find(". ", end, end + WINDOW)
            if period != -1:
                end = period + 1
            found.append({
                "chunk_id": doc["id"],
                "text": text[start:end].strip(),
                "heading": doc["title"],
                "source_type": doc["source_type"],
                "url": doc["url"],
                "score": float(text.lower().count(needle)),
            })

        found.sort(key=lambda hit: hit["score"], reverse=True)
        return found[:MAX_HITS]


class SemanticTool:
    """Hybrid search. Use this when the wording is not an exact token."""

    name = "semantic"

    def __init__(self, index: SearchIndex) -> None:
        self.index = index

    def search(self, query: str) -> list[dict]:
        return self.index.retrieve(query)
