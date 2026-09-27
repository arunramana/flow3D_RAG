import json
import re
from pathlib import Path

import tiktoken

DOCS_PATH = Path("data/processed/docs.jsonl")
OUT_PATH = Path("data/processed/chunks.jsonl")
MAX_TOKENS = 700
OVERLAP_TOKENS = 100
ENCODING = tiktoken.get_encoding("cl100k_base")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def token_len(text: str) -> int:
    return len(ENCODING.encode(text))


def heading_sections(text: str, title: str, glossary: bool) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, body) sections."""
    stack = [(0, title)]  # level 0 stays under every real heading
    buf: list[str] = []
    sections: list[tuple[str, str]] = []

    def path() -> str:
        return " > ".join(name for _, name in stack)

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if body:
            sections.append((path(), body))

    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if not match:
            buf.append(line)
            continue

        flush()
        level = len(match.group(1))
        name = match.group(2).strip()

        # The article body sits under this heading, so drop the line, not the text after it.
        if name.lower() == "table of contents":
            continue
        if name == title:
            continue
        # Glossary "A", "B", ... are alphabet buckets, not definitions.
        if glossary and len(name) == 1:
            continue

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))

    flush()
    return sections


def split_section(heading: str, text: str) -> list[str]:
    """Keep a short section whole. Window a long one so a cut sentence is in both pieces."""
    prefix = f"{heading}\n\n"
    budget = MAX_TOKENS - token_len(prefix)
    if token_len(text) <= budget:
        return [prefix + text]

    tokens = ENCODING.encode(text)
    step = max(budget - OVERLAP_TOKENS, 1)
    pieces = []
    start = 0
    while start < len(tokens):
        pieces.append(prefix + ENCODING.decode(tokens[start:start + budget]))
        if start + budget >= len(tokens):
            break
        start += step
    return pieces


def chunk_doc(doc: dict) -> list[dict]:
    glossary = doc["source_type"] == "glossary"
    chunks = []
    i = 0
    for heading, body in heading_sections(doc["text"], doc["title"], glossary):
        parts = [f"{heading}\n\n{body}"] if glossary else split_section(heading, body)
        for text in parts:
            chunks.append({
                "chunk_id": f"{doc['id']}:{i}",
                "doc_id": doc["id"],
                "text": text,
                "heading": heading,
                "source_type": doc["source_type"],
                "url": doc["url"],
            })
            i += 1
    return chunks


def main() -> None:
    docs = [json.loads(line) for line in DOCS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = [chunk for doc in docs for chunk in chunk_doc(doc)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    glossary_n = sum(1 for chunk in chunks if chunk["source_type"] == "glossary")
    print(f"wrote {len(chunks)} chunks from {len(docs)} docs ({glossary_n} glossary) -> {OUT_PATH}")


if __name__ == "__main__":
    main()