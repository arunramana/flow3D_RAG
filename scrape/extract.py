import json
from pathlib import Path
from urllib.parse import urlparse

import fitz  # pymupdf
import trafilatura

URLS_PATH = Path("data/urls.json")
HTML_DIR = Path("data/raw/html")
PDF_DIR = Path("data/raw/pdf")
OUT_PATH = Path("data/processed/docs.jsonl")


def slug(url: str) -> str:
    """Same rule as fetch.py, so the filename matches the URL."""
    path = urlparse(url).path.strip("/").removesuffix(".pdf")
    return path.replace("/", "_") or "index"


def html_doc(record: dict) -> dict | None:
    """Turn one saved HTML page into a JSON record, or None if there is no article text."""
    url = record["url"]
    path = HTML_DIR / f"{slug(url)}.html"
    html = path.read_text(encoding="utf-8", errors="ignore")

    # Markdown keeps headings, which chunk.py will split on.
    text = trafilatura.extract(html, include_tables=True, output_format="markdown")
    if not text or len(text.strip()) < 200:
        return None

    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else slug(url))
    date = meta.date if meta else None

    return {
        "id": slug(url),
        "url": url,
        "title": title,
        "source_type": record["source_type"],
        "date": date,
        "text": text,
    }


def pdf_doc(path: Path) -> dict | None:
    """Turn one saved PDF into a JSON record. These files are not listed in urls.json."""
    with fitz.open(path) as doc:
        text = "\n\n".join(page.get_text() for page in doc)
        title = (doc.metadata or {}).get("title") or path.stem
    if not text.strip():
        return None

    # slug() replaced "/" with "_", so undo that for the one URL we did not store.
    url = "https://www.flow3d.com/" + path.stem.replace("_", "/") + ".pdf"
    return {
        "id": path.stem,
        "url": url,
        "title": title,
        "source_type": "pdf",
        "date": None,
        "text": text,
    }


def extract_all() -> None:
    """Write one JSON object per line to docs.jsonl."""
    records = json.loads(URLS_PATH.read_text(encoding="utf-8"))
    docs = []
    skipped = 0

    for record in records:
        doc = html_doc(record)
        if doc is None:
            skipped += 1
            continue
        docs.append(doc)

    for path in sorted(PDF_DIR.glob("*.pdf")):
        doc = pdf_doc(path)
        if doc is not None:
            docs.append(doc)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"wrote {len(docs)} docs, skipped {skipped} thin HTML pages -> {OUT_PATH}")


if __name__ == "__main__":
    extract_all()