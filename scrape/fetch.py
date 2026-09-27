import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "flow3d-rag-study/0.1 (personal learning project)"}
URLS_PATH = Path("data/urls.json")
HTML_DIR = Path("data/raw/html")
PDF_DIR = Path("data/raw/pdf")


def slug(url: str) -> str:
    """Turn a URL path into a safe filename. The URL itself stays in urls.json."""
    path = urlparse(url).path.strip("/")
    path = path.removesuffix(".pdf")
    return path.replace("/", "_") or "index"


def download(url: str, dest: Path) -> str:
    """Save one URL. Return 'saved', 'skipped', or 'failed'."""
    if dest.exists():
        return "skipped"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)  # raw bytes, so encoding is unchanged
        time.sleep(1.5)  # only after a real request
        return "saved"
    except requests.RequestException as exc:
        print(f"fail {url}: {exc}")
        return "failed"

def pdf_links(html: bytes, page_url: str) -> list[str]:
    """Return on-site PDF hrefs from one saved page."""
    soup = BeautifulSoup(html, "lxml")
    found = []
    seen = set()
    for tag in soup.find_all("a", href=True):
        link = urljoin(page_url, tag["href"]).split("#")[0]
        host = urlparse(link).netloc
        if host not in ("www.flow3d.com", "flow3d.com"):
            continue
        if "?" in link or not link.lower().endswith(".pdf"):
            continue
        if link not in seen:
            seen.add(link)
            found.append(link)
    return found

def fetch_all(urls_path: Path) -> None:
    """Download missing HTML, then PDFs linked from those pages."""
    records = json.loads(urls_path.read_text(encoding="utf-8"))
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"html_saved": 0, "html_skipped": 0, "pdf_saved": 0, "pdf_skipped": 0, "failed": 0}
    for record in records:
        url = record["url"]
        html_path = HTML_DIR / f"{slug(url)}.html"
        status = download(url, html_path)
        counts[f"html_{status}" if status != "failed" else "failed"] += 1
        if status == "failed":
            continue
        for pdf_url in pdf_links(html_path.read_bytes(), url):
            pdf_status = download(pdf_url, PDF_DIR / f"{slug(pdf_url)}.pdf")
            if pdf_status == "failed":
                counts["failed"] += 1
            else:
                counts[f"pdf_{pdf_status}"] += 1
    print(counts)

if __name__ == "__main__":
    fetch_all(URLS_PATH)