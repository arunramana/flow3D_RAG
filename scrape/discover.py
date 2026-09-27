import json
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup



RULES = [
    ("/resources/cfd-101/", "cfd-101"),
    ("/modeling-capabilities/", "modeling-capabilities"),
    ("/whats-new-in-flow-3d/", "whats-new"),
    ("/resources/case-studies/", "case-studies"),
    ("/cfd-glossary/", "glossary"),
    ("/conference-proceedings/", "proceedings"),
    ("/resources/bibliography/", "proceedings"),
]

HEADERS = {"User-Agent": "flow3d-rag-study/0.1 (personal learning project)"}
SITEMAP_INDEX = "https://www.flow3d.com/sitemap_index.xml"


def fetch_locs(url: str) -> list[str]:
    
    """Download one sitemap and return every <loc> URL in it."""

    response = requests.get(url, headers=HEADERS, timeout=30)

    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'xml')
    return [loc.text.strip() for loc in soup.find_all('loc')]

def source_type(url: str, from_post_sitemap: bool) -> str:

    """tag a URL or return None when it should be dropped"""

    if "?" in url or "/page/" in url:
        return None

    if "users.flow3d.com" in url or "/webinar" in url:
        return None

    if "/wp-content/uploads/" in url:
        return None

    for path, kind in RULES:
        if path in url:
            return kind
    
    return "blog" if from_post_sitemap else None


def discover(sitemap_index_url: str) -> list[dict]:

    """Return [{url, source_type}, ...] from the sitemap index."""

    records = []
    seen = set()

    for sitemap_url in fetch_locs(sitemap_index_url):

        from_post_sitemap = sitemap_url.endswith("/post-sitemap.xml")
        time.sleep(1.5)

        for url in fetch_locs(sitemap_url):

            kind = source_type(url, from_post_sitemap)

            if kind is None or url in seen:

                continue

            seen.add(url)
            records.append({"url": url, "source_type": kind})
    
    return records


if __name__ == "__main__":
    found = discover(SITEMAP_INDEX)
    out = Path("data/urls.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(found, indent=2), encoding="utf-8")

    counts ={}

    for record in found:
        counts[record["source_type"]] = counts.get(record["source_type"], 0) + 1

    print(counts)
    print(f"Wrote {len(found)} URLs to {out}")