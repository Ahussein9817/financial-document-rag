"""Fetch 10-K filings from SEC EDGAR and write a provenance manifest.

NOT run by CI and not run during development: it downloads files. Run it yourself:

    python scripts/fetch_filings.py --user-agent "Your Name your@email" \
        --filings JPM:0000019617-25-000270 --out data/raw_filings

SEC fair-access rules: declare a User-Agent with a contact, stay <= 10 requests/second
(this script uses <= 5), download only what you need. The accession numbers are given
explicitly (no guessing which filing is which fiscal year); `fiscal_year_from_text`
reads the cover page ("for the fiscal year ended ...") so the filename year can be
checked against the filing's own statement.

Each filing is stored as ticker_10K_<fiscal year>.<ext> and recorded in manifest.json
with accession number, source URL, SHA-256 and size.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

CIKS = {"JPM": "0000019617", "BAC": "0000070858", "C": "0000831001"}
MIN_INTERVAL_S = 0.2  # <= 5 requests/second


def http_get(url: str, user_agent: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fiscal_year_from_text(text: str) -> int | None:
    """Parse 'for the fiscal year ended December 31, 2024' from a cover page."""
    m = re.search(r"fiscal\s+year\s+ended\s+[A-Za-z]+\s+\d{1,2},\s*(\d{4})", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def index_url(cik: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{accession}-index.json"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pick_primary_document(index: dict) -> str:
    """Primary 10-K document from an EDGAR filing index: prefer the largest .htm document
    whose name does not look like an exhibit."""
    items = [i for i in index.get("directory", {}).get("item", [])
             if i["name"].lower().endswith((".htm", ".html", ".pdf"))
             and not re.search(r"ex[-_]?\d|exhibit|index", i["name"], re.IGNORECASE)]
    if not items:
        raise ValueError("no candidate primary document in filing index")
    return max(items, key=lambda i: int(i.get("size") or 0))["name"]


def fetch_filing(ticker: str, accession: str, out: Path, user_agent: str,
                 get=http_get, sleep=time.sleep) -> dict:
    cik = CIKS[ticker]
    idx = json.loads(get(index_url(cik, accession), user_agent))
    sleep(MIN_INTERVAL_S)
    name = pick_primary_document(idx)
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{name}"
    body = get(url, user_agent)
    sleep(MIN_INTERVAL_S)
    text = re.sub(r"<[^>]+>", " ", body[:400_000].decode("utf-8", "ignore"))
    fy = fiscal_year_from_text(text)
    ext = Path(name).suffix.lower()
    dest = out / f"{ticker}_10K_{fy if fy else 'UNKNOWN'}{ext}"
    out.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return {"ticker": ticker, "accession": accession, "source_url": url,
            "fiscal_year_from_text": fy, "file": dest.name,
            "sha256": sha256_bytes(body), "bytes": len(body)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-agent", required=True, help='e.g. "Name email@example.com"')
    ap.add_argument("--filings", nargs="+", required=True, help="TICKER:ACCESSION ...")
    ap.add_argument("--out", default="data/raw_filings")
    args = ap.parse_args()
    out = Path(args.out)
    entries = []
    for spec in args.filings:
        ticker, accession = spec.split(":")
        entries.append(fetch_filing(ticker, accession, out, args.user_agent))
        print("fetched", entries[-1]["file"])
    (out / "manifest.json").write_text(json.dumps({"filings": entries}, indent=2))


if __name__ == "__main__":
    main()
