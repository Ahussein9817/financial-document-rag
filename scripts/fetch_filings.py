"""Fetch 10-K filings from SEC EDGAR and write a provenance manifest.

NOT run by CI and not run during development: it downloads files. Run it yourself:

    python scripts/fetch_filings.py --user-agent "Your Name your@email" \
        --filings JPM:0000019617-25-000270 --out data/raw_filings

SEC fair-access rules: declare a User-Agent with a contact, stay <= 10 requests/second
(this script uses <= 5), download only what you need. The accession numbers are given
explicitly (no guessing which filing is which fiscal year); `fiscal_year_from_text`
reads the fiscal year and period of report from the filing itself (iXBRL cover tags, then
the cover sentence) via `html_filing.read_filing_metadata`; it never takes the year from the
filing date or a filename. If the filing's own metadata cannot be read the entry is stored as
`<ticker>_10K_UNRESOLVED_<accession>` with fiscal_year null and ingestion refuses it.

Each filing is stored as <ticker>_10K_<fiscal year>.<ext> and recorded in manifest.json with
accession number, source URL, form type, period of report, fiscal year and how it was
resolved, SHA-256 and size.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from html_filing import locate, read_filing_metadata  # noqa: E402

CIKS = {"JPM": "0000019617", "BAC": "0000070858", "C": "0000831001"}
MIN_INTERVAL_S = 0.2  # <= 5 requests/second


def http_get(url: str, user_agent: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


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
    text = body.decode("utf-8", "ignore")
    meta = read_filing_metadata(text)
    units, locator_kind = locate(text) if name.lower().endswith((".htm", ".html")) else ([], "pdf_page")
    fy = meta["fiscal_year"]
    ext = Path(name).suffix.lower()
    label = str(fy) if fy else f"UNRESOLVED_{accession}"
    dest = out / f"{ticker}_10K_{label}{ext}"
    out.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return {"ticker": ticker, "accession": accession, "source_url": url,
            "form_type": meta["form_type"], "period_of_report": meta["period_of_report"],
            "fiscal_year": fy, "fiscal_year_source": meta["fiscal_year_source"],
            "locator_kind": locator_kind, "n_locators": len(units),
            "file": dest.name, "sha256": sha256_bytes(body), "bytes": len(body),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


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
