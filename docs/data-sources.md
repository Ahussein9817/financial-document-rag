# Data sources

**Documents:** SEC Form 10-K annual reports, public domain, from SEC EDGAR (https://www.sec.gov/edgar/search/).

| Ticker | Company | Fiscal years in the evaluated vector store |
|---|---|---|
| JPM | JPMorgan Chase & Co. | 2023, 2024, 2025 (by filename) |
| BAC | Bank of America Corp. | 2022, 2023, 2024 |
| C | Citigroup Inc. | 2022, 2023, 2024 |

These nine PDFs are **not committed** (`.gitignore`) and the exact files used for the historical evaluation are not recorded in this repo (no checksums, no EDGAR accession numbers). Anyone re-creating the corpus gets filings of the same companies and years, but not guaranteed byte-identical PDFs; expect different chunk counts than the historical 17,306.

**Naming convention (required):** `data/raw_pdfs/TICKER_10K_YEAR.pdf`, e.g. `JPM_10K_2024.pdf`. `src/ingestion.py` takes ticker and year from the filename; a file with fewer than three `_`-separated parts is loaded **without** ticker/year metadata and will silently never match a metadata filter.

**Credentials:** `OPENAI_API_KEY` in `.env` (see `.env.example`). Embeddings and generation are paid OpenAI calls.

**Pipeline:** `python scripts/ingest.py` -> `python scripts/run_eval.py` -> manual scoring. `python scripts/reproduce_reported_metrics.py` recomputes the README aggregates offline from committed files.


## Recovering the filings (not yet done)

`scripts/fetch_filings.py` downloads named filings from EDGAR (free; declare a User-Agent with a contact; limit 10 requests/second, the script stays under 5) and writes `manifest.json` (accession, URL, SHA-256, bytes). CIKs (JPM 0000019617, BAC 0000070858, C 0000831001) were checked against EDGAR's company pages. EDGAR serves the primary document usually as HTML, so chunk counts will differ from the historical 17,306. The filename year is taken from the filing's own "fiscal year ended" statement, not from the filing date: for example the 10-K filed in February 2025 covers fiscal 2024. Which fiscal year the historical `*_2025` files represent is unrecorded and must be settled before any v1 vs v2 comparison.
