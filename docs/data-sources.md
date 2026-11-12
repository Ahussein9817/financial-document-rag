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
