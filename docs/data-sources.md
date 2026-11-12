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

`scripts/fetch_filings.py` downloads named filings from EDGAR (free; declare a User-Agent with a contact; EDGAR allows 10 requests/second, the script stays at or below 5) and writes `manifest.json`. CIKs (JPM 0000019617, BAC 0000070858, C 0000831001) were checked against EDGAR's company pages; the full corpus has not been downloaded.

**Fiscal year is read from the filing, never from a filename or filing date.** Order of precedence: iXBRL cover tags (`dei:DocumentFiscalYearFocus`, then `dei:DocumentPeriodEndDate`), then the cover sentence "for the fiscal year ended <date>". The manifest records, per filing: accession number, source URL, form type, period of report, fiscal year, **how the year was resolved** (`fiscal_year_source`), SHA-256, size and retrieval time. If the filing's own metadata cannot be read, the entry is stored as `<ticker>_10K_UNRESOLVED_<accession>` with a null fiscal year, and `ingestion.load_filings` refuses it (as it does any file whose SHA-256 differs from the manifest). Example of why this matters: a 10-K filed in February 2025 reports fiscal 2024.

Chunk metadata then carries `ticker`, `year` (= the filing's fiscal year), `accession`, `period_of_report`, `fiscal_year_source`, `source_file`, `page`, `passage_id` (`<file>#p<page>`), `chunk_id` (`<passage_id>#c<n>`), `start_index` and `content_sha1`. Retrieval results return these same fields for the chunks the answer was generated from.

**Pages in HTML filings are an approximation.** EDGAR serves the primary 10-K document as HTML, which has no fixed pagination. `src/html_filing.py` splits on the filing's own print page-break markers; these "pages" are not any PDF's page numbers, so passage ids from the new corpus are not comparable to the historical PDF corpus. A filing with no page-break markers becomes a single page (page-level evidence would then be too coarse; check `n_pages` per file after the first fetch).

**Historical corpus:** the original `*_2025`-style PDFs are not recorded anywhere with accession numbers, so which fiscal year they hold is unknown. The legacy PDF loader tags those chunks `fiscal_year_source='filename'`. Historical (v1) results stay README-reported and are not compared against v2 results.

**Filter limitation:** `detect_filters` only recognises the years 2022-2025 in a question and treats the year as a fiscal year; a question about "2024" is filtered to fiscal 2024.
