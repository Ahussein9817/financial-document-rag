"""Run the ingestion pipeline: PDFs -> chunks -> OpenAI embeddings -> Chroma.

Requires OPENAI_API_KEY (paid embedding calls) and either data/raw_filings/manifest.json
(preferred; fiscal year read from each filing) or legacy PDFs in data/raw_pdfs/ named
TICKER_10K_YEAR.pdf, whose year then comes from the filename (see docs/data-sources.md).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (copy .env.example to .env).")
    have_manifest = (Path("data/raw_filings") / "manifest.json").exists()
    if not have_manifest and not list(Path("data/raw_pdfs").glob("*.pdf")):
        sys.exit("No data/raw_filings/manifest.json and no PDFs in data/raw_pdfs/. "
                 "Run scripts/fetch_filings.py (see docs/data-sources.md).")
    import ingestion

    ingestion.main()


if __name__ == "__main__":
    main()
