"""Run the ingestion pipeline: PDFs -> chunks -> OpenAI embeddings -> Chroma.

Requires OPENAI_API_KEY (paid embedding calls) and the 10-K PDFs in
data/raw_pdfs/ named TICKER_10K_YEAR.pdf (see docs/data-sources.md).
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
    pdfs = list(Path("data/raw_pdfs").glob("*.pdf"))
    if not pdfs:
        sys.exit("No PDFs in data/raw_pdfs/. See docs/data-sources.md for how to obtain them.")
    import ingestion

    ingestion.main()


if __name__ == "__main__":
    main()
