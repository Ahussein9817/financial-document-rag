"""Recompute the README's evaluation numbers from committed files. Offline: no
API key, PDFs or vector store needed.

    python scripts/reproduce_reported_metrics.py

This reproduces the AGGREGATION of the author's manual scores (53.3% overall,
86.7% document-level recall@4). It does not re-run retrieval or generation and
cannot re-validate the manual scoring itself.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval_metrics import document_recall, summarize_scores  # noqa: E402


def main() -> dict:
    results = pd.read_csv(ROOT / "data/processed/evaluation_results_complete.csv")
    recall = pd.read_csv(ROOT / "data/processed/retrieval_recall_analysis.csv")
    out = {"answer_scores": summarize_scores(results), "retrieval": document_recall(recall)}
    out["note"] = (
        "README text states 60% factual accuracy; the committed scores give "
        f"{out['answer_scores']['category_accuracy_pct'].get('factual'):.1f}%."
    )
    dest = ROOT / "outputs" / "reproduced_metrics.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
