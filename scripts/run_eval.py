"""Run the question set through the live pipeline and save answers, sources and latency.

Requires OPENAI_API_KEY and a populated data/vectorstore (paid API calls).
Writes outputs/run_<timestamp>.json. Scoring remains manual; this script does
not grade answers.

    python scripts/run_eval.py [--questions data/eval/eval_questions_v1.json] [--k 4]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def run_questions(questions, ask, k):
    """Call `ask(question, k)` for each item; record answer, sources, wall time, errors."""
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            res = ask(q["question"], k)
            err = None
        except Exception as e:  # keep going; record the failure
            res, err = {"answer": None, "sources": []}, f"{type(e).__name__}: {e}"
        rows.append({
            "id": q.get("id"), "category": q.get("category"), "question": q["question"],
            "expected_source": q.get("source"), "answer": res["answer"],
            "retrieved_files": [s["file"] for s in res["sources"]],
            "retrieved_passages": [s.get("passage_id") for s in res["sources"]],
            "cited_passages": res.get("cited_passages"),
            "invalid_citations": res.get("invalid_citations"),
            "prompt_version": res.get("prompt_version"),
            "retrieved": res["sources"],
            "filter": res.get("filter"),
            "latency_s": round(time.perf_counter() - t0, 3), "error": err,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "data/eval/eval_questions_v1.json"))
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    import rag_pipeline

    vs = rag_pipeline.load_vectorstore(str(ROOT / "data/vectorstore"))
    questions = json.load(open(args.questions))["questions"]
    rows = run_questions(questions, lambda q, k: rag_pipeline.smart_ask(q, vs, k=k, verbose=False), args.k)
    out = ROOT / "outputs" / f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps({"k": args.k, "questions_file": args.questions, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
