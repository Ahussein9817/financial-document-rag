"""Machine-check eval items that have drafted gold evidence against the local filing corpus.

    python scripts/machine_check_candidates.py [--questions data/eval/eval_questions_v2.json]
                                               [--filings data/raw_filings]

Offline (reads data/raw_filings/manifest.json and the downloaded filings). For each item with
gold evidence (or absence_terms), checks that snippets occur on the cited passages, numeric
values appear on a cited passage, and absence terms occur nowhere in the corpus. Passing items
become `machine_checked`; failing ones stay `candidate` with the failure recorded. Never
writes `human_validated`: a person does that after reading the filing.
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from eval_schema import apply_machine_check  # noqa: E402
from html_filing import locate  # noqa: E402


def corpus_pages(directory: Path) -> tuple[dict[str, str], str]:
    """{passage_id: text} for every filing in the manifest, plus the manifest's SHA-256.
    Uses the same locate() as ingestion, so passage ids match what retrieval returns."""
    manifest_bytes = (directory / "manifest.json").read_bytes()
    pages: dict[str, str] = {}
    for e in json.loads(manifest_bytes)["filings"]:
        html = (directory / e["file"]).read_text(encoding="utf-8", errors="ignore")
        units, _ = locate(html)
        for i, text in enumerate(units):
            pages[f"{e['file']}#p{i + 1}"] = text
    return pages, hashlib.sha256(manifest_bytes).hexdigest()


def run(data: dict, pages: dict[str, str], corpus_sha: str, today: str) -> dict:
    out = []
    for q in data["questions"]:
        has_gold = q.get("gold", {}).get("evidence") or q.get("absence_terms")
        out.append(apply_machine_check(q, pages, today, corpus_sha) if has_gold else q)
    return {**data, "corpus_manifest_sha256": corpus_sha, "questions": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "data/eval/eval_questions_v2.json"))
    ap.add_argument("--filings", default=str(ROOT / "data/raw_filings"))
    args = ap.parse_args()
    pages, sha = corpus_pages(Path(args.filings))
    data = json.loads(Path(args.questions).read_text())
    result = run(data, pages, sha, date.today().isoformat())
    Path(args.questions).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for q in result["questions"]:
        mc = q.get("machine_check")
        print(q["id"], q["status"], "" if not mc else ("ok" if mc["passed"] else mc["problems"]))


if __name__ == "__main__":
    main()
