import hashlib
import json

import machine_check_candidates as mcc


def _corpus(tmp_path):
    html = ('<p>Net income $ 58,471 $ 49,552</p><hr style="page-break-after:always">'
            "<p>Other page</p>")
    (tmp_path / "JPM_10K_2024.htm").write_text(html)
    manifest = {"filings": [{"file": "JPM_10K_2024.htm"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_corpus_pages_match_ingestion_locators_and_hash_manifest(tmp_path):
    d = _corpus(tmp_path)
    pages, sha = mcc.corpus_pages(d)
    assert set(pages) == {"JPM_10K_2024.htm#p1"} and "58,471" in pages["JPM_10K_2024.htm#p1"]
    assert sha == hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()


def test_run_promotes_passing_items_leaves_others_and_never_validates(tmp_path):
    pages, sha = mcc.corpus_pages(_corpus(tmp_path))
    good = {"id": "g", "stratum": "single_text", "question": "q", "answerable": True,
            "status": "candidate", "origin": "claude_drafted",
            "gold": {"answer": "x", "evidence": [{"passage_id": "JPM_10K_2024.htm#p1", "snippet": "Net income $ 58,471"}]}}
    bad = {**good, "id": "b", "gold": {"answer": "x", "evidence": [{"passage_id": "JPM_10K_2024.htm#p1", "snippet": "not there"}]}}
    nogold = {"id": "n", "stratum": "single_text", "question": "q", "answerable": True,
              "status": "candidate", "origin": "claude_drafted"}
    out = mcc.run({"questions": [good, bad, nogold]}, pages, sha, "2026-11-12")
    st = {q["id"]: q["status"] for q in out["questions"]}
    assert st == {"g": "machine_checked", "b": "candidate", "n": "candidate"}
    assert "machine_check" not in out["questions"][2]
    assert out["questions"][1]["machine_check"]["passed"] is False
    assert out["corpus_manifest_sha256"] == sha and "human_validated" not in st.values()


def test_shipped_manifest_is_complete_and_resolved():
    from pathlib import Path
    m = json.loads((Path(__file__).resolve().parent.parent / "data/raw_filings/manifest.json").read_text())["filings"]
    assert len(m) == 9
    assert all(e["fiscal_year"] and e["period_of_report"] and e["accession"] and len(e["sha256"]) == 64 for e in m)
    assert all(e["fiscal_year_source"] == "ixbrl_dei" and e["form_type"] == "10-K" for e in m)
    assert {e["file"] for e in m} == {f"{t}_10K_{y}.htm" for t in ("JPM", "BAC", "C") for y in (2022, 2023, 2024)}
