import copy
import json
from pathlib import Path

from eval_schema import (apply_machine_check, machine_check, numeric_forms, scorable,
                         validate_dataset, validate_question)

ROOT = Path(__file__).resolve().parent.parent
PAGES = {
    "JPM_10K_2024.htm#p12": "Net income was $ 58,471 million for the year. Diluted EPS 19.75.",
    "JPM_10K_2024.htm#p13": "Other text",
}


def _checked_numeric():
    return {
        "id": "n1", "stratum": "single_numeric", "question": "JPM net income FY2024?",
        "answerable": True, "status": "machine_checked", "origin": "human_drafted",
        "gold": {"answer": "$58,471 million",
                 "numeric": {"value": 58471e6, "unit": "USD", "scale": "million", "period": "FY2024"},
                 "evidence": [{"passage_id": "JPM_10K_2024.htm#p12", "snippet": "Net income was"}]},
        "machine_check": {"passed": True},
    }


def _validated():
    q = _checked_numeric()
    q["status"] = "human_validated"
    q["validation"] = {"validated_by": "A.H.", "validated_on": "2026-11-20"}
    return q


def _unanswerable():
    return {"id": "u1", "stratum": "unanswerable", "question": "JPM 2031 revenue?",
            "answerable": False, "status": "machine_checked", "origin": "claude_drafted",
            "absence_terms": ["2031 revenue"], "gold": {}, "machine_check": {"passed": True}}


def test_shipped_candidate_file_is_valid_and_has_nothing_scorable():
    data = json.loads((ROOT / "data/eval/eval_questions_v2.json").read_text())
    rep = validate_dataset(data)
    assert rep["problems"] == {}
    assert rep["by_status"]["candidate"] == rep["n_questions"] == 19
    assert all(n >= 3 for n in rep["by_stratum"].values())
    assert scorable(data) == []
    assert all("gold" not in q for q in data["questions"])      # no invented gold


def test_candidate_needs_no_gold_but_checked_items_do():
    c = {"id": "c", "stratum": "single_text", "question": "q", "answerable": True,
         "status": "candidate", "origin": "claude_drafted"}
    assert validate_question(c) == []
    c["status"] = "machine_checked"
    errs = validate_question(c)
    assert "needs gold.evidence" in errs and "needs a passing machine_check record" in errs


def test_human_validated_requires_reviewer_and_machine_check():
    q = _validated()
    assert validate_question(q) == []
    del q["validation"]
    assert any("validated_by" in e for e in validate_question(q))
    q2 = _validated()
    q2["machine_check"] = {"passed": False}
    assert "needs a passing machine_check record" in validate_question(q2)


def test_only_human_validated_is_scorable():
    mc, hv, cand = _checked_numeric(), _validated(), copy.deepcopy(_validated())
    hv["id"], cand["id"], cand["status"] = "h1", "x", "candidate"
    assert [q["id"] for q in scorable({"questions": [mc, hv, cand, _unanswerable()]})] == ["h1"]


def test_evidence_rules_and_unanswerable_rules():
    q = _checked_numeric()
    q["gold"]["evidence"][0] = {"passage_id": "JPM.pdf#12", "snippet": ""}
    errs = validate_question(q)
    assert any("bad passage_id" in e for e in errs) and any("snippet" in e for e in errs)
    q = _checked_numeric()
    del q["gold"]["numeric"]
    assert "single_numeric question needs gold.numeric" in validate_question(q)
    q = _checked_numeric()
    q["stratum"] = "multi_cross_company"
    assert any(">=2 files" in e for e in validate_question(q))
    u = _unanswerable()
    del u["absence_terms"]
    assert any("absence_terms" in e for e in validate_question(u))
    u = _unanswerable()
    u["gold"] = {"evidence": [{"passage_id": "A.htm#p1", "snippet": "x"}]}
    assert any("must not list evidence" in e for e in validate_question(u))
    u = _unanswerable()
    u["answerable"] = True
    assert any("answerable=false" in e for e in validate_question(u))


def test_filter_sensitive_needs_expected_filter():
    q = _checked_numeric()
    q["stratum"] = "filter_sensitive"
    assert "filter_sensitive question needs expected_filter" in validate_question(q)


def test_dataset_duplicates():
    a = _validated()
    rep = validate_dataset({"questions": [a, copy.deepcopy(a)]})
    assert rep["problems"]["n1"] == ["duplicate id"]


def test_machine_check_finds_snippet_and_number():
    q = _checked_numeric()
    assert machine_check(q, PAGES)["passed"]
    assert "58,471" in numeric_forms(q["gold"]["numeric"])


def test_machine_check_fails_on_wrong_snippet_page_or_number():
    q = _checked_numeric()
    q["gold"]["evidence"][0]["snippet"] = "Revenue was"
    assert not machine_check(q, PAGES)["passed"]
    q = _checked_numeric()
    q["gold"]["evidence"][0]["passage_id"] = "JPM_10K_2024.htm#p99"
    assert "page not in corpus" in machine_check(q, PAGES)["problems"][0]
    q = _checked_numeric()
    q["gold"]["numeric"]["value"] = 11111e6
    r = machine_check(q, PAGES)
    assert not r["passed"] and r["numeric_found"] is False


def test_machine_check_number_not_matched_inside_longer_number():
    q = _checked_numeric()
    q["gold"]["numeric"]["value"] = 8471e6
    assert machine_check(q, PAGES)["numeric_found"] is False   # '8,471' is inside '58,471'


def test_machine_check_unanswerable_searches_whole_corpus():
    u = _unanswerable()
    assert machine_check(u, PAGES)["passed"]
    u["absence_terms"] = ["Diluted EPS"]
    r = machine_check(u, PAGES)
    assert not r["passed"] and "found in corpus" in r["problems"][0]
    u["absence_terms"] = []
    assert not machine_check(u, PAGES)["passed"]


def test_apply_machine_check_promotes_but_never_validates():
    cand = {k: v for k, v in _checked_numeric().items() if k != "machine_check"}
    cand["status"] = "candidate"
    out = apply_machine_check(cand, PAGES, "2026-11-20", "abc")
    assert out["status"] == "machine_checked" and out["machine_check"]["corpus_manifest_sha256"] == "abc"
    assert cand["status"] == "candidate"                         # input untouched
    bad = copy.deepcopy(cand)
    bad["gold"]["evidence"][0]["snippet"] = "nope"
    assert apply_machine_check(bad, PAGES, "d", "s")["status"] == "candidate"
    hv = apply_machine_check(_validated(), PAGES, "d", "s")
    assert hv["status"] == "human_validated"                     # status not changed by the check
