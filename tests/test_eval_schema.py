import copy

from eval_schema import scorable, validate_dataset, validate_question


def _numeric():
    return {
        "id": "n1", "stratum": "single_numeric", "question": "JPM net income FY2024?",
        "answerable": True,
        "gold": {"numeric": {"value": 58471e6, "unit": "USD", "scale": "million", "period": "FY2024"},
                 "evidence": [{"passage_id": "JPM_10K_2024.pdf#p12", "snippet": "net income"}]},
        "validation": {"status": "validated", "validated_by": "A.H.", "validated_on": "2026-11-20"},
    }


def _unanswerable():
    return {
        "id": "u1", "stratum": "unanswerable", "question": "JPM 2031 revenue?", "answerable": False,
        "gold": {}, "validation": {"status": "validated", "validated_by": "A.H.",
                                   "validated_on": "2026-11-20",
                                   "absence_check": "full-text search of all filings for 2031 guidance"},
    }


def test_valid_items_pass():
    assert validate_question(_numeric()) == []
    assert validate_question(_unanswerable()) == []


def test_validated_requires_reviewer():
    q = _numeric()
    q["validation"] = {"status": "validated"}
    assert "validated item needs validated_by and validated_on" in validate_question(q)


def test_bad_passage_and_missing_snippet():
    q = _numeric()
    q["gold"]["evidence"][0] = {"passage_id": "JPM.pdf#12", "snippet": ""}
    errs = validate_question(q)
    assert any("bad passage_id" in e for e in errs) and any("snippet" in e for e in errs)


def test_numeric_stratum_requires_numeric_gold():
    q = _numeric()
    del q["gold"]["numeric"]
    assert "single_numeric question needs gold.numeric" in validate_question(q)


def test_multi_needs_two_files_and_unanswerable_needs_absence_check():
    q = _numeric()
    q["stratum"] = "multi_cross_company"
    assert any(">=2 files" in e for e in validate_question(q))
    u = _unanswerable()
    del u["validation"]["absence_check"]
    assert any("absence_check" in e for e in validate_question(u))
    u2 = _unanswerable()
    u2["gold"] = {"evidence": [{"passage_id": "A_10K_2024.pdf#p1", "snippet": "x"}]}
    assert any("must not list evidence" in e for e in validate_question(u2))


def test_dataset_counts_duplicates_and_scorable():
    a, b = _numeric(), _unanswerable()
    proposed = copy.deepcopy(a)
    proposed["id"] = "n2"
    proposed["validation"] = {"status": "proposed"}
    dup = copy.deepcopy(a)
    rep = validate_dataset({"questions": [a, b, proposed, dup]})
    assert rep["n_questions"] == 4 and rep["n_validated"] == 3
    assert rep["problems"]["n1"] == ["duplicate id"]
    assert [q["id"] for q in scorable({"questions": [a, b, proposed]})] == ["n1", "u1"]
