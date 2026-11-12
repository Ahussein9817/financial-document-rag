import json

import pandas as pd
import pytest

import eval_metrics as em
import reproduce_reported_metrics as rep
import run_eval


def test_summarize_scores_known_values():
    df = pd.DataFrame({"category": ["a", "a", "b", "b"], "score": [1.0, 0.5, 0.0, 1.0]})
    s = em.summarize_scores(df)
    assert s["overall_accuracy_pct"] == pytest.approx(62.5)
    assert (s["n_correct"], s["n_partial"], s["n_incorrect"]) == (2, 1, 1)
    assert s["category_accuracy_pct"] == {"a": 75.0, "b": 50.0}


def test_summarize_scores_validates():
    with pytest.raises(ValueError):
        em.summarize_scores(pd.DataFrame({"category": [], "score": []}))
    with pytest.raises(ValueError):
        em.summarize_scores(pd.DataFrame({"category": ["a"], "score": [2.0]}))


def test_document_recall_known_values_and_empty_branches():
    df = pd.DataFrame({"recall_success": [True, True, False], "answer_score": [1.0, 0.5, 0.0]})
    r = em.document_recall(df)
    assert r["recall_pct"] == pytest.approx(200 / 3)
    assert r["accuracy_pct_when_retrieved"] == pytest.approx(75.0)
    assert r["accuracy_pct_when_not_retrieved"] == 0.0
    all_ok = em.document_recall(df.iloc[:2])
    assert all_ok["accuracy_pct_when_not_retrieved"] is None


def test_committed_historical_numbers_reproduce(tmp_path, monkeypatch):
    """The README's aggregate numbers recomputed from the committed CSVs."""
    monkeypatch.setattr(rep, "ROOT", rep.ROOT)  # real repo files
    out = rep.main()
    a, r = out["answer_scores"], out["retrieval"]
    assert a["n_questions"] == 15
    assert a["overall_accuracy_pct"] == pytest.approx(53.333, abs=0.01)
    assert (a["n_correct"], a["n_partial"], a["n_incorrect"]) == (6, 4, 5)
    assert a["category_accuracy_pct"]["factual"] == pytest.approx(65.0)  # README prose says 60%
    assert r["n_success"] == 13 and r["recall_pct"] == pytest.approx(86.667, abs=0.01)


def test_question_file_is_well_formed():
    data = json.load(open(rep.ROOT / "data/eval/eval_questions_v1.json"))
    qs = data["questions"]
    assert len(qs) == 15
    assert {q["category"] for q in qs} == {"factual", "comparative", "trend"}
    assert all(q["question"] and q["ground_truth"] and q["source"] for q in qs)


def test_run_questions_records_latency_sources_and_errors():
    def ask(q, k):
        if "boom" in q:
            raise RuntimeError("api down")
        return {"answer": "A", "sources": [{"file": "JPM_10K_2024.pdf"}]}

    rows = run_eval.run_questions(
        [{"id": 1, "question": "ok", "source": "JPM_10K_2024.pdf", "category": "factual"},
         {"id": 2, "question": "boom"}], ask, k=4)
    assert rows[0]["retrieved_files"] == ["JPM_10K_2024.pdf"] and rows[0]["error"] is None
    assert rows[0]["latency_s"] >= 0
    assert rows[1]["answer"] is None and "api down" in rows[1]["error"]
