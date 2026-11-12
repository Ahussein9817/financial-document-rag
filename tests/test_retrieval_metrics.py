import math

import pytest

import retrieval_metrics as rm


def test_recall_at_k_known():
    ranked = ["a", "b", "c", "d"]
    assert rm.recall_at_k(ranked, {"a", "d"}, 1) == 0.5
    assert rm.recall_at_k(ranked, {"a", "d"}, 4) == 1.0
    assert rm.recall_at_k(ranked, {"z"}, 4) == 0.0


def test_hit_at_k_differs_from_recall_with_multiple_relevant():
    ranked = ["a", "b"]
    assert rm.hit_at_k(ranked, {"a", "z"}, 2) == 1.0
    assert rm.recall_at_k(ranked, {"a", "z"}, 2) == 0.5


def test_reciprocal_rank():
    assert rm.reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert rm.reciprocal_rank(["x"], {"a"}) == 0.0
    assert rm.reciprocal_rank(["a"], {"a", "b"}) == 1.0


def test_ndcg_perfect_and_worse_orderings():
    rel = {"a", "b"}
    assert rm.ndcg_at_k(["a", "b", "x"], rel, 3) == pytest.approx(1.0)
    worse = rm.ndcg_at_k(["x", "a", "b"], rel, 3)
    assert 0 < worse < 1
    # hand check: dcg = 1/log2(3) + 1/log2(4); ideal = 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert worse == pytest.approx(expected)


def test_ndcg_duplicates_not_double_counted_and_none_found_is_zero():
    assert rm.ndcg_at_k(["a", "a", "a"], {"a"}, 3) == pytest.approx(1.0)
    assert rm.ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_empty_gold_is_nan_and_bad_k_raises():
    assert math.isnan(rm.recall_at_k(["a"], set(), 1))
    assert math.isnan(rm.ndcg_at_k(["a"], set(), 1))
    assert math.isnan(rm.reciprocal_rank(["a"], set()))
    for f in (rm.recall_at_k, rm.hit_at_k, rm.ndcg_at_k):
        with pytest.raises(ValueError):
            f(["a"], {"a"}, 0)


def test_evaluate_retrieval_means_and_skips_unlabelled():
    res = [(["a", "b"], {"a"}), (["x", "a"], {"a"}), (["q"], set())]
    out = rm.evaluate_retrieval(res, ks=(1, 2))
    assert out["n_questions"] == 3 and out["n_scored"] == 2
    assert out["mrr"] == pytest.approx(0.75)
    assert out["hit@1"] == 0.5 and out["hit@2"] == 1.0
    assert rm.evaluate_retrieval([(["a"], set())]) == {"n_questions": 1, "n_scored": 0}


def test_document_level_recall_matches_committed_historical_value():
    """Re-derive the README's 13/15 from the stored retrieved file lists (hit@4, file-level)."""
    import ast

    import pandas as pd
    df = pd.read_csv("data/processed/retrieval_recall_analysis.csv")
    # expected_source may list several files ("A.pdf, B.pdf"); the notebook counts a hit if ANY is retrieved
    results = [(ast.literal_eval(r), {x.strip() for x in e.split(",")})
               for r, e in zip(df["retrieved_sources"], df["expected_source"])]
    out = rm.evaluate_retrieval(results, ks=(4,))
    assert out["hit@4"] == pytest.approx(13 / 15)  # README's document-level recall@4
    # stricter: fraction of ALL expected files retrieved (multi-document questions penalised)
    assert out["recall@4"] < out["hit@4"]
    print("strict file-level recall@4:", out["recall@4"], "MRR:", out["mrr"])
