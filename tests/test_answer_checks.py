import math

import pytest

from answer_checks import (abstention_scores, citation_accuracy, extract_numbers, is_abstention,
                           numeric_groundedness, numeric_match)


def test_extract_numbers_scales_and_years():
    nums = extract_numbers("Revenue was $224.5 billion in 2024, up 12.5%, vs 1,234 million")
    scaled = sorted(n["scaled"] for n in nums if not n["is_percent"])
    assert scaled == pytest.approx([1.234e9, 2.245e11])
    assert [n["value"] for n in nums if n["is_percent"]] == [12.5]
    assert not any(n["value"] == 2024 for n in nums)          # years skipped


def test_numeric_match_across_scales():
    gold = 224_532e6
    assert numeric_match("about $224.5 billion", gold)["match"]
    assert numeric_match("224,532 million dollars", gold)["match"]
    r = numeric_match("$200 billion", gold)
    assert not r["match"] and not r["unit_error"]


def test_numeric_match_flags_unit_error():
    r = numeric_match("$224,532 billion", 224_532e6)
    assert not r["match"] and r["unit_error"]


def test_numeric_match_percent_vs_dollar_kept_separate():
    assert numeric_match("CET1 was 15.7%", 15.7, gold_is_percent=True)["match"]
    assert not numeric_match("15.7 billion", 15.7, gold_is_percent=True)["match"]


def test_numeric_match_no_numbers():
    assert numeric_match("I don't know", 5.0) == {"match": False, "unit_error": False, "matched": None}


def test_groundedness_detects_unsupported_numbers():
    ctx = ["Total net revenue was 177,556 million and net income 58,471 million."]
    r = numeric_groundedness("Revenue 177,556 million, net income 58,471 million, margin 99.9%", ctx)
    assert r["n_numbers"] == 3 and r["unsupported"] == ["99.9%"]
    assert r["grounded_fraction"] == pytest.approx(2 / 3)


def test_groundedness_comma_insensitive_and_no_numbers_nan():
    assert numeric_groundedness("177556 million", ["177,556 million"])["n_unsupported"] == 0
    assert math.isnan(numeric_groundedness("no figures here", ["x"])["grounded_fraction"])


def test_citation_accuracy():
    r = citation_accuracy(["A#p1", "A#p9", "Z#p1"], ["A#p1", "A#p9"], {"A#p1"})
    assert r["n_cited"] == 3
    assert r["valid"] == pytest.approx(2 / 3) and r["correct"] == pytest.approx(1 / 3)
    assert math.isnan(citation_accuracy([], ["A#p1"], {"A#p1"})["valid"])


def test_is_abstention():
    assert is_abstention("The context does not contain this information.")
    assert is_abstention("I don't know.")
    assert not is_abstention("Revenue was 5 billion.")


def test_abstention_scores():
    rows = [
        {"answerable": False, "abstained": True}, {"answerable": False, "abstained": False},
        {"answerable": True, "abstained": True}, {"answerable": True, "abstained": False},
    ]
    r = abstention_scores(rows)
    assert r["recall"] == 0.5 and r["precision"] == 0.5 and r["false_abstain_rate"] == 0.5
    assert math.isnan(abstention_scores([{"answerable": True, "abstained": False}])["recall"])
