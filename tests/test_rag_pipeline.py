import pytest

import rag_pipeline as rp


@pytest.mark.parametrize("q,ticker,year,flt", [
    ("What was JPMorgan's total revenue in 2024?", "JPM", 2024,
     {"$and": [{"ticker": "JPM"}, {"year": "2024"}]}),
    ("Bank of America CET1 ratio 2023", "BAC", 2023,
     {"$and": [{"ticker": "BAC"}, {"year": "2023"}]}),
    ("What are Citigroup's main revenue sources?", "C", None, {"ticker": "C"}),
    ("What happened in 2022?", None, 2022, {"year": "2022"}),
    ("What is a 10-K?", None, None, None),
])
def test_detect_filters(q, ticker, year, flt):
    assert rp.detect_filters(q) == (ticker, year, flt)


def test_detect_filters_year_outside_hardcoded_range_is_ignored():
    # documents a limitation: only 2022-2025 are recognised
    assert rp.detect_filters("JPMorgan revenue in 2021")[1] is None


@pytest.mark.parametrize("q", [
    "Summarize the feedback loop described in risk factors",   # contains 'bac' inside 'feedback'
    "How does the firm manage a backlog of requests?",          # 'bac' inside 'backlog'
    "Which cities have the most branches?",                     # 'citi' inside 'cities'
])
def test_detect_filters_does_not_match_inside_other_words(q):
    assert rp.detect_filters(q)[0] is None


def test_detect_filters_company_alias_matches_whole_words():
    assert rp.detect_filters("What did BofA report in 2024?")[0] == "BAC"
    assert rp.detect_filters("jp morgan 2023 net income")[0] == "JPM"


def test_format_docs_joins_page_content():
    class D:
        def __init__(self, t): self.page_content = t  # noqa: E701
    assert rp.format_docs([D("a"), D("b")]) == "a\n\nb"


class _Doc:
    def __init__(self, f, t, y):
        self.page_content = "x"
        self.metadata = {"source_file": f, "ticker": t, "year": y}


class _FakeChain:
    def invoke(self, q): return f"answer to: {q}"  # noqa: E701


class _FakeRetriever:
    def invoke(self, q): return [_Doc("JPM_10K_2024.pdf", "JPM", "2024")]  # noqa: E701


def test_smart_ask_builds_filter_and_formats_result(monkeypatch):
    seen = {}

    def fake_build(vs, k=4, filter_dict=None):
        seen["k"], seen["filter"] = k, filter_dict
        return _FakeChain(), _FakeRetriever()

    monkeypatch.setattr(rp, "build_chain", fake_build)
    res = rp.smart_ask("JPMorgan revenue 2024?", vectorstore=None, k=3, verbose=False)
    assert seen == {"k": 3, "filter": {"$and": [{"ticker": "JPM"}, {"year": "2024"}]}}
    assert res["answer"].startswith("answer to:")
    assert res["sources"] == [{"file": "JPM_10K_2024.pdf", "ticker": "JPM", "year": "2024"}]
    assert res["k"] == 3


def test_ask_with_filter_explicit_ticker_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(rp, "build_chain",
                        lambda vs, k=4, filter_dict=None: (seen.setdefault("f", filter_dict), (_FakeChain(), _FakeRetriever()))[1])
    rp.ask_with_filter("q", None, ticker="BAC", verbose=False)
    assert seen["f"] == {"ticker": "BAC"}
