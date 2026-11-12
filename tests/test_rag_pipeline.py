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
    def __init__(self, f, t, y, page=86, text="x"):
        self.page_content = text
        self.metadata = {"source_file": f, "ticker": t, "year": y, "page": page}


class _FakeVS:
    """Counts retrievals so tests can assert one search per query."""
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def similarity_search_with_score(self, q, **kw):
        self.calls.append((q, kw))
        return self.hits


def _patch_generate(monkeypatch):
    seen = {}
    monkeypatch.setattr(rp, "generate", lambda q, docs: seen.update(docs=docs) or f"answer to: {q}")
    return seen


def test_smart_ask_single_retrieval_filter_and_sources(monkeypatch):
    seen = _patch_generate(monkeypatch)
    vs = _FakeVS([(_Doc("JPM_10K_2024.pdf", "JPM", "2024", page=86, text="rev"), 0.31)])
    res = rp.smart_ask("JPMorgan revenue 2024?", vs, k=3, verbose=False)
    assert len(vs.calls) == 1                                  # no second retrieval
    assert vs.calls[0][1] == {"k": 3, "filter": {"$and": [{"ticker": "JPM"}, {"year": "2024"}]}}
    assert seen["docs"][0].page_content == "rev"               # same docs used for the prompt
    src = res["sources"][0]
    assert src["passage_id"] == "JPM_10K_2024.pdf#p87"         # 0-based loader page -> 1-based id
    assert src["page"] == 87 and src["score"] == 0.31 and src["text"] == "rev"
    assert res["k"] == 3 and res["filter"]["$and"][0] == {"ticker": "JPM"}


def test_ask_with_filter_explicit_ticker_only(monkeypatch):
    _patch_generate(monkeypatch)
    vs = _FakeVS([])
    rp.ask_with_filter("q", vs, ticker="BAC", verbose=False)
    assert vs.calls[0][1]["filter"] == {"ticker": "BAC"}


def test_ask_question_has_no_filter(monkeypatch):
    _patch_generate(monkeypatch)
    vs = _FakeVS([])
    rp.ask_question("q", vs, verbose=False)
    assert "filter" not in vs.calls[0][1]


def test_passage_id_missing_metadata_is_none():
    assert rp.passage_id({"source_file": "A_10K_2024.pdf"}) is None
    assert rp.passage_id({"page": 3}) is None


def test_query_is_logged_as_json_with_error(monkeypatch, caplog):
    import json, logging
    monkeypatch.setattr(rp, "generate", lambda q, docs: (_ for _ in ()).throw(RuntimeError("boom")))
    vs = _FakeVS([(_Doc("BAC_10K_2023.pdf", "BAC", "2023", page=0), 0.5)])
    with caplog.at_level(logging.INFO, logger="rag"):
        with pytest.raises(RuntimeError):
            rp.ask_question("q", vs, verbose=False)
    rec = json.loads(caplog.records[-1].getMessage())
    assert rec["error"].startswith("RuntimeError") and rec["passages"] == ["BAC_10K_2023.pdf#p1"]


def test_empty_filtered_result_logs_warning(monkeypatch, caplog):
    import logging
    _patch_generate(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="rag"):
        rp.ask_with_filter("q", _FakeVS([]), ticker="C", verbose=False)
    assert any("matched no chunks" in r.getMessage() for r in caplog.records)


def test_labeled_context_and_prompt_version():
    docs = [_Doc("A_10K_2024.htm", "A", "2024", page=0, text="alpha"),
            _Doc("B_10K_2024.htm", "B", "2024", page=4, text="beta")]
    assert rp.format_docs_labeled(docs) == "[S1] alpha\n\n[S2] beta"
    assert "[S2]" in rp.PROMPT_TEMPLATE and rp.PROMPT_VERSION == "v2-cited"
    assert "[S" not in rp.PROMPT_TEMPLATE_V1          # historical prompt unchanged: no citations


def test_extract_citations_maps_labels_and_flags_invented_sources():
    docs = [_Doc("A_10K_2024.htm", "A", "2024", page=0), _Doc("B_10K_2024.htm", "B", "2024", page=4)]
    cited, invalid = rp.extract_citations("Revenue rose [S2]. Also [S1][S2] and [S7].", docs)
    assert cited == ["B_10K_2024.htm#p5", "A_10K_2024.htm#p1"]
    assert invalid == ["S7"]
    assert rp.extract_citations("no citations", docs) == ([], [])
    assert rp.extract_citations("[S1]", [_Doc("A.htm", "A", "1", page=None)])[0] == ["S1"]  # no page metadata


def test_smart_ask_returns_cited_passages_from_the_answer(monkeypatch):
    monkeypatch.setattr(rp, "generate", lambda q, docs: "Net income was 5 [S1].")
    vs = _FakeVS([(_Doc("JPM_10K_2024.htm", "JPM", "2024", page=11), 0.2),
                  (_Doc("JPM_10K_2024.htm", "JPM", "2024", page=12), 0.3)])
    res = rp.smart_ask("JPMorgan net income 2024?", vs, verbose=False)
    assert res["cited_passages"] == ["JPM_10K_2024.htm#p12"] and res["invalid_citations"] == []
    assert res["prompt_version"] == "v2-cited"
    assert {s["passage_id"] for s in res["sources"]} >= set(res["cited_passages"])
