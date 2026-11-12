"""Passage-level retrieval metrics (binary relevance), plus document-level recall.

A retrieval result is a ranked list of passage ids (any hashable, e.g.
"JPM_10K_2024.pdf#p87"); `relevant` is the set of gold passage ids for that
question. These work on any labelled set; the historical 15-question set has
only file-level labels, so only `document_recall_at_k` applies to it.
"""

import math
from collections.abc import Hashable, Sequence


def recall_at_k(ranked: Sequence[Hashable], relevant: set, k: int) -> float:
    """Fraction of relevant items found in the top k. Undefined (NaN) if no relevant items."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        return math.nan
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[Hashable], relevant: set, k: int) -> float:
    """1.0 if at least one relevant item is in the top k (document-level 'recall@k' in the README)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        return math.nan
    return float(bool(set(ranked[:k]) & relevant))


def reciprocal_rank(ranked: Sequence[Hashable], relevant: set) -> float:
    """1/rank of the first relevant item; 0 if none retrieved; NaN if no relevant items."""
    if not relevant:
        return math.nan
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[Hashable], relevant: set, k: int) -> float:
    """Binary-relevance nDCG@k. Duplicate retrieved ids only count once (first occurrence)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        return math.nan
    seen: set = set()
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant and item not in seen:
            dcg += 1.0 / math.log2(i + 1)
        seen.add(item)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def evaluate_retrieval(results: Sequence[tuple[Sequence[Hashable], set]], ks=(1, 4, 10)) -> dict:
    """Mean of each metric over questions, skipping questions with no gold passages (counted)."""
    scored = [(r, rel) for r, rel in results if rel]
    out: dict = {"n_questions": len(results), "n_scored": len(scored)}
    if not scored:
        return out
    n = len(scored)
    out["mrr"] = sum(reciprocal_rank(r, rel) for r, rel in scored) / n
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in scored) / n
        out[f"hit@{k}"] = sum(hit_at_k(r, rel, k) for r, rel in scored) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, rel, k) for r, rel in scored) / n
    return out


def file_ids(passage_ids: Sequence[Hashable]) -> list:
    """Collapse passage ids ('file#pN') to file ids, keeping rank order (duplicates kept).

    File-level relevance is a coarser question than passage-level relevance: a file
    counts as hit if ANY of its passages was retrieved, even a wrong page. Always
    report the two separately.
    """
    return [str(p).split("#", 1)[0] for p in passage_ids if p is not None]


def multi_doc_coverage(ranked_files: Sequence[Hashable], gold_files: set, k: int) -> dict:
    """For questions needing evidence from several documents.

    coverage = fraction of gold files with >=1 retrieved chunk in the top k;
    all_hit = 1.0 if every gold file is covered. NaN if no gold files.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not gold_files:
        return {"coverage": math.nan, "all_hit": math.nan}
    found = set(ranked_files[:k]) & gold_files
    return {"coverage": len(found) / len(gold_files), "all_hit": float(found == gold_files)}


def filter_accuracy(detected: Sequence[tuple], gold: Sequence[tuple]) -> dict:
    """Metadata-filter detection accuracy over (ticker, year) pairs (None = no filter).

    Reports exact-match rate, plus per-field accuracy and the two error kinds that
    matter: `wrong_filter` (a filter was applied but differs from gold, which can
    exclude the gold document) and `missed_filter` (gold wanted a filter, none detected).
    """
    if len(detected) != len(gold):
        raise ValueError("detected and gold must have the same length")
    n = len(gold)
    if n == 0:
        return {"n": 0}
    exact = sum(d == g for d, g in zip(detected, gold))
    tick = sum(d[0] == g[0] for d, g in zip(detected, gold))
    year = sum(d[1] == g[1] for d, g in zip(detected, gold))
    wrong = sum(
        any(dv is not None and dv != gv for dv, gv in zip(d, g)) for d, g in zip(detected, gold)
    )
    missed = sum(
        any(dv is None and gv is not None for dv, gv in zip(d, g)) for d, g in zip(detected, gold)
    )
    return {"n": n, "exact": exact / n, "ticker": tick / n, "year": year / n,
            "wrong_filter": wrong / n, "missed_filter": missed / n}
