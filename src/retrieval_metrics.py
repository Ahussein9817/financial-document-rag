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
