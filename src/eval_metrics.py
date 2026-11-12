"""Pure, offline aggregation of the manual evaluation (no API calls).

Scoring itself is manual (1 correct / 0.5 partial / 0 incorrect, one grader);
these functions only aggregate committed scores and retrieval flags.
"""

import pandas as pd


def summarize_scores(df: pd.DataFrame) -> dict:
    """Overall and per-category accuracy from a frame with `category`, `score`."""
    if df.empty:
        raise ValueError("no evaluation rows")
    if not df["score"].between(0, 1).all():
        raise ValueError("scores must be within [0, 1]")
    by_cat = df.groupby("category")["score"].agg(["mean", "count"])
    return {
        "n_questions": int(len(df)),
        "overall_accuracy_pct": 100 * float(df["score"].mean()),
        "n_correct": int((df["score"] == 1.0).sum()),
        "n_partial": int((df["score"] == 0.5).sum()),
        "n_incorrect": int((df["score"] == 0.0).sum()),
        "category_accuracy_pct": {c: 100 * float(r["mean"]) for c, r in by_cat.iterrows()},
        "category_count": {c: int(r["count"]) for c, r in by_cat.iterrows()},
    }


def document_recall(df: pd.DataFrame) -> dict:
    """Document-level recall: fraction of questions whose expected source FILE
    appears among the retrieved chunks. Not passage-level recall."""
    if df.empty:
        raise ValueError("no evaluation rows")
    ok = df["recall_success"].astype(bool)
    return {
        "recall_pct": 100 * float(ok.mean()),
        "n_success": int(ok.sum()),
        "n_total": int(len(df)),
        "accuracy_pct_when_retrieved": 100 * float(df.loc[ok, "answer_score"].mean()) if ok.any() else None,
        "accuracy_pct_when_not_retrieved": 100 * float(df.loc[~ok, "answer_score"].mean()) if (~ok).any() else None,
    }
