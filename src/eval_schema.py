"""Schema, lifecycle and checks for the v2 evaluation set (data/eval/eval_questions_v2.json).

Every item has one of three statuses, and only the last is gold-standard data:

  candidate        question drafted, no gold yet (may say where to look: `target`).
                   Never scored.
  machine_checked  gold evidence (passage id + verbatim snippet, numeric value for numeric
                   items) was found in the corpus text by `apply_machine_check`. This proves
                   the snippet exists on that page, NOT that the question or answer is right.
                   Never scored.
  human_validated  a person confirmed question, gold answer and evidence against the filing
                   (`validated_by`, `validated_on`). The only status `scorable` returns.

`rejected` marks items dropped after review (kept for the audit trail, never scored).
Nothing in this module can set `human_validated`; that edit is made by hand.
"""

import re

STRATA = ("single_numeric", "single_text", "multi_same_company", "multi_cross_company",
          "filter_sensitive", "unanswerable")
STATUSES = ("candidate", "machine_checked", "human_validated", "rejected")
ORIGINS = ("claude_drafted", "human_drafted", "historical_v1")
PASSAGE_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(?:pdf|htm|html)#p\d+$")
_SCALES = {"unit": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9}


def _validate_evidence(q: dict, errs: list[str]) -> None:
    g = q["gold"]
    ev = g.get("evidence", [])
    if not ev:
        errs.append("needs gold.evidence")
    for e in ev:
        if not PASSAGE_RE.match(e.get("passage_id", "")):
            errs.append(f"bad passage_id {e.get('passage_id')!r}")
        if not e.get("snippet"):
            errs.append("evidence needs a verbatim snippet")
    if not g.get("answer"):
        errs.append("needs gold.answer")
    num = g.get("numeric")
    if q["stratum"] == "single_numeric" and not num:
        errs.append("single_numeric question needs gold.numeric")
    if num:
        for f in ("value", "unit", "scale", "period"):
            if f not in num:
                errs.append(f"gold.numeric missing '{f}'")
        if num.get("scale") not in _SCALES:
            errs.append(f"gold.numeric.scale must be one of {sorted(_SCALES)}")
    if q["stratum"].startswith("multi") and len({e["passage_id"].split("#")[0] for e in ev if "passage_id" in e}) < 2:
        errs.append("multi-document question needs evidence from >=2 files")
    if q["stratum"] == "filter_sensitive" and "expected_filter" not in q:
        errs.append("filter_sensitive question needs expected_filter")


def validate_question(q: dict) -> list[str]:
    """Problems with one item (empty list if well-formed for its declared status)."""
    errs = [f"missing '{k}'" for k in
            ("id", "stratum", "question", "answerable", "status", "origin") if k not in q]
    if errs:
        return errs
    if q["stratum"] not in STRATA:
        errs.append(f"unknown stratum {q['stratum']!r}")
    if q["status"] not in STATUSES:
        errs.append(f"bad status {q['status']!r}")
        return errs
    if q["origin"] not in ORIGINS:
        errs.append(f"bad origin {q['origin']!r}")
    if q["stratum"] == "unanswerable" and q["answerable"]:
        errs.append("unanswerable stratum must have answerable=false")
    if q["status"] in ("candidate", "rejected"):
        if q["status"] == "candidate" and q.get("gold", {}).get("evidence") and "machine_check" not in q:
            errs.append("candidate with gold evidence should be run through machine check")
        return errs
    # machine_checked and human_validated both need full gold + a machine-check record
    q.setdefault("gold", {})
    if q["answerable"]:
        _validate_evidence(q, errs)
    else:
        if q["gold"].get("evidence"):
            errs.append("unanswerable question must not list evidence")
        if not q.get("absence_terms"):
            errs.append("unanswerable question needs absence_terms (searched across the whole corpus)")
    mc = q.get("machine_check")
    if not mc or not mc.get("passed"):
        errs.append("needs a passing machine_check record")
    if q["status"] == "human_validated":
        v = q.get("validation", {})
        if not (v.get("validated_by") and v.get("validated_on")):
            errs.append("human_validated item needs validation.validated_by and validated_on")
    return errs


def validate_dataset(data: dict) -> dict:
    """Validate every item; unique ids; counts per status and per (validated) stratum."""
    qs = data.get("questions", [])
    problems: dict = {}
    ids = [q.get("id") for q in qs]
    for i in {x for x in ids if ids.count(x) > 1}:
        problems.setdefault(i, []).append("duplicate id")
    for q in qs:
        e = validate_question(q)
        if e:
            problems.setdefault(q.get("id"), []).extend(e)
    by_status = {s: 0 for s in STATUSES}
    by_stratum = {s: 0 for s in STRATA}
    for q in qs:
        by_status[q.get("status")] = by_status.get(q.get("status"), 0) + 1
        by_stratum[q.get("stratum")] = by_stratum.get(q.get("stratum"), 0) + 1
    return {"n_questions": len(qs), "by_status": by_status, "by_stratum": by_stratum,
            "problems": problems}


def scorable(data: dict) -> list[dict]:
    """Gold-standard items only: status human_validated and no schema problems."""
    return [q for q in data.get("questions", [])
            if q.get("status") == "human_validated" and not validate_question(q)]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def numeric_forms(num: dict) -> set[str]:
    """Strings under which a gold numeric value could be printed on the page, at the stated
    scale (224532e6 at scale 'million' -> '224,532'). Only renderings that round-trip to the
    gold value within 0.005 are kept, so 11.8 never matches a page that says '12'."""
    v = num["value"] / _SCALES[num["scale"]]
    forms = {f"{v:,.{d}f}" for d in (0, 1, 2)} | {f"{v:.{d}f}" for d in (0, 1, 2)}
    return {f for f in forms if abs(float(f.replace(",", "")) - v) <= 0.005}


def machine_check(q: dict, pages: dict[str, str]) -> dict:
    """Do the gold snippets occur on their cited pages, and the numeric value on a cited page?

    `pages` maps passage id -> full page text from the corpus. Returns
    {"passed", "snippets_found", "numeric_found"(or None), "problems"}. A pass means the
    evidence text exists where cited; it does not validate the question or the answer.
    For unanswerable items, passes only if NO `absence_terms` occur anywhere in the corpus
    (a heuristic: absence of a string is not proof the topic is absent; a human decides).
    """
    if not q.get("answerable", True):
        terms = q.get("absence_terms", [])
        hits = [t for t in terms if any(_norm(t) in _norm(txt) for txt in pages.values())]
        return {"passed": bool(terms) and not hits, "snippets_found": None, "numeric_found": None,
                "problems": ([] if terms else ["no absence_terms to search for"])
                + [f"search term found in corpus: {t!r}" for t in hits]}
    problems, snippets_ok = [], True
    ev = q.get("gold", {}).get("evidence", [])
    for e in ev:
        text = pages.get(e["passage_id"])
        if text is None:
            snippets_ok = False
            problems.append(f"page not in corpus: {e['passage_id']}")
        elif _norm(e["snippet"]) not in _norm(text):
            snippets_ok = False
            problems.append(f"snippet not found on {e['passage_id']}")
    numeric_ok = None
    num = q.get("gold", {}).get("numeric")
    if num:
        cited_text = _norm(" ".join(pages.get(e["passage_id"], "") for e in ev))
        numeric_ok = any(re.search(rf"(?<![\d,.]){re.escape(f)}(?![\d,])", cited_text) for f in numeric_forms(num))
        if not numeric_ok:
            problems.append("gold numeric value not found on cited pages")
    return {"passed": bool(ev) and snippets_ok and numeric_ok in (None, True),
            "snippets_found": snippets_ok, "numeric_found": numeric_ok, "problems": problems}


def apply_machine_check(q: dict, pages: dict[str, str], checked_on: str, corpus_sha: str) -> dict:
    """Return a copy of a candidate with gold evidence, promoted to machine_checked if the
    check passes. Never produces human_validated."""
    out = {**q, "gold": dict(q.get("gold", {}))}
    res = machine_check(out, pages)
    out["machine_check"] = {**res, "checked_on": checked_on, "corpus_manifest_sha256": corpus_sha}
    if res["passed"] and q.get("status") in ("candidate", "machine_checked"):
        out["status"] = "machine_checked"
    return out
