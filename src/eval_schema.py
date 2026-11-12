"""Schema and consistency checks for the v2 evaluation set (data/eval/eval_questions_v2.json).

Only questions with `validation.status == "validated"` may be scored; the rest are
reported as excluded. Gold evidence is page-level: file + 1-based page + a short
verbatim snippet so a human can re-check each item against the filing.
"""

import re

STRATA = ("single_numeric", "single_text", "multi_same_company", "multi_cross_company",
          "filter_sensitive", "unanswerable")
STATUSES = ("proposed", "validated", "rejected")
PASSAGE_RE = re.compile(r"^[A-Z]+_10K_\d{4}\.pdf#p\d+$")


def validate_question(q: dict) -> list[str]:
    """Return a list of problems (empty if the item is well-formed)."""
    errs = []
    for key in ("id", "stratum", "question", "answerable", "gold", "validation"):
        if key not in q:
            errs.append(f"missing '{key}'")
    if errs:
        return errs
    if q["stratum"] not in STRATA:
        errs.append(f"unknown stratum {q['stratum']!r}")
    v = q["validation"]
    if v.get("status") not in STATUSES:
        errs.append(f"bad validation.status {v.get('status')!r}")
    if v.get("status") == "validated" and not (v.get("validated_by") and v.get("validated_on")):
        errs.append("validated item needs validated_by and validated_on")
    g = q["gold"]
    if q["answerable"]:
        ev = g.get("evidence", [])
        if not ev:
            errs.append("answerable question needs gold.evidence")
        for e in ev:
            if not PASSAGE_RE.match(e.get("passage_id", "")):
                errs.append(f"bad passage_id {e.get('passage_id')!r}")
            if not e.get("snippet"):
                errs.append("evidence needs a verbatim snippet")
        num = g.get("numeric")
        if q["stratum"] == "single_numeric" and not num:
            errs.append("single_numeric question needs gold.numeric")
        if num:
            for f in ("value", "unit", "scale", "period"):
                if f not in num:
                    errs.append(f"gold.numeric missing '{f}'")
        if q["stratum"].startswith("multi") and len({e["passage_id"].split("#")[0] for e in ev}) < 2:
            errs.append("multi-document question needs evidence from >=2 files")
    else:
        if g.get("evidence"):
            errs.append("unanswerable question must not list evidence")
        if not v.get("absence_check"):
            errs.append("unanswerable question needs validation.absence_check (how absence was confirmed)")
    return errs


def validate_dataset(data: dict) -> dict:
    """Validate every question; also check unique ids. Returns problems and counts."""
    qs = data.get("questions", [])
    problems = {}
    ids = [q.get("id") for q in qs]
    for i in {x for x in ids if ids.count(x) > 1}:
        problems.setdefault(i, []).append("duplicate id")
    for q in qs:
        e = validate_question(q)
        if e:
            problems.setdefault(q.get("id"), []).extend(e)
    counts = {s: 0 for s in STRATA}
    n_valid = 0
    for q in qs:
        if q.get("validation", {}).get("status") == "validated":
            n_valid += 1
            counts[q.get("stratum")] = counts.get(q.get("stratum"), 0) + 1
    return {"n_questions": len(qs), "n_validated": n_valid, "validated_by_stratum": counts,
            "problems": problems}


def scorable(data: dict) -> list[dict]:
    """Only validated, well-formed questions."""
    return [q for q in data.get("questions", [])
            if q.get("validation", {}).get("status") == "validated" and not validate_question(q)]
