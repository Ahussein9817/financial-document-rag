"""Deterministic answer checks (no LLM): numeric correctness, numeric groundedness,
citation accuracy and abstention.

All functions are pure. They operate on text, so they can be unit-tested and rerun
from stored run artifacts without API calls. Limits are stated in each docstring;
none of these replaces human review of free-text correctness.
"""

import math
import re

_SCALE = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mm": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

# optional $, number with thousands separators/decimals, optional scale word.
# "(1,234)" negative parentheses are not handled specially: callers use gold values with signs.
_NUM = re.compile(
    r"(?P<neg>-|−)?\s*\$?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<scale>trillion|billion|million|thousand|tn|bn|mm|[tbmk])?\b(?P<pct>\s*%)?",
    re.IGNORECASE,
)

ABSTAIN_PHRASES = (
    "i don't know", "i do not know", "not enough information", "insufficient information",
    "cannot be determined", "can't be determined", "not provided in the context",
    "not mentioned in the context", "does not contain", "doesn't contain",
    "no information", "not specified in the", "unable to find", "cannot answer",
    "context does not", "not available in the",
)


def extract_numbers(text: str) -> list[dict]:
    """Numbers in text as {raw, value, scaled, is_percent}. `scaled` applies a scale word
    (million/billion/...) so '$224.5 billion' and '224,532 million' become comparable.

    Years (4-digit numbers 1900-2100 with no comma/scale/decimal) are skipped.
    """
    out = []
    for m in _NUM.finditer(text or ""):
        raw_num = m.group("num")
        value = float(raw_num.replace(",", ""))
        scale = m.group("scale")
        is_pct = bool(m.group("pct"))
        if (scale is None and not is_pct and "," not in raw_num and "." not in raw_num
                and 1900 <= value <= 2100):
            continue
        if m.group("neg"):
            value = -value
        mult = _SCALE.get(scale.lower(), 1.0) if scale else 1.0
        out.append({"raw": m.group(0).strip(), "digits": raw_num, "value": value, "scaled": value * mult,
                    "is_percent": is_pct})
    return out


def numeric_match(answer: str, gold_value: float, rel_tol: float = 0.005,
                  gold_is_percent: bool = False) -> dict:
    """Does any number in the answer equal the gold value (after scale normalization)?

    `gold_value` is in base units (e.g. 224_532e6 dollars) or percent points when
    `gold_is_percent`. rel_tol 0.5% default tolerates '224.5 billion' vs '224,532 million'.
    `unit_error` is set when the digits match some scale of the gold (x1e3 or 1e-3 off) but
    the value does not, so unit mistakes are reported separately from value mistakes.
    """
    cands = [n for n in extract_numbers(answer) if n["is_percent"] == gold_is_percent]
    key = "value" if gold_is_percent else "scaled"
    for n in cands:
        if math.isclose(n[key], gold_value, rel_tol=rel_tol, abs_tol=0.0):
            return {"match": True, "unit_error": False, "matched": n["raw"]}
    unit_error = False
    if gold_value:
        for n in cands:
            ratio = n[key] / gold_value if gold_value else math.nan
            for f in (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
                if math.isclose(ratio, f, rel_tol=rel_tol):
                    unit_error = True
    return {"match": False, "unit_error": unit_error, "matched": None}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("−", "-"))


def numeric_groundedness(answer: str, context_chunks: list[str]) -> dict:
    """Share of answer numbers whose digits appear in the retrieved text.

    A number is 'supported' if its digit string (commas kept or removed, decimals kept)
    occurs in some chunk. This is a string check: it catches numbers not in the context
    (likely hallucinated or computed) but cannot tell whether the number answers the
    question or is used correctly. Derived numbers (sums, growth rates) will count as
    unsupported by design; report them separately rather than treating them as errors.
    """
    context = _norm(" ".join(context_chunks))
    context_nocomma = context.replace(",", "")
    nums = extract_numbers(answer)
    unsupported = []
    for n in nums:
        digits = n["digits"]
        if digits not in context and digits.replace(",", "") not in context_nocomma:
            unsupported.append(n["raw"])
    total = len(nums)
    return {"n_numbers": total, "n_unsupported": len(unsupported), "unsupported": unsupported,
            "grounded_fraction": (1 - len(unsupported) / total) if total else math.nan}


def citation_accuracy(cited: list[str], retrieved: list[str], gold: set) -> dict:
    """Citations are passage ids ('file#pN') the answer claims to rely on.

    valid = cited id is among the retrieved passages (not invented);
    correct = cited id is also a gold passage. Precision over citations; NaN if none cited.
    """
    if not cited:
        return {"n_cited": 0, "valid": math.nan, "correct": math.nan}
    ret = set(retrieved)
    return {"n_cited": len(cited),
            "valid": sum(c in ret for c in cited) / len(cited),
            "correct": sum(c in gold for c in cited) / len(cited)}


def is_abstention(answer: str) -> bool:
    """True if the answer contains a fixed refusal phrase. Phrase list is a heuristic:
    check its false-negative rate against manual reading before trusting it."""
    a = _norm(answer or "")
    return any(p in a for p in ABSTAIN_PHRASES)


def abstention_scores(rows: list[dict]) -> dict:
    """rows: {'answerable': bool, 'abstained': bool}. Reports abstention precision/recall on
    the unanswerable stratum and the false-abstain rate on answerable questions."""
    un = [r for r in rows if not r["answerable"]]
    an = [r for r in rows if r["answerable"]]
    abst = [r for r in rows if r["abstained"]]
    tp = sum(r["abstained"] for r in un)
    return {
        "n_unanswerable": len(un), "n_answerable": len(an),
        "recall": tp / len(un) if un else math.nan,
        "precision": tp / len(abst) if abst else math.nan,
        "false_abstain_rate": sum(r["abstained"] for r in an) / len(an) if an else math.nan,
    }
