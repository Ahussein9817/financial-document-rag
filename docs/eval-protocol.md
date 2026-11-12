# Evaluation protocol (v2, in preparation)

Status: **schema, checks and tooling exist; no v2 questions exist yet.** Questions need the
filings (`scripts/fetch_filings.py`, not yet run) and human validation. The historical
15-question set (`data/eval/eval_questions_v1.json`) is kept unchanged and is not comparable
to v2: different corpus, unvalidated gold, file-level labels only.

## Dataset (target about 60 validated questions)

| Stratum | Target n | Gold |
|---|---|---|
| single_numeric | 18 | value, unit, scale, period + page evidence |
| single_text | 10 | page evidence |
| multi_same_company | 8 | evidence in >=2 filings of one company |
| multi_cross_company | 8 | evidence in >=2 companies |
| filter_sensitive | 8 | evidence + expected (ticker, year) filter |
| unanswerable | 8 | no evidence; `validation.absence_check` states how absence was confirmed |

Item format and rules are enforced by `src/eval_schema.py` (`tests/test_eval_schema.py`).

## Validation (nothing is accepted because an LLM said so)

1. Candidate questions may be LLM-proposed; they are stored with `status: "proposed"`.
2. Gold evidence is located by text search in the filing and stored as
   `file#page` plus a short verbatim snippet.
3. A human opens that page and confirms the question, the gold answer and the evidence;
   only then `status` becomes `validated` with `validated_by` / `validated_on`.
4. Unanswerable items need a documented search showing the answer is not in the corpus.
5. Only validated items are scored (`eval_schema.scorable`); the report lists how many were excluded.

## Metrics

- **Retrieval, passage level** (gold = `file#page`): Recall@K, hit@K, MRR, nDCG (`retrieval_metrics`).
- **Retrieval, file level** (`file_ids` collapses passages): reported separately. A file-level
  hit can be true while the passage-level answer page was missed.
- **Multi-document:** `multi_doc_coverage` (share of gold files covered, all-gold-hit).
- **Filter accuracy:** `filter_accuracy` (exact, per field, wrong-filter and missed-filter rates).
- **Answers** (`answer_checks`, no LLM): numeric match with unit-error flag, numeric groundedness
  against retrieved text, citation validity/correctness, abstention precision/recall/false-abstain.
  Free-text correctness stays a human judgement on a subsample.

Limits: groundedness is a string check; derived numbers (sums, growth) count as unsupported;
the abstention phrase list is a heuristic; n about 60 gives wide intervals, so report them.
