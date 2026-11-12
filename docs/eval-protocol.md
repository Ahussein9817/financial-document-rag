# Evaluation protocol (v2, in preparation)

Status: **schema, checks and tooling exist; v2 has candidate questions only, none validated.**
Gold needs the filings (`scripts/fetch_filings.py`, not yet run) and human validation. The historical
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

## Item lifecycle (enforced by `src/eval_schema.py`)

| Status | Meaning | Scored? |
|---|---|---|
| `candidate` | Question drafted (origin recorded: `claude_drafted` / `human_drafted`); no gold. May name where to look (`target`). | never |
| `machine_checked` | Gold evidence (passage id + verbatim snippet; numeric value for numeric items) was found in the corpus text by `apply_machine_check`; unanswerable items pass only if none of their `absence_terms` occur anywhere in the corpus. Proves the text exists where cited, **not** that the question or answer is right. | never |
| `human_validated` | A person confirmed question, gold answer and evidence against the filing and recorded `validated_by` / `validated_on`. | **only these** (`scorable`) |
| `rejected` | Dropped after review; kept for audit. | never |

No code path sets `human_validated`; that edit is made by hand in the JSON.

Current file: `data/eval/eval_questions_v2.json` holds 19 items across all six strata. Gold (answer, page-level evidence with verbatim snippets, numeric values) was **drafted by Claude from the downloaded filings** and then machine-checked with `scripts/machine_check_candidates.py`: all 19 pass (snippets found on the cited passages, numeric values present on them, absence terms absent from the whole corpus), so all 19 are `machine_checked`. **None is `human_validated`, so none is scorable.** Passage numbers are page-break units of the EDGAR HTML, not printed page numbers.

Known weaknesses a reviewer should look for: a machine check proves the text is on the page, not that it answers the question (for example BAC's capital table lists Standardized and Advanced ratios side by side); unanswerable items rest on string searches; several questions rely on one table row.

Workflow to finish: review each item against the filing, then set `status` to `human_validated` with `validation.validated_by` / `validated_on` by hand (or reject it). Add further items toward the ~60 target the same way.

## Metrics

- **Retrieval, passage level** (gold = `file#page`): Recall@K, hit@K, MRR, nDCG (`retrieval_metrics`).
- **Retrieval, file level** (`file_ids` collapses passages): reported separately. A file-level
  hit can be true while the passage-level answer page was missed.
- **Multi-document:** `multi_doc_coverage` (share of gold files covered, all-gold-hit).
- **Filter accuracy:** `filter_accuracy` (exact, per field, wrong-filter and missed-filter rates).
- **Answers** (`answer_checks.evaluate_answer`, no LLM) report separate sections, never one score:
  `numeric_match` (heuristic; unit error flagged separately from value error), `evidence_support`
  (exact: was a gold passage retrieved; heuristic: groundedness proxy = answer numbers appear in the
  retrieved text; citation validity/correctness from the `[S#]` citations the answer prompt now asks for), `abstention` (heuristic phrase list), and `semantic_correctness`
  (always `requires_human_review`). A numerically matching answer can still lack retrieved evidence,
  and a grounded answer can still be wrong; the sections are kept apart for that reason.

Limits: groundedness is a string check; derived numbers (sums, growth) count as unsupported;
the abstention phrase list is a heuristic; n about 60 gives wide intervals, so report them.

## Citations and prompt versions

The generation prompt (`PROMPT_VERSION = "v2-cited"`) labels each retrieved passage `[S1]`..`[Sk]` and asks the model to cite the passages it used. `extract_citations` maps labels to passage ids and lists any label outside the retrieved set as an invalid (invented) citation. The model's own citations are claims, checked by `citation_accuracy`; they are not evidence that the answer is correct. The historical 15-question results used the earlier uncited prompt (`PROMPT_TEMPLATE_V1`, kept in code) and are not comparable with v2 runs.
