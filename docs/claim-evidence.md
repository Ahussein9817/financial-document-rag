# Claim-to-evidence table (financial-document-rag)

Evidence types: **README-reported** (author's original run, not reproduced), **regenerated** (recomputed from committed output files), **test** (offline unit tests), **CI** (GitHub Actions run), **not run**. Status labels: VERIFIED / PARTIAL / BLOCKED / PROPOSED / UNSUPPORTED. State as of branch `improve/portfolio-refinement`; update when the evidence changes.

| Claim | Status | Evidence type | Where | Caveat |
|---|---|---|---|---|
| RAG pipeline: PDF/HTML ingestion, chunking, Chroma, OpenAI embeddings, filtered retrieval, answer generation | PARTIAL | test (stubbed), README-reported | `src/ingestion.py`, `src/rag_pipeline.py`, `tests/` | No live ingest or query run with the current code; needs filings + OpenAI key |
| ~17,306 chunks from nine 10-K filings (JPM, BAC, C) | UNSUPPORTED (not reproducible) | README-reported | `data/processed/sample_chunks.json` (5 samples) | PDFs, vector store, accession numbers and fiscal years of the original files are not recorded |
| 15-question historical evaluation: 53.3% manually scored accuracy (6 correct, 4 partial, 5 incorrect) | VERIFIED (aggregation only) | regenerated | `scripts/reproduce_reported_metrics.py`, `outputs/reproduced_metrics.json` | Reproduces the aggregation of the author's manual scores; scoring itself is not re-validated; n=15; ground truth not re-checked against the filings |
| Factual accuracy 65.0% (README previously said 60%) | VERIFIED | regenerated | same | README corrected |
| Document-level hit@4 86.7% (13/15); strict file-level recall@4 83.3%; MRR 0.733 | VERIFIED (aggregation only) | regenerated, test | `src/eval_metrics.py`, `src/retrieval_metrics.py`, tests | File-level only; no passage-level labels exist for v1 |
| 57.7% accuracy when the expected file is retrieved; "generation is the bottleneck" | PARTIAL | regenerated (number); interpretation unsupported | `outputs/reproduced_metrics.json` | Right file does not mean right passage; n=15 cannot separate causes |
| Sub-2-second responses; ~$0.01/query; $3-5 project cost | UNSUPPORTED | README-reported | none | No timing/cost data committed; marked as unmeasured in README |
| README roadmap "expected accuracy gains" | UNSUPPORTED | hypotheses | README | Untested; labelled as hypotheses |
| Whole-word company/year filter detection (fixes "bac" in "feedback") | VERIFIED | test | `tests/test_rag_pipeline.py`; old vs new detection identical on the 15 v1 questions | Years 2022-2025 only |
| One retrieval per query; recorded sources are the generation context | VERIFIED | test | `tests/test_rag_pipeline.py::test_smart_ask_single_retrieval_filter_and_sources` | Stub vector store; not run live |
| Passage-level Recall@K/hit@K/MRR/nDCG, multi-doc coverage, filter accuracy | VERIFIED (code) | test | `src/retrieval_metrics.py`, `tests/test_retrieval_metrics.py` | No passage-level results exist (no gold) |
| Deterministic numeric / groundedness-proxy / citation / abstention checks | VERIFIED (code); heuristic by design | test | `src/answer_checks.py`, `tests/test_answer_checks.py` | Heuristics; semantic correctness always needs a human |
| EDGAR recovery with provenance manifest, fiscal year read from the filing | PARTIAL | test (mocked HTTP) | `scripts/fetch_filings.py`, `src/html_filing.py` | Never run against EDGAR; HTML "pages" are print-break approximations |
| v2 evaluation set | PROPOSED | schema + 19 candidates, no gold | `data/eval/eval_questions_v2.json` | 0 machine-checked, 0 human-validated; nothing is scorable |
| CI (ruff + pytest) | VERIFIED | CI | run 43312329331 (head 19f12db) | Later local commits not yet pushed/run |
| Answers cite passages (`[S#]`) and citations are checked against retrieved passages | PARTIAL | test (stubbed) | `src/rag_pipeline.py`, `tests/test_rag_pipeline.py` | Never run with a real model; models may cite wrongly or not at all |
| Live answers on a validated set; answer accuracy of any v2 run | BLOCKED | not run | none | Needs filings, human validation, OpenAI key and approved spend |
