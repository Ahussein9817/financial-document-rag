# Financial Document RAG System

Retrieval-Augmented Generation (RAG) system for querying SEC 10-K filings from major US banks.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-green.svg)](https://github.com/langchain-ai/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that enables natural language querying of financial documents. The system processes 9 SEC 10-K filings from JPMorgan Chase, Bank of America, and Citigroup (fiscal years 2022-2025), creating a semantic search engine powered by large language models.

**Key Features:**
- Processes 9 financial PDFs (~17,300 document chunks)
- Semantic search using OpenAI embeddings
- Natural language Q&A with GPT-3.5-turbo
- Source citation for all answers
- Metadata filtering by company and year
- Sub-2-second response time (author's claim, not measured here)

## Performance

**Evaluation Methodology:**
- Test set: 15 questions with manually written ground truth (historical; not re-validated against the filings, see `docs/eval-protocol.md`)
- Categories: 10 factual, 3 comparative, 2 trend-based questions
- Metrics: Answer accuracy + Retrieval recall@4

**Results:**

| Metric | Score | Interpretation |
|--------|-------|----------------
| **Document-level hit@4** (README label: "Recall@4") | 86.7% | At least one chunk from the expected file in the top 4 (13/15). Strict file-level recall@4 is 83.3% when several files are expected; MRR 0.733. File-level only: no passage-level labels exist for this set |
| **Factual Recall** | 90.0% | Excellent at finding factual data |
| **Comparative Recall** | 66.7% | Moderate for multi-company queries |
| **Trend Recall** | 100.0% | Perfect for temporal queries |

**Diagnostic Analysis:**
- **When retrieval succeeds (86.7% of cases):**
- **When retrieval fails (13.3% of cases):** 25.0% answer accuracy
- **Primary bottleneck:** Generation quality (GPT-3.5), not retrieval
- **Key insight:** System finds correct documents but struggles with answer synthesis

**Performance by Question Type:**

| Category | Accuracy | Retrieval Recall | Best Use Case |
|----------|----------|------------------|---------------|
| **Factual** | 65.0% | 90.0% | Single-company lookups |
| **Comparative** | 33.3% | 66.7% | Cross-company synthesis |
| **Trend** | 25.0% | 100.0% | Multi-year analysis |

**Evaluation Limitations:**
- Small sample size (n=15) provides directional insights with ±25% confidence interval
- Subjective partial scoring (0.5 points) for near-correct answers
- Does not separate unit correctness from value correctness
- For production: recommend 80-150 question test set with automated metrics

## Strengths

- **Excellent retrieval:** 86.7% recall - finds correct documents reliably  
- **Fast queries:** Sub-2-second response time (UNSUPPORTED: no timing was committed; the old pipeline also ran retrieval twice per question)  
- **Source attribution:** Full transparency on document sources  
- **Cost-effective:** $3-5 total project cost, ~$0.01/query (author's estimate, not measured in this repo)

## Limitations

- **Generation bottleneck:** 57.7% accuracy even with correct documents  
  (57.7% is reproduced from the committed scores, but "generation is the bottleneck" is an interpretation: retrieval is scored at file level, so the right file does not mean the right passage was retrieved, and n=15 is too small to separate the two causes)
- **Cross-document synthesis:** 33% accuracy on comparative questions  
- **Multi-year trends:** 25% accuracy on temporal analysis  
- **Small evaluation set:** Results not statistically robust (n=15)

## Improvement Roadmap

> **All "expected gain", impact and cost figures in this roadmap are the author's hypotheses. None has been tested.** The measured results are in the Performance section above. The evaluation rebuild is described in `docs/eval-protocol.md`.

**Based on retrieval recall analysis, generation quality is the primary bottleneck.**

### Phase 1: Quick Wins (2-3 hours, Expected: 53% → 70-75%)

**1. Upgrade to GPT-4** (HIGHEST IMPACT)
```python
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
```
- **Impact:** +15-20% accuracy (addresses generation bottleneck)
- **Cost:** $10-20 additional for evaluation
- **Rationale:** 86.7% retrieval recall but only 57.7% accuracy when retrieval succeeds
- **Target:** Would fix Q6, Q7, Q11-Q15 (generation failures with good chunks)

**2. Add Re-ranking**
```python
# Retrieve k=6 chunks, re-rank to best 4
reranked_docs = rerank_with_llm(query, docs)
```
- **Impact:** +5-10% accuracy
- **Cost:** Minimal (~$1 for evaluation)
- **Target:** Would improve Q12, Q14 (better chunk prioritization)

**3. Adaptive k Values**
```python
if question_type == "comparative": k = 6
elif question_type == "trend": k = 5
```
- **Impact:** +3-5% accuracy
- **Cost:** None
- **Target:** Would improve Q11, Q13, Q15 (more context for synthesis)

**Expected Result: 53% → 70-75% accuracy**

### Phase 2: Address Retrieval Gaps (4-6 hours, Expected: +5-10%)

**4. Hybrid Search (BM25 + Vector)**
- Fix Q5 (BAC CET1 ratio) - retrieval failure
- Catch exact keyword matches
- **Impact:** +5% on factual questions

**5. Better Chunking Strategy**
- Semantic chunking (topic-based)
- Larger chunks (1000 chars)
- **Requires:** Complete re-ingestion

**Expected Result: 70-75% → 80-85% accuracy**

### Phase 3: Advanced (Research Phase)

**6. Fine-tuned Embeddings**
- Train on financial documents
- Better domain understanding
- **Impact:** +5-10% overall

**7. Query Decomposition**
- Break complex questions into sub-queries
- Better for comparative/trend questions
- **Impact:** +10% on complex questions

**8. GraphRAG / Agentic RAG**
- Knowledge graph representation
- Multi-step reasoning
- **Impact:** +10-15% on comparative questions

**Expected Result: 80-85% → 90%+ accuracy**

## Cost-Benefit Analysis

| Improvement | Time | Cost | Accuracy Gain | ROI | Priority |
|-------------|------|------|---------------|-----|----------|
| GPT-4 Upgrade | 5 min | $10-20 | +15-20% | High | **HIGHEST** |
| Re-ranking | 1-2 hrs | ~$1 | +5-10% | High | **HIGH** |
| Adaptive k | 30 min | $0 | +3-5% | High | **HIGH** |
| Hybrid Search | 4-6 hrs | $0 | +5% | Medium | Medium |
| Better Chunking | 3-4 hrs | $2 | +5-10% | Medium | Medium |
| Fine-tune Embeddings | 2-3 days | $50+ | +5-10% | Low | Low |
| GraphRAG | 1-2 weeks | $100+ | +10-15% | Low | Research |

**Recommended Path:** Phase 1 (GPT-4 + Re-ranking + Adaptive k) → Validate → Phase 2 if needed

## Architecture
```
User Question
    ↓
[Embedding Model] → Query Vector
    ↓
[ChromaDB Vector Store] → Retrieve Top 4 Chunks
    ↓
[GPT-3.5-turbo] → Generate Answer
    ↓
Answer + Sources
```

### Technology Stack

- **LLM:** OpenAI GPT-3.5-turbo (temperature=0)
- **Embeddings:** OpenAI text-embedding-ada-002
- **Vector Store:** ChromaDB (local, persistent)
- **Framework:** LangChain 1.2+
- **Language:** Python 3.10+

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Chunk Size: 800 chars** | Balances context vs precision; captures full paragraphs |
| **Overlap: 150 chars** | Prevents context loss at chunk boundaries |
| **k=4 retrieval** | Optimal balance; tested k=3,4,5 |
| **ChromaDB** | Free, local, built for LLM apps; easy deployment |
| **GPT-3.5** | Cost-effective ($3-5 total project cost) |

## Project Structure
```
financial-document-rag/
├── data/
│   ├── raw_pdfs/          # 9 original SEC 10-K PDFs
│   ├── processed/         # Processed data, test results
│   └── vectorstore/       # ChromaDB database (17K+ chunks)
├── src/
│   ├── ingestion.py       # PDF loading & chunking
│   └── rag_pipeline.py    # Q&A system (if converted from notebook)
├── notebooks/
│   ├── day2_rag_pipeline.ipynb    # RAG development
│   └── evaluation.ipynb      # Evaluation & analysis
├── outputs/
│   ├── evaluation_accuracy.png
│   ├── question_performance.png
│   └── evaluation_report.md
├── requirements.txt
├── .env                   # API keys (not in repo)
├── .gitignore
└── README.md
```

## Installation

### Prerequisites
- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Setup

1. **Clone repository**
```bash
git clone https://github.com/Ahussein9817/financial-document-rag.git
cd financial-document-rag
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API key**
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

5. **Run ingestion** (if vector store not included)
```bash
python src/ingestion.py
```

## Usage

### Interactive Q&A (Notebook)
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load system
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory='data/vectorstore', embedding_function=embeddings)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Ask question
question = "What was JPMorgan's revenue in 2024?"
# ... (chain setup)
answer = rag_chain.invoke(question)
```

### Example Queries
```python
# Factual lookup
"What was Bank of America's net income in 2024?"

# Comparative
"Which bank had higher revenue: JPMorgan or Citigroup?"

# Trend analysis
"How has JPMorgan's revenue changed from 2023 to 2024?"

# Filtered search
ask_with_filter("What was the revenue?", ticker='JPM', year=2024)
```

## Evaluation

### Test Set
- 15 questions with ground truth
- 10 factual, 3 comparative, 2 trend-based
- Manually verified against source PDFs

### Strengths
- Accurate factual lookups (single company/year)  
- Correct source attribution  
- Fast response time (<2 seconds)

### Limitations
- Cross-document comparisons challenging  
- Multi-year trend analysis incomplete  
- Numerical precision issues (unit conversions)

See [evaluation_report.md](outputs/evaluation_report.md) for details.

## Cost Analysis

**Total project cost:** ~$3-5 USD

- Embeddings (17,300 chunks): ~$2.50
- LLM queries (~50 questions): ~$0.50
- Evaluation (15 questions): ~$0.20

**Per-query cost:** ~$0.01

## Future Improvements

### Short-term
- Increase k for comparative questions
- Add numerical normalization
- Better error handling

### Long-term
- Query routing (factual vs comparative)
- Hybrid retrieval (dense + sparse)
- Fine-tuned embeddings
- Multi-document synthesis

## Known Issues

1. **Comparative questions:** System struggles to synthesize across banks
2. **Numerical formats:** $M vs $B vs $T inconsistencies
3. **Multi-year trends:** Limited context span across years

## License

MIT License - see [LICENSE](LICENSE) file

## Author

**Amina Hussein**
- GitHub: [@Ahussein9817](https://github.com/Ahussein9817)
- Project: Financial Document RAG System
- Date: February 2026

## Acknowledgments

- SEC Edgar database for 10-K filings
- LangChain community
- OpenAI API

---

**Built with:** Python • LangChain • ChromaDB • OpenAI GPT-3.5


## Reproducibility and development (added 2026-11-11, branch `improve/portfolio-refinement`)

The historical results above are unchanged. What a fresh clone can and cannot do:

| Step | Command | Needs | Status |
|---|---|---|---|
| Unit tests + lint | `pip install -r requirements.txt -r requirements-dev.txt && pytest -q && ruff check .` | nothing external | 24 tests pass locally (Python 3.12) |
| Recompute README aggregates | `python scripts/reproduce_reported_metrics.py` | committed CSVs only | Reproduces 53.3% overall, 65.0% factual (README prose says 60%), 86.7% document-level recall@4. This re-aggregates the author's manual scores; it does not re-score answers or re-run retrieval. |
| Build vector store | `python scripts/ingest.py` | `OPENAI_API_KEY`, the 9 PDFs (`docs/data-sources.md`) | **Not run** in this pass (paid API, PDFs not in repo) |
| Run questions live | `python scripts/run_eval.py` | key + vector store | **Not run**; the runner logic is unit tested with a stub. It records latency, which is the first latency measurement this repo would have (the README's "sub-2-second" and "$3-5" remain unmeasured) |

`data/eval/eval_questions_v1.json` is the original 15 questions copied verbatim from the notebook; their ground truth has not been re-validated against the filings. `src/test_retrieval.py` is a manual smoke script, not part of the test suite.

Bug fixed: company/year detection used substring matching, so "feedback"/"backlog" were read as Bank of America ("bac") and "cities" as Citigroup ("citi"). It now matches whole words (regression tests in `tests/test_rag_pipeline.py`). Historical results were produced with the old matcher; I re-ran old vs. new detection on the 15 evaluation questions and the detected filters are identical for all 15, so the reported results are not affected by this bug.

### Retrieval metrics (added 2026-11-11)

`src/retrieval_metrics.py` implements Recall@K, hit@K, MRR and nDCG@K for binary relevance, unit tested on hand-computed cases (`tests/test_retrieval_metrics.py`). They apply to passage-level gold labels, which do not exist yet: the historical 15 questions only record an expected *file*. Applied at file level to the committed retrieved-file lists (4 chunks per question, duplicates kept):

| Measure (file level, 15 questions) | Value |
|---|---|
| hit@4: any expected file retrieved (the README's "recall@4") | 86.7% (13/15) |
| recall@4: fraction of all expected files retrieved (stricter for multi-company questions) | 83.3% |
| MRR over the 4 retrieved chunks | 0.733 |

These are re-computations of the same historical retrieval run, not new experiments. Passage-level metrics and a comparison of retrieval configurations still need a larger gold set with page or passage labels (not yet built).


## Evaluation tooling added in the refinement branch

- `src/rag_pipeline.py`: one retrieval per query (the prompt and the reported sources are the same chunks); sources carry `file`, 1-based `page`, `passage_id` (`FILE.pdf#pN`), distance score and text; every query emits a JSON log line (logger `rag`) with filters, passages, scores, latency and any error; a filter that matches no chunks logs a warning.
- `src/retrieval_metrics.py`: Recall@K, hit@K, MRR, nDCG (passage level), file-level collapse, multi-document coverage, filter accuracy.
- `src/answer_checks.py`: deterministic numeric, groundedness, citation and abstention checks (no LLM).
- `src/eval_schema.py`, `docs/eval-protocol.md`, `data/eval/eval_questions_v2.json`: three-state lifecycle (candidate / machine_checked / human_validated; only human_validated is scorable) and 19 candidate questions with no gold. **Nothing in v2 is validated.**
- Generation prompt `v2-cited` labels passages `[S1]..[Sk]` and asks for citations; results include `cited_passages` and `invalid_citations`. The historical results used an uncited prompt and are not comparable.
- `src/html_filing.py`: EDGAR HTML page splitting (falls back to fixed-size text blocks when the filing has too few page breaks; `locator_kind` records which) and fiscal-year/period extraction from the filing itself; chunk ids and provenance are in `ingestion.py`.
- `scripts/fetch_filings.py`: EDGAR download with provenance manifest (accession, period of report, fiscal year read from the filing). **Not yet run; corpus not downloaded.**
- `src/answer_checks.py` results are **heuristic** (numeric match, groundedness proxy, abstention phrases) and are reported separately from semantic correctness, which always needs a human.
- **Not done:** no live ingest or evaluation has been run with these changes (needs the filings and an OpenAI key). The 17,306-chunk corpus and the historical results come from the original run and are not reproduced here.
