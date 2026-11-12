"""
RAG Pipeline for Financial Document Q&A
Queries SEC 10-K filings from JPMorgan, Bank of America, and Citigroup
"""

import json
import logging
import re
import time

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
VECTORSTORE_DIR = "data/vectorstore"
LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_K = 4

COMPANY_MAP = {
    "jpmorgan": "JPM",
    "jpm": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "bofa": "BAC",
    "bac": "BAC",
    "citigroup": "C",
    "citi": "C",
}

PROMPT_TEMPLATE = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""


# --- Core Setup ---

def load_vectorstore(persist_directory=VECTORSTORE_DIR):
    """Load ChromaDB vector store"""
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join([doc.page_content for doc in docs])


# --- Filtering Logic ---

def detect_filters(question):
    """Auto-detect company and year from question text"""
    question_lower = question.lower()

    ticker = None
    for name, t in COMPANY_MAP.items():
        # whole-word match: plain substring matching made "bac" fire inside
        # "feedback"/"backlog" and "citi" inside "cities"
        if re.search(rf"\b{re.escape(name)}\b", question_lower):
            ticker = t
            break

    year = None
    for y in [2022, 2023, 2024, 2025]:
        if re.search(rf"\b{y}\b", question):
            year = y
            break

    filter_dict = None
    if ticker and year:
        filter_dict = {"$and": [{"ticker": ticker}, {"year": str(year)}]}
    elif ticker:
        filter_dict = {"ticker": ticker}
    elif year:
        filter_dict = {"year": str(year)}

    return ticker, year, filter_dict


# --- Retrieval and generation (one retrieval per query) ---

logger = logging.getLogger("rag")


def passage_id(metadata):
    """Stable page-level passage id, e.g. 'JPM_10K_2024.pdf#p87' (1-based page).

    Returns None when the chunk has no file or page metadata (such chunks can never
    match a gold passage and are counted by the diagnostics).
    """
    f, page = metadata.get("source_file"), metadata.get("page")
    if f is None or page is None:
        return None
    return f"{f}#p{int(page) + 1}"


def retrieve(question, vectorstore, k=DEFAULT_K, filter_dict=None):
    """Single similarity search; returns [(Document, distance)]. The same documents
    are used for the prompt and for the reported sources."""
    kwargs = {"k": k}
    if filter_dict:
        kwargs["filter"] = filter_dict
    return vectorstore.similarity_search_with_score(question, **kwargs)


def generate(question, docs):
    """Answer from the given documents (the only context the model sees)."""
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | llm | StrOutputParser()
    return chain.invoke({"context": format_docs(docs), "question": question})


def _run(question, vectorstore, k, ticker, year, filter_dict, mode, verbose):
    t0 = time.perf_counter()
    err = None
    scored, answer = [], None
    try:
        scored = retrieve(question, vectorstore, k=k, filter_dict=filter_dict)
        docs = [d for d, _ in scored]
        if filter_dict and not docs:
            logger.warning("filter matched no chunks: %s", filter_dict)
        answer = generate(question, docs)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        record = {
            "event": "rag_query", "mode": mode, "question": question, "k": k,
            "filter": filter_dict, "n_retrieved": len(scored),
            "passages": [passage_id(d.metadata) for d, _ in scored],
            "scores": [round(float(s), 4) for _, s in scored],
            "latency_s": round(time.perf_counter() - t0, 3), "error": err,
        }
        logger.info(json.dumps(record))
    if verbose:
        _print_result(question, answer, [d for d, _ in scored], ticker=ticker, year=year)
    return _format_result(question, answer, scored, k, filter_dict)


# --- Main Query Functions ---

def ask_question(question, vectorstore, k=DEFAULT_K, verbose=True):
    """Ask a question without any metadata filtering"""
    return _run(question, vectorstore, k, None, None, None, "no_filter", verbose)


def ask_with_filter(question, vectorstore, ticker=None, year=None, k=DEFAULT_K, verbose=True):
    """Ask a question with explicit metadata filtering"""
    filter_dict = None
    if ticker and year:
        filter_dict = {"$and": [{"ticker": ticker}, {"year": str(year)}]}
    elif ticker:
        filter_dict = {"ticker": ticker}
    elif year:
        filter_dict = {"year": str(year)}
    return _run(question, vectorstore, k, ticker, year, filter_dict, "explicit_filter", verbose)


def smart_ask(question, vectorstore, k=DEFAULT_K, verbose=True):
    """Ask a question with automatic company/year detection"""
    ticker, year, filter_dict = detect_filters(question)
    if verbose and (ticker or year):
        filters = [f"ticker={ticker}"] if ticker else []
        if year:
            filters.append(f"year={year}")
        print(f"Auto-detected filters: {', '.join(filters)}")
    return _run(question, vectorstore, k, ticker, year, filter_dict, "smart_filter", verbose)


# --- Helpers ---

def _print_result(question, answer, docs, ticker=None, year=None):
    filter_str = ""
    if ticker or year:
        parts = []
        if ticker:
            parts.append(f"ticker={ticker}")
        if year:
            parts.append(f"year={year}")
        filter_str = f" [Filter: {', '.join(parts)}]"

    print(f"\n{'='*60}")
    print(f"Q: {question}{filter_str}")
    print(f"{'='*60}")
    print(f"\nA: {answer}\n")
    print("Sources:")
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] {passage_id(doc.metadata)} "
              f"(Ticker: {doc.metadata.get('ticker')}, "
              f"Year: {doc.metadata.get('year')})")


def _format_result(question, answer, scored, k, filter_dict=None):
    return {
        "question": question,
        "answer": answer,
        "k": k,
        "filter": filter_dict,
        "sources": [
            {
                "file": doc.metadata.get("source_file"),
                "ticker": doc.metadata.get("ticker"),
                "year": doc.metadata.get("year"),
                "page": (int(doc.metadata["page"]) + 1) if doc.metadata.get("page") is not None else None,
                "passage_id": passage_id(doc.metadata),
                "score": round(float(score), 4),
                "text": doc.page_content,
            }
            for doc, score in scored
        ],
    }


# --- Entry Point ---

if __name__ == "__main__":
    print("Loading vector store...")
    vectorstore = load_vectorstore()
    print(f"Vector store loaded: {vectorstore._collection.count()} chunks\n")

    # Example queries
    smart_ask("What was JPMorgan's total revenue in 2024?", vectorstore)
    smart_ask("Compare JPMorgan and Bank of America's revenue growth", vectorstore)
    smart_ask("What are Citigroup's main revenue sources?", vectorstore)
