"""
RAG Pipeline for Financial Document Q&A
Queries SEC 10-K filings from JPMorgan, Bank of America, and Citigroup
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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


def build_chain(vectorstore, k=DEFAULT_K, filter_dict=None):
    """Build a RAG chain with optional metadata filtering"""
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    search_kwargs = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# --- Filtering Logic ---

def detect_filters(question):
    """Auto-detect company and year from question text"""
    question_lower = question.lower()

    ticker = None
    for name, t in COMPANY_MAP.items():
        if name in question_lower:
            ticker = t
            break

    year = None
    for y in [2022, 2023, 2024, 2025]:
        if str(y) in question:
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


# --- Main Query Functions ---

def ask_question(question, vectorstore, k=DEFAULT_K, verbose=True):
    """Ask a question without any metadata filtering"""
    chain, retriever = build_chain(vectorstore, k=k)

    answer = chain.invoke(question)
    docs = retriever.invoke(question)

    if verbose:
        _print_result(question, answer, docs)

    return _format_result(question, answer, docs, k)


def ask_with_filter(question, vectorstore, ticker=None, year=None, k=DEFAULT_K, verbose=True):
    """Ask a question with explicit metadata filtering"""
    filter_dict = None
    if ticker and year:
        filter_dict = {"$and": [{"ticker": ticker}, {"year": str(year)}]}
    elif ticker:
        filter_dict = {"ticker": ticker}
    elif year:
        filter_dict = {"year": str(year)}

    chain, retriever = build_chain(vectorstore, k=k, filter_dict=filter_dict)

    answer = chain.invoke(question)
    docs = retriever.invoke(question)

    if verbose:
        _print_result(question, answer, docs, ticker=ticker, year=year)

    return _format_result(question, answer, docs, k)


def smart_ask(question, vectorstore, k=DEFAULT_K, verbose=True):
    """Ask a question with automatic company/year detection"""
    ticker, year, filter_dict = detect_filters(question)

    if verbose and (ticker or year):
        filters = []
        if ticker:
            filters.append(f"ticker={ticker}")
        if year:
            filters.append(f"year={year}")
        print(f"Auto-detected filters: {', '.join(filters)}")

    chain, retriever = build_chain(vectorstore, k=k, filter_dict=filter_dict)

    answer = chain.invoke(question)
    docs = retriever.invoke(question)

    if verbose:
        _print_result(question, answer, docs, ticker=ticker, year=year)

    return _format_result(question, answer, docs, k)


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
        print(f"  [{i+1}] {doc.metadata.get('source_file')} "
              f"(Ticker: {doc.metadata.get('ticker')}, "
              f"Year: {doc.metadata.get('year')})")


def _format_result(question, answer, docs, k):
    return {
        "question": question,
        "answer": answer,
        "k": k,
        "sources": [
            {
                "file": doc.metadata.get("source_file"),
                "ticker": doc.metadata.get("ticker"),
                "year": doc.metadata.get("year"),
            }
            for doc in docs
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
