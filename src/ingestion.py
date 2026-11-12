"""
Document Ingestion Pipeline
Loads PDFs, chunks them, and creates vector embeddings
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import hashlib
import json

from html_filing import locate

# Load environment variables
load_dotenv()

def load_pdfs(pdf_directory="data/raw_pdfs"):
    """Load all PDFs from specified directory"""
    pdf_path = Path(pdf_directory)
    documents = []
    
    print(f"Loading PDFs from {pdf_directory}...")
    
    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        print(f"  Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        
        # Extract metadata from filename (e.g., JPM_10K_2025.pdf)
        filename = pdf_file.stem  # Gets filename without .pdf
        parts = filename.split('_')
        if len(parts) >= 3:
            ticker = parts[0]
            year = parts[2]
            
            # Add metadata to each page
            for doc in docs:
                doc.metadata['ticker'] = ticker
                doc.metadata['year'] = year
                doc.metadata['source_file'] = pdf_file.name
                # historical corpus only: the year comes from the filename, not the filing
                doc.metadata['fiscal_year_source'] = 'filename'
        
        documents.extend(docs)
    
    print(f"✓ Loaded {len(documents)} pages from {len(list(pdf_path.glob('*.pdf')))} PDFs")
    return documents

def load_filings(directory="data/raw_filings"):
    """Load filings listed in `<directory>/manifest.json` (written by scripts/fetch_filings.py).

    Ticker, fiscal year, period of report and accession come from the manifest, which in
    turn read them from the filing itself. Entries whose fiscal year could not be resolved
    are refused (never guessed). PDFs are loaded page by page; HTML filings are split on
    their print page-break markers, or into fixed-size text blocks when there are too few
    markers (see html_filing.locate; `locator_kind` records which).
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    documents = []
    for e in manifest["filings"]:
        if not e.get("fiscal_year"):
            raise ValueError(f"{e.get('file')}: fiscal year not resolved from the filing; refusing to ingest")
        path = directory / e["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != e["sha256"]:
            raise ValueError(f"{e['file']}: SHA-256 does not match manifest")
        base = {
            "ticker": e["ticker"], "year": str(e["fiscal_year"]), "source_file": e["file"],
            "accession": e["accession"], "period_of_report": e.get("period_of_report") or "",
            "fiscal_year_source": e.get("fiscal_year_source") or "",
        }
        if path.suffix.lower() == ".pdf":
            for d in PyPDFLoader(str(path)).load():
                d.metadata.update(base)
                documents.append(d)
        else:
            html = path.read_text(encoding="utf-8", errors="ignore")
            units, kind = locate(html)
            for i, text in enumerate(units):
                documents.append(Document(page_content=text,
                                          metadata={**base, "page": i, "locator_kind": kind}))
    return documents


def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    """Split documents into chunks for better retrieval"""
    print(f"\nChunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    assign_chunk_ids(chunks)
    print(f"✓ Created {len(chunks)} chunks")
    
    return chunks

def assign_chunk_ids(chunks):
    """Stable provenance ids, in place.

    passage_id = '<file>#p<1-based page>' (what gold evidence points at);
    chunk_id = '<passage_id>#c<index within that page>';
    content_sha1 = hash of the chunk text (detects re-chunking drift between runs).
    Needs source_file and page metadata; chunks lacking them get no ids.
    """
    counters = {}
    for c in chunks:
        f, page = c.metadata.get("source_file"), c.metadata.get("page")
        if f is None or page is None:
            continue
        pid = f"{f}#p{int(page) + 1}"
        n = counters.get(pid, 0)
        counters[pid] = n + 1
        c.metadata["passage_id"] = pid
        c.metadata["chunk_id"] = f"{pid}#c{n}"
        c.metadata["content_sha1"] = hashlib.sha1(c.page_content.encode("utf-8")).hexdigest()[:16]
    return chunks


def save_sample_chunks(chunks, output_file="data/processed/sample_chunks.json", n=5):
    """Save sample chunks for inspection"""
    print(f"\nSaving {n} sample chunks to {output_file}...")
    
    samples = []
    for i, chunk in enumerate(chunks[:n]):
        samples.append({
            'chunk_id': i,
            'content': chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content,
            'metadata': chunk.metadata,
            'length': len(chunk.page_content)
        })
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print("✓ Saved sample chunks")

def create_vectorstore(chunks, persist_directory="data/vectorstore"):
    """Create ChromaDB vector store with embeddings"""
    print(f"\nCreating vector store in {persist_directory}...")
    print("  (This may take 2-3 minutes - creating embeddings for all chunks)")
    
    embeddings = OpenAIEmbeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✓ Vector store created with {len(chunks)} chunks")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("="*60)
    print("FINANCIAL DOCUMENT INGESTION PIPELINE")
    print("="*60)
    
    # Step 1: Load filings (manifest, fiscal year read from the filing) or legacy PDFs
    if (Path("data/raw_filings") / "manifest.json").exists():
        documents = load_filings()
    else:
        print("WARNING: no data/raw_filings/manifest.json; using legacy PDFs whose fiscal year "
              "comes from the filename (metadata fiscal_year_source='filename').")
        documents = load_pdfs()
    
    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    
    # Step 3: Save sample chunks for inspection
    save_sample_chunks(chunks)
    
    # Step 4: Create vector store
    create_vectorstore(chunks)
    
    print("\n" + "="*60)
    print("✓ INGESTION COMPLETE!")
    print("="*60)
    print(f"Total chunks: {len(chunks)}")
    print("Vector store location: data/vectorstore")
    print("\nNext step: Test retrieval with test queries")

if __name__ == "__main__":
    main()
