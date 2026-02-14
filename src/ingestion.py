"""
Document Ingestion Pipeline
Loads PDFs, chunks them, and creates vector embeddings
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import json

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
        
        documents.extend(docs)
    
    print(f"✓ Loaded {len(documents)} pages from {len(list(pdf_path.glob('*.pdf')))} PDFs")
    return documents

def chunk_documents(documents, chunk_size=800, chunk_overlap=150):
    """Split documents into chunks for better retrieval"""
    print(f"\nChunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
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
    
    print(f"✓ Saved sample chunks")

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
    
    # Step 1: Load PDFs
    documents = load_pdfs()
    
    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    
    # Step 3: Save sample chunks for inspection
    save_sample_chunks(chunks)
    
    # Step 4: Create vector store
    vectorstore = create_vectorstore(chunks)
    
    print("\n" + "="*60)
    print("✓ INGESTION COMPLETE!")
    print("="*60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector store location: data/vectorstore")
    print("\nNext step: Test retrieval with test queries")

if __name__ == "__main__":
    main()
