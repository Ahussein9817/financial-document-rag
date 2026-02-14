"""
Quick test to verify vector store retrieval works
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_retrieval():
    print("Testing vector store retrieval...")
    print("="*60)
    
    # Load vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory='data/vectorstore',
        embedding_function=embeddings
    )
    
    # Test query
    query = "What was JPMorgan's revenue in 2024?"
    print(f"\nQuery: {query}\n")
    
    results = vectorstore.similarity_search(query, k=3)
    
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Ticker: {doc.metadata.get('ticker', 'Unknown')}")
        print(f"Year: {doc.metadata.get('year', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:300]}...")
        print("-"*60)
    
    print("\n✓ Retrieval test complete!")

if __name__ == "__main__":
    test_retrieval()
