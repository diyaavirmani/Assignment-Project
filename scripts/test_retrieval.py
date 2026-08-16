"""
Test retrieval with different parameters
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.retriever import DocumentRetriever


def test_top_k(query: str):
    """Test different top_k values"""
    print(f"\n{'='*60}")
    print(f"Testing top_k values")
    print(f"Query: '{query}'")
    print("=" * 60)
    
    for k in [3, 5, 10]:
        retriever = DocumentRetriever(top_k=k)
        docs = retriever.retrieve(query)
        
        print(f"\ntop_k={k}:")
        print(f"  Retrieved {len(docs)} documents")
        if docs:
            print(f"  Similarity range: {docs[-1]['similarity']:.3f} - {docs[0]['similarity']:.3f}")
            print(f"  First result: {docs[0]['text'][:80]}...")


def test_source_filter(query: str):
    """Test filtering by source"""
    print(f"\n{'='*60}")
    print(f"Testing source filtering")
    print(f"Query: '{query}'")
    print("=" * 60)
    
    retriever = DocumentRetriever(top_k=5)
    
    # Without filter
    docs_all = retriever.retrieve(query)
    print(f"\nWithout filter:")
    print(f"  Retrieved: {len(docs_all)} documents")
    if docs_all:
        sources = set(d['source'] for d in docs_all)
        print(f"  Sources: {', '.join(sources)}")
    
    # With filter (adjust based on your files)
    # Find a common source from results
    if docs_all:
        common_source = docs_all[0]['source'].split('-')[0]  # e.g., "38300" from "38300-h30.docx"
        docs_filtered = retriever.retrieve(query, source_filter=common_source)
        print(f"\nWith filter ('{common_source}'):")
        print(f"  Retrieved: {len(docs_filtered)} documents")
        if docs_filtered:
            sources_filtered = set(d['source'] for d in docs_filtered)
            print(f"  Sources: {', '.join(sources_filtered)}")


def test_query_variations(base_query: str):
    """Test query reformulations"""
    print(f"\n{'='*60}")
    print(f"Testing query variations")
    print("=" * 60)
    
    queries = [
        base_query,
        f"Explain {base_query}",
        f"What is {base_query}?",
        f"Technical details of {base_query}"
    ]
    
    retriever = DocumentRetriever(top_k=3)
    
    for query in queries:
        docs = retriever.retrieve(query)
        if docs:
            print(f"\nQuery: '{query}'")
            print(f"  Best match similarity: {docs[0]['similarity']:.3f}")
            print(f"  Preview: {docs[0]['text'][:80]}...")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Retrieval Parameter Testing")
    print("="*60)
    
    # Check if vector store has data
    from src.core.vector_store import VectorStore
    store = VectorStore()
    stats = store.get_stats()
    
    if stats['total_chunks'] == 0:
        print("\n❌ Error: Vector store is empty!")
        print("Please run: python scripts/build_index.py\n")
        sys.exit(1)
    
    print(f"\nVector store: {stats['total_chunks']} chunks loaded\n")
    
    # Run tests
    test_top_k("What is gNB architecture?")
    test_source_filter("5G protocol stack")
    test_query_variations("NG-RAN")
    
    print("\n" + "="*60)
    print("✅ Retrieval tests complete!")
    print("="*60 + "\n")
