"""
Test semantic search functionality
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings import LocalEmbeddingGenerator
from src.core.vector_store import VectorStore


def test_query(query_text: str, n_results: int = 3):
    """Test a search query"""
    print(f"\nQuery: '{query_text}'")
    print("-" * 60)
    
    # Generate query embedding
    generator = LocalEmbeddingGenerator(model_name="mini")
    query_embedding = generator.generate_embedding(query_text)
    
    # Search vector store
    store = VectorStore()
    results = store.query(query_embedding, n_results=n_results)
    
    # Display results
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity = 1 - distance
        print(f"\nResult {i+1} (similarity: {similarity:.3f}):")
        print(f"Source: {metadata['source']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(f"Text preview: {doc[:200]}...")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Semantic Search Test")
    print("="*60)
    
    # Check if vector store exists
    store = VectorStore()
    stats = store.get_stats()
    
    if stats['total_chunks'] == 0:
        print("\n❌ Error: Vector store is empty!")
        print("Please run: python scripts/build_index.py")
        sys.exit(1)
    
    print(f"\nVector Store Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Collection: {stats['collection_name']}")
    
    # Test queries
    test_queries = [
        "What is the gNB architecture?",
        "How does 5G NR differ from LTE?",
        "Explain the protocol stack",
        "What is NG-RAN?"
    ]
    
    for query in test_queries:
        test_query(query, n_results=3)
    
    print("\n" + "="*60)
    print("✅ Search tests complete!")
    print("="*60 + "\n")
