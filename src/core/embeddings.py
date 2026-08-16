"""
Local embedding generation via sentence-transformers (zero cost, no API key).

Embeddings are dense vector representations of text that capture semantic
meaning. This module downloads and caches models from HuggingFace on first
use, then runs entirely locally.

Supported model shortcuts
-------------------------
``mini``      all-MiniLM-L6-v2        Fast, 384-dim, ~80 MB
``mpnet``     all-mpnet-base-v2        Better quality, 768-dim, ~420 MB
``bge-small`` BAAI/bge-small-en-v1.5  Best for technical/domain text, 384-dim, ~130 MB (default)
``bge-base``  BAAI/bge-base-en-v1.5   Higher accuracy than bge-small, 768-dim, ~440 MB

The embedding model used at **index time** must match the one used at
**query time**. The project defaults to ``bge-small`` throughout.

Example:
    gen = LocalEmbeddingGenerator(model_name="bge-small")
    vec = gen.generate_embedding("What is the F1 interface?")  # List[float]
    chunks = gen.embed_chunks(chunk_list)  # adds 'embedding' key to each chunk
"""
import logging
from typing import List, Optional
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed")
    print("Install with: pip install sentence-transformers")
    SentenceTransformer = None

logger = logging.getLogger(__name__)



class LocalEmbeddingGenerator:
    """Generate embeddings using local models (FREE, no API)"""
    def generate(self, text: str) -> List[float]:
        """Alias for generate_embedding to match original script"""
        return self.generate_embedding(text)
    
    # Model options (all free!)
    MODELS = {
        "mini": "all-MiniLM-L6-v2",           # Fast, small (80MB)
        "mpnet": "all-mpnet-base-v2",         # Better quality (420MB)
        "bge-small": "BAAI/bge-small-en-v1.5", # Great for tech docs (130MB)
        "bge-base": "BAAI/bge-base-en-v1.5",   # Even better (440MB)
    }
    
    def __init__(
        self,
        model_name: str = "mini",
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: Shortcut key from MODELS dict, or a full HuggingFace
                model ID (e.g. "BAAI/bge-small-en-v1.5").
            batch_size: Number of texts encoded per forward pass. Larger
                values are faster but use more memory.
            device: Force a specific device: ``"cuda"``, ``"cpu"``, or
                ``None`` for automatic selection (GPU if available).

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if not SentenceTransformer:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Get actual model name
        actual_model = self.MODELS.get(model_name, model_name)
        
        logger.info(f"Loading model: {actual_model}")
        logger.info("First time will download model (1-5 minutes depending on model)")
        
        # Load model (downloads first time, then cached locally)
        self.model = SentenceTransformer(actual_model, device=device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded: {actual_model}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Device: {self.model.device}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch
        
        More efficient than calling generate_embedding multiple times
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings.tolist()
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Generate embeddings for document chunks
        
        Processes in batches for efficiency
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            Chunks with added 'embedding' field
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings (with progress bar)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding.tolist()
            embedded_chunks.append(chunk_with_embedding)
        
        logger.info(f"Generated {len(embedded_chunks)} embeddings")
        logger.info(f"Embedding dimension: {len(embedded_chunks[0]['embedding'])}")
        logger.info("💰 Cost: $0.00 (completely FREE!)")
        
        return embedded_chunks


if __name__ == "__main__":
    # Test the embedding generator
    import json
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Local Embedding Generator (FREE - No API needed)")
    print("="*60 + "\n")
    
    # Load processed chunks
    chunks_file = Path("data/processed/chunks.json")
    if not chunks_file.exists():
        print("Error: chunks.json not found. Run document processor first.")
        exit(1)
    
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks\n")
    
    # Show available models
    print("Available models:")
    print("  'mini'      - Fast, small, good quality (RECOMMENDED)")
    print("  'mpnet'     - Better quality, slower")
    print("  'bge-small' - Great for technical docs")
    print("  'bge-base'  - Best quality, largest")
    print()
    
    # Generate embeddings for first 5 chunks (test)
    print("Testing with 'mini' model (fastest)...")
    generator = LocalEmbeddingGenerator(model_name="mini")
    
    test_chunks = chunks[:5]
    embedded = generator.embed_chunks(test_chunks)
    
    print(f"\n✅ Success!")
    print(f"Embedding dimensions: {len(embedded[0]['embedding'])}")
    print(f"Sample embedding (first 10 values): {embedded[0]['embedding'][:10]}")
    print(f"\n💰 Total cost: $0.00 (FREE!)")
    print("="*60 + "\n")
