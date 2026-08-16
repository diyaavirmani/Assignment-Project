"""
CLI for querying the 3GPP RAG Assistant end-to-end

Usage:
    # Single question
    python scripts/query.py "What is the gNB-CU architecture?"

    # Interactive mode
    python scripts/query.py

    # Show retrieval sources
    python scripts/query.py --show-sources "How does handover work in 5G?"

    # Stream the answer token-by-token
    python scripts/query.py --stream "Explain the NG-RAN architecture"

    # Filter to a specific spec document
    python scripts/query.py --source 38300 "What are the NR frequency bands?"
"""
import argparse
import sys
import logging
from pathlib import Path

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag_chain import RAGChain
from src.core.vector_store import VectorStore
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the 3GPP RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "--show-sources", "-s",
        action="store_true",
        help="Print retrieved source chunks",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answer token-by-token",
    )
    parser.add_argument(
        "--source",
        help="Filter retrieval to documents containing this string (e.g. '38300')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model to use (default: llama3.2)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def check_prerequisites(chain: RAGChain) -> bool:
    """Verify vector store has data and Ollama is running."""
    store = VectorStore()
    stats = store.get_stats()

    if stats["total_chunks"] == 0:
        print("Vector store is empty. Build the index first:")
        print("  python scripts/build_index.py")
        return False

    print(f"Vector store: {stats['total_chunks']} chunks indexed")

    if not chain.llm.is_available():
        print(f"\nOllama is not running or model '{chain.llm.model}' not available.")
        print("1. Install Ollama: https://ollama.com")
        print("2. Start server:   ollama serve")
        print(f"3. Pull model:     ollama pull {chain.llm.model}")
        return False

    print(f"LLM: {chain.llm.model} (Ollama)")
    return True


def run_query(chain: RAGChain, tracker: MetricsTracker, args, question: str) -> None:
    """Execute one query and print the result."""
    print(f"\nQ: {question}")
    print("-" * 60)

    if args.stream:
        sources_printed = False
        for chunk in chain.stream_query(
            question,
            source_filter=args.source,
            top_k=args.top_k,
        ):
            if chunk["type"] == "sources":
                if args.show_sources and chunk["sources"]:
                    _print_sources(chunk["sources"])
                sources_printed = True
                print("A: ", end="", flush=True)
            elif chunk["type"] == "token":
                print(chunk["token"], end="", flush=True)
            elif chunk["type"] == "done":
                print(f"\n\n[{chunk['query_time']:.2f}s]")
    else:
        result = chain.query(
            question,
            source_filter=args.source,
            top_k=args.top_k,
        )

        print(f"A: {result['answer']}")
        print(
            f"\n[total={result['query_time']}s  "
            f"retrieve={result['retrieve_time']}s  "
            f"generate={result['generate_time']}s]"
        )

        if args.show_sources:
            _print_sources(result["sources"])

        tracker.record(
            query=question,
            retrieve_time=result["retrieve_time"],
            generate_time=result["generate_time"],
            num_sources=len(result["sources"]),
            answer_length=len(result["answer"]),
        )


def _print_sources(sources: list) -> None:
    print(f"\nSources ({len(sources)}):")
    for i, s in enumerate(sources, 1):
        print(f"  {i}. {s['source']}  (similarity={s['similarity']:.3f})")
        if s.get("text"):
            preview = s["text"][:120].replace("\n", " ")
            print(f"     {preview}...")


def interactive_mode(chain: RAGChain, tracker: MetricsTracker, args) -> None:
    """Run a REPL loop for multi-turn conversation."""
    print("\nInteractive mode - type 'quit' or Ctrl-C to exit")
    print("Commands: !clear (reset history)  !stats (show metrics)  !sources (toggle sources)")
    print("=" * 60)

    show_sources = args.show_sources

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if question == "!clear":
            chain.clear_history()
            print("Conversation history cleared.")
            continue

        if question == "!stats":
            tracker.print_summary()
            continue

        if question == "!sources":
            show_sources = not show_sources
            print(f"Sources display: {'ON' if show_sources else 'OFF'}")
            continue

        # Temporarily patch show_sources into args for this call
        original = args.show_sources
        args.show_sources = show_sources
        try:
            run_query(chain, tracker, args, question)
        except RuntimeError as e:
            print(f"\nError: {e}")
        finally:
            args.show_sources = original


def main():
    args = parse_args()

    setup_logger(
        level="DEBUG" if args.verbose else "WARNING"
    )

    from src.core.llm import OllamaLLM
    from src.core.retriever import DocumentRetriever

    llm = OllamaLLM(model=args.model)
    retriever = DocumentRetriever(top_k=args.top_k)
    chain = RAGChain(retriever=retriever, llm=llm)
    tracker = MetricsTracker(persist_path="data/metrics.json")

    print("\n3GPP RAG Assistant")
    print("=" * 60)

    if not check_prerequisites(chain):
        sys.exit(1)

    if args.question:
        try:
            run_query(chain, tracker, args, args.question)
        except RuntimeError as e:
            print(f"\nError: {e}")
            sys.exit(1)
    else:
        interactive_mode(chain, tracker, args)

    if not args.stream and tracker._records:
        tracker.print_summary()


if __name__ == "__main__":
    main()
