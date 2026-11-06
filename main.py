#!/usr/bin/env python3
"""
Main Application Entrypoint
===========================
WSM Assignment: Web Search and Mining Information Retrieval Engine.

Usage:
    # Launch Frontend GUI (Default):
    python main.py

    # Launch with custom corpus directory:
    python main.py --data ./data

    # Run in CLI mode for automated testing or headless environments:
    python main.py --cli --query "retrieval AND index"
    python main.py --cli --query "pro*"
    python main.py --cli --query "\"web mining\""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.document_loader import DocumentLoader
from src.inverted_index import InvertedIndex
from src.normalizer import TextNormalizer
from src.query_processor import QueryProcessor
from src.tokenizer import Tokenizer
from src.wildcard_processor import WildcardProcessor


def run_cli_search(corpus_dir: str, query: str, mode: str = "auto") -> None:
    """Runs a query via command-line interface and outputs results."""
    loader = DocumentLoader()
    tokenizer = Tokenizer()
    normalizer = TextNormalizer()
    index = InvertedIndex(tokenizer=tokenizer, normalizer=normalizer)
    wildcard_proc = WildcardProcessor()
    processor = QueryProcessor(
        index=index,
        tokenizer=tokenizer,
        normalizer=normalizer,
        wildcard_processor=wildcard_proc,
    )

    path = Path(corpus_dir).resolve()
    print(f"[*] Loading documents from: {path}")
    docs = loader.load_directory(path)
    print(f"[*] Loaded {len(docs)} documents.")

    index.build(docs)
    processor.sync_wildcard_index()
    stats = index.get_statistics()
    print(f"[*] Index built: {stats['vocabulary_size']} unique terms, {stats['total_postings']} postings.")

    print(f"\n[?] Query: '{query}' (Mode: {mode})")
    results, desc = processor.search(query, mode=mode)
    print(f"[!] {desc}")
    print(f"[+] Found {len(results)} matching document(s):\n" + "=" * 60)

    for rank, res in enumerate(results, start=1):
        print(f"#{rank} [Doc ID {res.doc_id}] {res.title} (Score: {res.score:.4f})")
        if res.matched_terms:
            print(f"    Matched Terms: {', '.join(res.matched_terms)}")
        print(f"    Snippet: {res.snippet}")
        print("-" * 60)


def run_evaluation_benchmark(corpus_dir: str) -> None:
    """Runs a standard Information Retrieval evaluation benchmark across test queries."""
    from src.evaluator import Evaluator

    loader = DocumentLoader()
    tokenizer = Tokenizer()
    normalizer = TextNormalizer()
    index = InvertedIndex(tokenizer=tokenizer, normalizer=normalizer)
    wildcard_proc = WildcardProcessor()
    processor = QueryProcessor(
        index=index,
        tokenizer=tokenizer,
        normalizer=normalizer,
        wildcard_processor=wildcard_proc,
    )

    path = Path(corpus_dir).resolve()
    print("=" * 70)
    print("  WSM Search Engine - Information Retrieval Benchmark & Evaluation")
    print("=" * 70)
    docs = loader.load_directory(path)
    index.build(docs)
    processor.sync_wildcard_index()
    print(f"[*] Loaded {len(docs)} documents. Vocabulary: {index.vocabulary_size} terms.\n")

    # Benchmark test queries and ground-truth relevant document IDs
    benchmarks = [
        ("inverted index", {1, 3, 6}),
        ("web crawlers spider", {2}),
        ("wildcard queries permuterm", {3}),
        ("tokenization stemming case folding", {4}),
        ("vector space model tf-idf cosine", {5}),
        ("boolean retrieval precision recall", {6}),
    ]

    query_retrieved_lists = []
    query_ground_truths = []

    print(f"{'Query':<36} | {'P@3':<6} | {'R@3':<6} | {'AP':<6} | {'RR':<6}")
    print("-" * 70)

    for query_str, relevant_docs in benchmarks:
        results, _ = processor.search(query_str, mode="ranked")
        retrieved_ids = [r.doc_id for r in results]

        query_retrieved_lists.append(retrieved_ids)
        query_ground_truths.append(relevant_docs)

        p3 = Evaluator.precision_at_k(retrieved_ids, relevant_docs, k=3)
        r3 = Evaluator.recall_at_k(retrieved_ids, relevant_docs, k=3)
        ap = Evaluator.average_precision(retrieved_ids, relevant_docs)
        rr = Evaluator.reciprocal_rank(retrieved_ids, relevant_docs)

        print(f"{query_str:<36} | {p3:<6.3f} | {r3:<6.3f} | {ap:<6.3f} | {rr:<6.3f}")

    map_score = Evaluator.mean_average_precision(query_retrieved_lists, query_ground_truths)
    mrr_score = Evaluator.mean_reciprocal_rank(query_retrieved_lists, query_ground_truths)

    print("-" * 70)
    print(f"Mean Average Precision (MAP) : {map_score:.4f}")
    print(f"Mean Reciprocal Rank (MRR)   : {mrr_score:.4f}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Web Search and Mining (WSM) Information Retrieval Engine")
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=str(Path(__file__).resolve().parent / "data"),
        help="Path to documents folder (default: ./data)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in command-line mode instead of launching GUI",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Query to search in CLI mode",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run Information Retrieval evaluation benchmark (MAP, MRR, P@K, R@K)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="auto",
        choices=["auto", "ranked", "boolean", "phrase", "wildcard"],
        help="Search query evaluation mode",
    )

    args = parser.parse_args()

    if args.eval:
        run_evaluation_benchmark(args.data)
        return

    if args.cli:
        if not args.query:
            print("Error: --query argument is required when running in --cli mode.", file=sys.stderr)
            sys.exit(1)
        run_cli_search(args.data, args.query, args.mode)
    else:
        # Launch GUI
        try:
            from src.gui import launch_gui
            launch_gui(corpus_dir=args.data)
        except Exception as ex:
            print(f"Failed to launch GUI: {ex}", file=sys.stderr)
            print("Falling back to CLI interactive mode or check Tkinter installation.")
            sys.exit(1)


if __name__ == "__main__":
    main()

