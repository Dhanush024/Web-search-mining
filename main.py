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
        "--mode",
        "-m",
        type=str,
        default="auto",
        choices=["auto", "ranked", "boolean", "phrase", "wildcard"],
        help="Search query evaluation mode",
    )

    args = parser.parse_args()

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

